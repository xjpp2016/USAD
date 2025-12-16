import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # Set HuggingFace endpoint to mirror site
import torch
from torch import tensor
from torch.utils.data import DataLoader
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import timm
from scipy.ndimage import gaussian_filter

from src.data import IMAGENET_MEAN, IMAGENET_STD

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class DINOv2AnomalyDetector(torch.nn.Module):
    def __init__(
            self,
            model_name: str = "vit_base_patch14_dinov2.lvd142m",
            clusters: int = 2,
            img_size: int = 224,
            gaussian_sigma: float = 4.0,  # Add Gaussian filter parameter
    ):
        super(DINOv2AnomalyDetector, self).__init__()
        
        self.model_name = model_name
        self.clusters = clusters
        self.img_size = img_size
        self.gaussian_sigma = gaussian_sigma
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize feature extractor
        self.feature_extractor = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            img_size=self.img_size
        ).to(self.device)
        
        self.feature_extractor.eval()

    def _extract_features(self, image_tensor):
        """Extract features"""
        with torch.no_grad():
            outputs = self.feature_extractor.forward_features(image_tensor)
            patch_tokens = outputs[:, 1:, :]
            cls_tokens = outputs[:, :1, :]

        return patch_tokens[0], cls_tokens[0]

    def train(self, train_dataloader: DataLoader, scale: int = 1) -> None:
        """
        Training phase: Build only CLS token memory bank
        """
        tot = len(train_dataloader) // scale
        counter = 0

        cls_tokens_list = []
        patch_tokens_list = []

        for sample, _ in tqdm(train_dataloader, total=tot):
            sample = sample.to(self.device)
            patch_tokens, _ = self._extract_features(sample)
            cls_tokens = torch.mean(patch_tokens, dim=0, keepdim=True)
            cls_tokens_list.append(cls_tokens)
            patch_tokens_list.append(patch_tokens)
            
            counter += 1
            if counter > tot:
                break

        # Combine all CLS tokens
        self.memory_bank_cls = torch.cat(cls_tokens_list, 0)  # [num_samples, hidden_dim]
        # Compute memory bank statistics for Mahalanobis distance (based on CLS tokens)
        self._compute_mahalanobis_params()
        print(f"Memory bank built with {self.memory_bank_cls.shape[0]} CLS tokens")

    def _compute_mahalanobis_params(self):
        """
        Compute mean and covariance matrix required for Mahalanobis distance of CLS tokens
        """
        if self.memory_bank_cls is None:
            raise ValueError("Memory bank is not built yet. Call fit() first.")
        
        # Compute mean vector
        self.mean_vector = torch.mean(self.memory_bank_cls, dim=0)
        
        # Compute covariance matrix
        centered_data = self.memory_bank_cls - self.mean_vector
        self.cov_matrix = torch.mm(centered_data.T, centered_data) / (self.memory_bank_cls.shape[0] - 1)
        
        # Compute inverse of covariance matrix (add small value to avoid singular matrix)
        reg_matrix = torch.eye(self.cov_matrix.shape[0]) * 1e-6
        reg_matrix = reg_matrix.to(self.device)
        self.inv_cov_matrix = torch.inverse(self.cov_matrix + reg_matrix)
        
        print(f"Computed Mahalanobis parameters: mean shape {self.mean_vector.shape}, cov shape {self.cov_matrix.shape}")

    def _mahalanobis_distance_batch(self, test_tokens):
        """
        Compute Mahalanobis distance in batch (using statistics of CLS tokens)
        """
        if self.mean_vector is None or self.inv_cov_matrix is None:
            raise ValueError("Mahalanobis parameters not computed. Call fit() first.")
        
        diff = test_tokens - self.mean_vector.unsqueeze(0)
        
        try:
        # Compute Mahalanobis distance in batch: sqrt((x-μ)^T Σ^(-1) (x-μ)) 
            temp = torch.einsum('bi,ij->bj', diff, self.inv_cov_matrix)
            quad_form = torch.einsum('bi,bi->b', temp, diff)
        except:
            # If it fails, use diagonal approximation
            if not hasattr(self, 'variances'):
                self.variances = torch.diag(self.cov_matrix)
                self.inv_variances = 1.0 / torch.clamp(self.variances, min=1e-6)
            
            quad_form = torch.sum(diff**2 * self.inv_variances.unsqueeze(0), dim=1)
        
        # Ensure non-negative
        quad_form = torch.clamp(quad_form, min=0.0)
        mahal_dists = torch.sqrt(quad_form + 1e-12)
        
        return mahal_dists
    
    def _compute_patch_contributions_ablation(self, patch_tokens: tensor):
        """
        Contribution calculation based on ablation experiment - mathematical optimization version
        """
        patch_tokens = patch_tokens.to(self.device)
        num_patches, hidden_dim = patch_tokens.shape
        
        # Compute complete CLS token
        full_cls = torch.mean(patch_tokens, dim=0, keepdim=True)  # (1, hidden_dim)
        full_mahal_distance = self._mahalanobis_distance_batch(full_cls).item()
        
        # Mathematical derivation: CLS after removing i-th patch = (n*full_cls - patch_i) / (n-1)
        # where n = num_patches
        n = num_patches
        approximated_cls = (n * full_cls - patch_tokens) / (n - 1)  # (num_patches, hidden_dim)
        
        # Compute Mahalanobis distance in batch
        approx_mahal_distances = self._mahalanobis_distance_batch(approximated_cls)
        
        # Compute contributions
        contributions = torch.abs(full_mahal_distance - approx_mahal_distances).cpu().numpy()
        
        return contributions

    def predict(self, sample: tensor):
        """
        Perform anomaly detection on test sample
        
        Args:
        - sample: Test sample tensor [B, C, H, W]
        
        Returns:
        - Image-level anomaly score
        - Pixel-level anomaly map
        """
        batch_size = sample.shape[0]
        if batch_size > 1:
            raise ValueError("Currently only supports single sample prediction")
        
        # Move sample to device
        sample = sample.to(self.device)
        
        # Adjust image size for clustering feature extraction
        original_size = sample.shape[2:]  # [H, W]
        
        # Process size
        if max(original_size) > self.img_size:
            scale = self.img_size / max(original_size)
            new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
            sample_resized = torch.nn.functional.interpolate(
                sample, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            sample_resized = sample
            new_h, new_w = original_size
        
        # Extract features
        patch_tokens, _ = self._extract_features(sample)
        cls_tokens = torch.mean(patch_tokens, dim=0, keepdim=True)

        # Use Mahalanobis distance of CLS token as image-level anomaly score
        cls_mahal_distances = self._mahalanobis_distance_batch(cls_tokens)
        image_anomaly_score = cls_mahal_distances[0]  # Assuming batch_size=1

        # Compute patch contributions
        patch_contributions = self._compute_patch_contributions_ablation(patch_tokens)
        # Convert contributions to segmentation map
        batch_size = sample.shape[0]
        num_patches = patch_contributions.shape[0]
        patch_map_size = int(np.sqrt(num_patches))
        
        # Reshape to grid format
        segm_map = patch_contributions.reshape(patch_map_size, patch_map_size)
        segm_map = cv2.resize(segm_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

        segm_map = gaussian_filter(segm_map, sigma=self.gaussian_sigma)
        
        return image_anomaly_score, segm_map

    def evaluate(self, test_dataloader: DataLoader, save_dir: str = "result/dinov2_anomaly/", cal_pro: bool = False):
        """
        Compute anomaly detection score and relative segmentation map
        Returns ROC AUC computed from prediction scores

        Args:
            test_dataloader: Test data loader
            save_dir: Path to save results

        Returns:
            - image-level ROC-AUC score
            - pixel-level ROC-AUC score
        """

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "segmentation_maps"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "heatmaps"), exist_ok=True)
        
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []
        pixel_pro_preds = []
        pixel_pro_labels = []

        for idx, (sample, mask, label) in enumerate(tqdm(test_dataloader)):
            image_labels.append(label)
            pixel_labels.extend(mask.flatten().numpy())

            score, segm_map = self.predict(sample)

            image_preds.append(score.detach().cpu().numpy())
            pixel_preds.extend(segm_map.flatten())

            # Normalize to 0-255
            if segm_map.max() - segm_map.min() > 1e-8:
                segm_map_255 = ((segm_map - segm_map.min()) / 
                                    (segm_map.max() - segm_map.min()) * 255).astype(np.uint8)
            else:
                segm_map_255 = (segm_map * 255).astype(np.uint8)
            
            seg_filename = os.path.join(save_dir, "segmentation_maps", f"{idx:04d}_seg_map.png")
            cv2.imwrite(seg_filename, segm_map_255)

            # Generate and save heatmap
            heatmap = cv2.applyColorMap(segm_map_255, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(os.path.join(save_dir, "heatmaps", f"{idx:04d}_heatmap.png"), heatmap)
            
            # Generate and save overlay image
            sample = self._denormalize_tensor(sample)
            sample_np = (sample[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sample_bgr = cv2.cvtColor(sample_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(sample_bgr, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(os.path.join(save_dir, "heatmaps", f"{idx:04d}_overlay.png"), overlay)

            if label == 1 and cal_pro:
                segm_map_np = segm_map.squeeze()
                segm_map_normalized = ((segm_map_np - segm_map_np.min()) / 
                                    (segm_map_np.max() - segm_map_np.min() + 1e-8))
                pixel_pro_preds.append(segm_map_normalized)
                pixel_pro_labels.append(mask.squeeze().cpu().numpy())

        # Compute pixel-level PRO AUC
        if cal_pro:
            from src.utils import compute_pro
            pixel_pro_preds = np.array(pixel_pro_preds)
            pixel_pro_labels = np.array(pixel_pro_labels)
            pixel_pro_auc = compute_pro(pixel_pro_preds, pixel_pro_labels)
            print(f"pixel_pro_auc: {pixel_pro_auc:.4f}")

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)

        # Compute ROC AUC
        image_level_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_level_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        # Save results
        with open(os.path.join(save_dir, "results.txt"), "w") as f:
            f.write(f"Image-level ROC-AUC: {image_level_rocauc:.4f}\n")
            f.write(f"Pixel-level ROC-AUC: {pixel_level_rocauc:.4f}\n")

        print(f"Results saved to {save_dir}")
        print(f"Image-level ROC-AUC: {image_level_rocauc:.4f}")
        print(f"Pixel-level ROC-AUC: {pixel_level_rocauc:.4f}")

        return image_level_rocauc, pixel_level_rocauc
    
    def _denormalize_tensor(self, tensor):
        """
        Denormalize ImageNet-normalized tensor back to [0,1] range
        """
        # Ensure mean and std have correct dimensions

        mean = IMAGENET_MEAN.clone().detach().view(-1, 1, 1).to(tensor.device)
        std = IMAGENET_STD.clone().detach().view(-1, 1, 1).to(tensor.device)
        
        # Denormalize
        tensor = tensor * std + mean
        
        # Clip to [0,1] range
        return torch.clamp(tensor, 0, 1)
    
    def save_ma_memory_bank(self, file_path: str):
        """
        Save memory bank and related parameters to file
        """
        save_data = {
            'memory_bank_cls': self.memory_bank_cls,
            'mean_vector': self.mean_vector,
            'cov_matrix': self.cov_matrix,
            'inv_cov_matrix': self.inv_cov_matrix,
        }
        
        torch.save(save_data, file_path)
        print(f"Memory bank saved to {file_path}")

    def load_ma_memory_bank(self, file_path: str):
        """
        Load memory bank and related parameters from file
        """
        load_data = torch.load(file_path, map_location='cpu')
        
        self.memory_bank_cls = load_data['memory_bank_cls']
        self.mean_vector = load_data['mean_vector']
        self.cov_matrix = load_data['cov_matrix']
        self.inv_cov_matrix = load_data['inv_cov_matrix']

        self.memory_bank_cls = self.memory_bank_cls.to(self.device)
        self.mean_vector = self.mean_vector.to(self.device)
        self.cov_matrix = self.cov_matrix.to(self.device)
        self.inv_cov_matrix = self.inv_cov_matrix.to(self.device)
        
        print(f"Memory bank loaded from {file_path}")
        print(f"Loaded {self.memory_bank_cls.shape[0]} CLS tokens")