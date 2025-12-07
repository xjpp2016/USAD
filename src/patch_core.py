import os
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from .utils import gaussian_blur, get_coreset

from src.data import IMAGENET_MEAN, IMAGENET_STD

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class PatchCore(torch.nn.Module):
    def __init__(
            self,
            f_coreset:float = 0.01,    # Fraction rate of training samples
            eps_coreset: float = 0.90, # SparseProjector parameter
            k_nearest: int = 3,        # k parameter for K-NN search
            backbone: str = 'wide_resnet50_2',
            image_size: int = 224
    ):
        assert f_coreset > 0
        assert eps_coreset > 0
        assert k_nearest > 0
        assert image_size > 0

        super(PatchCore, self).__init__()

        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        # Register hooks
        self.model = torch.hub.load('pytorch/vision:v0.13.0', backbone, pretrained=True)
        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook)            

        # Disable gradient computation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters
        self.memory_bank = []
        self.f_coreset = f_coreset      # Fraction rate of training samples
        self.eps_coreset = eps_coreset  # SparseProjector parameter
        self.k_nearest = k_nearest      # k parameter for K-NN search
        self.backbone = backbone
        self.image_size = image_size


    def forward(self, sample: tensor):
        """
        Initialize self.features and let the input sample pass through
        the backbone net self.model.
        The registered hooks will extract the layer 2 and 3 feature maps.
        Return:
            self.feature filled with extracted feature maps
        """

        self.features = []
        _ = self.model(sample)

        return self.features


    def fit(self, train_dataloader: DataLoader, scale: int=1) -> None:

        """
        Training phase
        Creates memory bank from train dataset and applies greedy coreset subsampling.
        """
        tot = len(train_dataloader) // scale
        counter = 0

        for sample, _ in tqdm(train_dataloader, total=tot):
            feature_maps = self(sample)                   # Extract feature maps

            # Create aggregation function of feature vectors in the neighbourhood
            self.avg = torch.nn.AvgPool2d(3, stride=1)
            fmap_size = feature_maps[0].shape[-2]         # Feature map sizes h, w
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

            # Create patch
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps
            patch = patch.reshape(patch.shape[1], -1).T   # Create a column tensor

            self.memory_bank.append(patch)                # Fill memory bank
            counter += 1
            if counter > tot:
                break

        self.memory_bank = torch.cat(self.memory_bank, 0) # VStack the patches

        # Coreset subsampling
        if self.f_coreset < 1:
            coreset_idx = get_coreset(
                self.memory_bank,
                l = int(self.f_coreset * self.memory_bank.shape[0]),
                eps = self.eps_coreset
            )
            self.memory_bank = self.memory_bank[coreset_idx]


    def evaluate(self, test_dataloader: DataLoader, save_dir: str = "result/patchcore/"):
        """
        Compute anomaly detection score and relative segmentation map for
        each test sample. Returns the ROC AUC computed from predictions scores.

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

        for idx, (sample, mask, label) in enumerate(tqdm(test_dataloader)):
            image_labels.append(label)
            pixel_labels.extend(mask.flatten().numpy())

            score, segm_map = self.predict(sample)  # Anomaly Detection

            image_preds.append(score.numpy())
            pixel_preds.extend(segm_map.flatten().numpy())

            # Save segmentation map as grayscale image
            segm_map_np = segm_map.squeeze().numpy()

            # Normalize to 0-255
            segm_map_255 = (segm_map_np * 255).astype(np.uint8)
            
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

    
    def predict(self, sample: tensor):
            """
            Perform anomaly detection on test sample
            Main steps:
            1. Create local-aware patch features for test sample
            2. Compute image-level anomaly detection score by comparing test patches
               with nearest neighbor patches in memory bank
            3. Generate segmentation map by rearranging computed patch anomaly scores
               according to their spatial positions.
            The segmentation map is then upsampled with bilinear interpolation
            and smoothed with Gaussian blur
            
            Args:
            - sample: Test sample
            
            Returns:
            - Segmentation score (anomaly score)
            - Segmentation map
            """
            
            # ========== Patch Feature Extraction ==========
            # Get feature maps through neural network forward pass
            feature_maps = self(sample)
            
            # Create average pooling layer for smoothing feature maps (3x3 window, stride 1)
            self.avg = torch.nn.AvgPool2d(3, stride=1)
            
            # Get height and width of first feature map (assuming all feature maps have same size)
            fmap_size = feature_maps[0].shape[-2]
            
            # Create adaptive average pooling layer to adjust all feature maps to same size
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
            
            # Smooth and resize each feature map
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            
            # Concatenate all processed feature maps along channel dimension
            patch = torch.cat(resized_maps, 1)
            
            # Reshape feature map to patch representation: shape becomes (number of patches, patch feature dimension)
            # where number of patches = feature map height × feature map width
            patch = patch.reshape(patch.shape[1], -1).T

            # ========== Compute Maximum Distance Score s* (Equation 6 in the paper) ==========
            # Compute L2 distance between each patch in test sample and all patches in memory bank
            device = patch.device
            self.memory_bank = self.memory_bank.to(device)
            distances = torch.cdist(patch, self.memory_bank, p=2.0)
            
            # Find minimum distance for each test patch to memory bank and its indices
            dist_score, dist_score_idxs = torch.min(distances, dim=1)
            
            # Find maximum value among all minimum distances (most anomalous patch)
            s_idx = torch.argmax(dist_score)                                # Index of anomaly candidate patch
            s_star = torch.max(dist_score)                                  # Maximum distance score s*
            m_test_star = torch.unsqueeze(patch[s_idx], dim=0)              # Anomaly candidate patch feature
            m_star = self.memory_bank[dist_score_idxs[s_idx]].unsqueeze(0)  # Nearest neighbor patch in memory bank to anomaly candidate patch

            # ========== K-Nearest Neighbor Search ==========
            # Compute distances of anomaly candidate patch's nearest neighbors in memory bank
            knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)
            
            # Find indices of k nearest neighbors (skip self, so start from 1st)
            _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)

            # ========== Compute Image-Level Anomaly Score s ==========
            # Get neighbor patches of anomaly candidate patch (skip first one as it's self)
            m_star_neighbourhood = self.memory_bank[nn_idxs[0, 1:]]
            
            # Compute L2 distance between anomaly candidate patch and its neighbor patches (denominator of Equation 7)
            w_denominator = torch.linalg.norm(m_test_star - m_star_neighbourhood, dim=1)
            
            # Compute normalization factor to prevent exponential overflow
            norm = torch.sqrt(torch.tensor(patch.shape[1]))
            
            # Compute weight w (Equation 7 in the paper)
            w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
            
            # Compute final image-level anomaly score
            s = w * s_star

            # ========== Generate Segmentation Map ==========
            # Get full dimensions of feature map (height and width)
            fmap_size = feature_maps[0].shape[-2:]
            
            # Reorganize distance scores into segmentation map format (1, 1, height, width)
            segm_map = dist_score.view(1, 1, *fmap_size)
            segm_map = (segm_map - segm_map.min())/(segm_map.max() - segm_map.min() + 1e-8)
            
            # Upsample segmentation map to original input image size using bilinear interpolation
            segm_map = torch.nn.functional.interpolate(
                            segm_map,
                            size=(self.image_size, self.image_size),
                            mode='bilinear'
                        )
            
            # Apply Gaussian blur to segmentation map for smoothing
            segm_map = gaussian_blur(segm_map)

            return s, segm_map
    
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
    
    def save_memory_bank(self, file_path: str):
        """
        Save the memory bank to a file.
        
        Args:
            file_path: Path to save the memory bank
        """
        if self.memory_bank is None:
            raise ValueError("Memory bank is empty. Please run fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save memory bank and model configuration
        save_dict = {
            'memory_bank': self.memory_bank,
            'f_coreset': self.f_coreset,
            'eps_coreset': self.eps_coreset,
            'k_nearest': self.k_nearest,
            'backbone': self.backbone,
            'image_size': self.image_size
        }
        
        torch.save(save_dict, file_path)
        print(f"Patch_core Memory bank saved to {file_path}")


    def load_memory_bank(self, file_path: str):
        """
        Load the memory bank from a file.
        
        Args:
            file_path: Path to load the memory bank from
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Memory bank file not found: {file_path}")
        
        # Load saved data
        saved_data = torch.load(file_path)
        
        # Set memory bank
        self.memory_bank = saved_data['memory_bank']
        
        # Verify configuration matches (optional, but recommended)
        config_mismatch = []
        if saved_data.get('f_coreset', self.f_coreset) != self.f_coreset:
            config_mismatch.append(f"f_coreset: {saved_data['f_coreset']} vs {self.f_coreset}")
        if saved_data.get('eps_coreset', self.eps_coreset) != self.eps_coreset:
            config_mismatch.append(f"eps_coreset: {saved_data['eps_coreset']} vs {self.eps_coreset}")
        if saved_data.get('backbone', self.backbone) != self.backbone:
            config_mismatch.append(f"backbone: {saved_data['backbone']} vs {self.backbone}")
        
        if config_mismatch:
            print(f"Warning: Configuration mismatch detected: {', '.join(config_mismatch)}")
            print("This may affect model performance.")
        
        #print(f"Patch_core Memory bank loaded from {file_path} with {self.memory_bank.shape[0]} patches")