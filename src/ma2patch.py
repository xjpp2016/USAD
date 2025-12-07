import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
            gaussian_sigma: float = 4.0,  # 添加高斯滤波参数
    ):
        super(DINOv2AnomalyDetector, self).__init__()
        
        self.model_name = model_name
        self.clusters = clusters
        self.img_size = img_size
        self.gaussian_sigma = gaussian_sigma
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化特征提取器
        self.feature_extractor = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            img_size=self.img_size
        ).to(self.device)
        
        self.feature_extractor.eval()

    def _extract_features(self, image_tensor):
        """提取特征"""
        with torch.no_grad():
            outputs = self.feature_extractor.forward_features(image_tensor)
            patch_tokens = outputs[:, 1:, :]
            cls_tokens = outputs[:, :1, :]

        return patch_tokens[0], cls_tokens[0]

    def train(self, train_dataloader: DataLoader, scale: int = 1) -> None:
        """
        训练阶段：只构建CLS token记忆库
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

        # 合并所有CLS tokens
        self.memory_bank_cls = torch.cat(cls_tokens_list, 0)  # [num_samples, hidden_dim]
        # 计算记忆库的统计量用于马氏距离（基于CLS tokens）
        self._compute_mahalanobis_params()
        print(f"Memory bank built with {self.memory_bank_cls.shape[0]} CLS tokens")

    def _compute_mahalanobis_params(self):
        """
        计算CLS tokens马氏距离所需的均值和协方差矩阵
        """
        if self.memory_bank_cls is None:
            raise ValueError("Memory bank is not built yet. Call fit() first.")
        
        # 计算均值向量
        self.mean_vector = torch.mean(self.memory_bank_cls, dim=0)
        
        # 计算协方差矩阵
        centered_data = self.memory_bank_cls - self.mean_vector
        self.cov_matrix = torch.mm(centered_data.T, centered_data) / (self.memory_bank_cls.shape[0] - 1)
        
        # 计算协方差矩阵的逆（添加小量避免奇异矩阵）
        reg_matrix = torch.eye(self.cov_matrix.shape[0]) * 1e-6
        reg_matrix = reg_matrix.to(self.device)
        self.inv_cov_matrix = torch.inverse(self.cov_matrix + reg_matrix)
        
        print(f"Computed Mahalanobis parameters: mean shape {self.mean_vector.shape}, cov shape {self.cov_matrix.shape}")

    def _mahalanobis_distance_batch(self, test_tokens):
        """
        批量计算马氏距离（使用CLS tokens的统计量）
        """
        if self.mean_vector is None or self.inv_cov_matrix is None:
            raise ValueError("Mahalanobis parameters not computed. Call fit() first.")
        
        # 批量计算差值
        diff = test_tokens - self.mean_vector.unsqueeze(0)
        
        # 批量计算马氏距离: sqrt((x-μ)^T Σ^(-1) (x-μ))
        temp = torch.einsum('bi,ij->bj', diff, self.inv_cov_matrix)
        mahal_dists = torch.sqrt(torch.einsum('bi,bi->b', temp, diff))
        
        return mahal_dists
    
    def _compute_patch_contributions_ablation(self, patch_tokens: tensor):
        """
        基于消融实验的贡献度计算 - 数学优化版本
        """
        patch_tokens = patch_tokens.to(self.device)
        num_patches, hidden_dim = patch_tokens.shape
        
        # 计算完整的CLS token
        full_cls = torch.mean(patch_tokens, dim=0, keepdim=True)  # (1, hidden_dim)
        full_mahal_distance = self._mahalanobis_distance_batch(full_cls).item()
        
        # 数学推导：移除第i个patch后的CLS = (n*full_cls - patch_i) / (n-1)
        # 其中 n = num_patches
        n = num_patches
        approximated_cls = (n * full_cls - patch_tokens) / (n - 1)  # (num_patches, hidden_dim)
        
        # 批量计算马氏距离
        approx_mahal_distances = self._mahalanobis_distance_batch(approximated_cls)
        
        # 计算贡献度
        contributions = torch.abs(full_mahal_distance - approx_mahal_distances).cpu().numpy()
        
        return contributions

    def predict(self, sample: tensor):
        """
        对测试样本进行异常检测
        
        参数：
        - sample: 测试样本tensor [B, C, H, W]
        
        返回：
        - 图像级异常分数
        - 像素级异常图
        """
        batch_size = sample.shape[0]
        if batch_size > 1:
            raise ValueError("目前只支持单样本预测")
        
        # 将样本移动到设备
        sample = sample.to(self.device)
        
        # 调整图像尺寸用于聚类特征提取
        original_size = sample.shape[2:]  # [H, W]
        
        # 处理尺寸
        if max(original_size) > self.img_size:
            scale = self.img_size / max(original_size)
            new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
            sample_resized = torch.nn.functional.interpolate(
                sample, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            sample_resized = sample
            new_h, new_w = original_size
        
        # 提取特征
        patch_tokens, _ = self._extract_features(sample)
        cls_tokens = torch.mean(patch_tokens, dim=0, keepdim=True)

        # 使用CLS token的马氏距离作为图像级异常分数
        cls_mahal_distances = self._mahalanobis_distance_batch(cls_tokens)
        image_anomaly_score = cls_mahal_distances[0]  # 假设batch_size=1

        # 计算patch贡献度
        patch_contributions = self._compute_patch_contributions_ablation(patch_tokens)
        # # 将贡献度转换为分割图
        batch_size = sample.shape[0]
        num_patches = patch_contributions.shape[0]
        patch_map_size = int(np.sqrt(num_patches))
        
        # # 重塑为网格格式
        segm_map = patch_contributions.reshape(patch_map_size, patch_map_size)
        segm_map = cv2.resize(segm_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

        segm_map = gaussian_filter(segm_map, sigma=self.gaussian_sigma)
        
        return image_anomaly_score, segm_map

    def evaluate(self, test_dataloader: DataLoader, save_dir: str = "result/dinov2_anomaly/"):
        """
        计算异常检测分数和相对分割图
        返回ROC AUC计算得到的预测分数

        Args:
            test_dataloader: 测试数据加载器
            save_dir: 保存结果的目录路径

        Returns:
            - image-level ROC-AUC score
            - pixel-level ROC-AUC score
        """

        # 创建保存目录
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

            score, segm_map = self.predict(sample)

            image_preds.append(score.detach().cpu().numpy())
            pixel_preds.extend(segm_map.flatten())

            # 归一化到0-255
            segm_map_255 = ((segm_map - segm_map.min()) / 
                                (segm_map.max() - segm_map.min()) * 255).astype(np.uint8)
            # segm_map_255 = (segm_map * 255).astype(np.uint8)
            
            seg_filename = os.path.join(save_dir, "segmentation_maps", f"{idx:04d}_seg_map.png")
            cv2.imwrite(seg_filename, segm_map_255)

            # 生成并保存热力图
            heatmap = cv2.applyColorMap(segm_map_255, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(os.path.join(save_dir, "heatmaps", f"{idx:04d}_heatmap.png"), heatmap)
            
            # 生成并保存叠加图
            sample = self._denormalize_tensor(sample)
            sample_np = (sample[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sample_bgr = cv2.cvtColor(sample_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(sample_bgr, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(os.path.join(save_dir, "heatmaps", f"{idx:04d}_overlay.png"), overlay)

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)

        # 计算ROC AUC
        image_level_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_level_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        # 保存结果
        with open(os.path.join(save_dir, "results.txt"), "w") as f:
            f.write(f"Image-level ROC-AUC: {image_level_rocauc:.4f}\n")
            f.write(f"Pixel-level ROC-AUC: {pixel_level_rocauc:.4f}\n")

        print(f"Results saved to {save_dir}")
        print(f"Image-level ROC-AUC: {image_level_rocauc:.4f}")
        print(f"Pixel-level ROC-AUC: {pixel_level_rocauc:.4f}")

        return image_level_rocauc, pixel_level_rocauc
    
    def _denormalize_tensor(self, tensor):
        """
        将ImageNet归一化的tensor反归一化到[0,1]范围
        """
        # 确保mean和std的维度正确

        mean = IMAGENET_MEAN.clone().detach().view(-1, 1, 1).to(tensor.device)
        std = IMAGENET_STD.clone().detach().view(-1, 1, 1).to(tensor.device)
        
        # 反归一化
        tensor = tensor * std + mean
        
        # 裁剪到[0,1]范围
        return torch.clamp(tensor, 0, 1)
    
    def save_ma_memory_bank(self, file_path: str):
        """
        保存记忆库和相关参数到文件
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
        从文件加载记忆库和相关参数
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
