import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score

from src.data import IMAGENET_MEAN, IMAGENET_STD

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class CombinedAnomalyDetector:
    def __init__(self, detector_clu, detector_m2p, detector_pc, bias=0.01, device=None):
        """
        合并两个异常检测器的类
        
        参数:
        - detector_clu: 第一个检测器 (VitBackbone, 物体级)
        - detector_m2p: 第二个检测器 (VitBackbone, 整体语义)
        - detector_pc: 第三个检测器 (PatchCore, 纹理细节)
        - device: 计算设备
        """
        self.detector_clu = detector_clu
        self.detector_m2p = detector_m2p
        self.detector_pc = detector_pc
        self.bias = bias
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


        self.clu_imgsize = detector_clu.img_size
        self.m2p_imgsize = detector_m2p.img_size
        self.pc_imgsize = detector_pc.image_size

        self.clu_resize = transforms.Resize(self.clu_imgsize)
        self.m2p_resize = transforms.Resize(self.m2p_imgsize)
        self.pc_resize = transforms.Resize(self.pc_imgsize)

        # 将模型移动到设备
        self._move_models_to_device()
    
    def _move_models_to_device(self):
        """将模型移动到指定设备"""
        if hasattr(self.detector_clu, 'device'):
            self.detector_clu.device = self.device
        if hasattr(self.detector_clu, 'feature_extractor'):
            self.detector_clu.feature_extractor = self.detector_clu.feature_extractor.to(self.device)

        if hasattr(self.detector_m2p, 'device'):
            self.detector_m2p.device = self.device
        if hasattr(self.detector_m2p, 'feature_extractor'):
            self.detector_m2p.feature_extractor = self.detector_m2p.feature_extractor.to(self.device)
  
        if hasattr(self.detector_pc, 'model'):
            self.detector_pc.model = self.detector_pc.model.to(self.device)
            if hasattr(self.detector_pc, 'memory_bank') and self.detector_pc.memory_bank is not None:
                self.detector_pc.memory_bank = self.detector_pc.memory_bank.to(self.device)
    
    def _ensure_4d_tensor(self, tensor):
        """确保张量为4维 [B, C, H, W]"""
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            return tensor.unsqueeze(0) if tensor.size(0) != 1 else tensor.unsqueeze(0)
        return tensor
    
    def _resize_to_target(self, tensor, target_size):
        """调整张量尺寸到目标大小"""
        if tensor.shape[2:] != target_size:
            return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return tensor
    
    def predict(self, sample):
        """
        对测试样本进行异常检测，合并两个模型的分割结果
        
        参数:
        - sample: 测试样本tensor [B, C, H, W]
        - return_individual: 是否返回单独检测器的结果
        
        返回:
        - 合并后的图像级异常分数
        - 合并后的像素级异常图
        - 如果return_individual为True，还返回各个检测器的结果和分数
        """
        if sample.shape[0] > 1:
            raise ValueError("目前只支持单样本预测")
        
        sample = sample.to(self.device)
        original_size = sample.shape[2:]  # [H, W]
        
        individual_results = {}

        max_imgsize = max(self.clu_imgsize, self.m2p_imgsize, self.pc_imgsize)

        clu_sample = sample if self.clu_imgsize == max_imgsize else self.clu_resize(sample)
        m2p_sample = sample if self.m2p_imgsize == max_imgsize else self.m2p_resize(sample)
        pc_sample = sample if self.pc_imgsize == max_imgsize else self.pc_resize(sample)
            
        # 获取第一个检测器的结果
        with torch.no_grad():
            score_clu, segm_map_clu = self.detector_clu.predict(clu_sample)
            segm_map_clu = torch.from_numpy(segm_map_clu)
            segm_map_clu = segm_map_clu.to(self.device)
            
            individual_results['detector_clu'] = {
                'score': score_clu,
                'segmentation_map': segm_map_clu.clone()
            }
        
        # 获取第二个检测器的结果
        with torch.no_grad():
            score_m2p, segm_map_m2p = self.detector_m2p.predict(m2p_sample)
            segm_map_m2p = torch.from_numpy(segm_map_m2p)
            segm_map_m2p = segm_map_m2p.to(self.device)
            
            individual_results['detector_m2p'] = {
                'score': score_m2p,
                'segmentation_map': segm_map_m2p.clone()
            }


        # 获取第三个检测器的结果
        with torch.no_grad():
            score_pc, segm_map_pc = self.detector_pc.predict(pc_sample)
            segm_map_pc = segm_map_pc.to(self.device)
            
            individual_results['detector_pc'] = {
                'score': score_pc,
                'segmentation_map': segm_map_pc.clone()
            }
        
        # 计算合并后的图像级分数
        combined_image_score = self.bias * score_clu + score_pc

        # 统一分辨率到原始图像尺寸
        segm_map_clu = self._ensure_4d_tensor(segm_map_clu)
        segm_map_m2p = self._ensure_4d_tensor(segm_map_m2p)
        segm_map_pc = self._ensure_4d_tensor(segm_map_pc)
        
        segm_map_clu_resized = self._resize_to_target(segm_map_clu, original_size)
        segm_map_m2p_resized = self._resize_to_target(segm_map_m2p, original_size)
        segm_map_pc_resized = self._resize_to_target(segm_map_pc, original_size)
        
    
        # Hadamard乘积合并分割图
        combined_segm_map = segm_map_clu_resized * segm_map_pc_resized * segm_map_m2p_resized
        
        # 对合并后的分割图进行后处理
        combined_segm_map_np = combined_segm_map.squeeze().cpu().numpy()
        
        if combined_segm_map_np.ndim > 2:
            combined_segm_map_np = combined_segm_map_np.squeeze()
            
        combined_segm_map_smoothed = cv2.GaussianBlur(combined_segm_map_np, (5, 5), 1.0)
        combined_segm_map = torch.from_numpy(combined_segm_map_smoothed).unsqueeze(0).unsqueeze(0).float()
        
        return combined_image_score, combined_segm_map, individual_results
    
    def evaluate(self, test_dataloader, save_dir="result/combined_anomaly/"):
        """
        评估合并模型（包含像素级和图像级评估）
        
        参数:
        - test_dataloader: 测试数据加载器
        - save_dir: 保存结果的目录路径
        
        返回:
        - 包含像素级和图像级ROC-AUC分数的字典
        """
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "segmentation_maps"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "heatmaps"), exist_ok=True)
        
        # 初始化存储列表
        pixel_preds = []
        pixel_labels = []
        image_preds = []
        image_labels = []

        for idx, (sample, mask, label) in enumerate(tqdm(test_dataloader)):
            sample = sample.to(self.device)
            
            # 收集像素级和图像级标签
            pixel_labels.extend(mask.flatten().numpy())
            image_labels.extend(label.flatten().numpy())

            # 获取合并结果
            combined_image_score, combined_segm_map, individual_results = self.predict(sample)

            # 收集合并检测器的预测结果
            pixel_preds.extend(combined_segm_map.flatten().cpu().numpy())
            image_preds.append(combined_image_score.cpu().numpy())

            # 保存分割图和热力图
            self._save_segmentation_map(combined_segm_map, os.path.join(save_dir, "segmentation_maps", f"{idx:04d}_seg_map.png"))
            self._save_heatmap(combined_segm_map, sample, os.path.join(save_dir, "heatmaps", f"{idx:04d}_heatmap.png"), os.path.join(save_dir, "heatmaps", f"{idx:04d}_overlay.png"))

        # 计算图像级ROC AUC
        image_level_rocauc = roc_auc_score(image_labels, image_preds)

        # 计算像素级ROC AUC
        pixel_level_rocauc = roc_auc_score(pixel_labels, pixel_preds)
        
        results = {
            'combined_image_rocauc': image_level_rocauc,
            'combined_pixel_rocauc': pixel_level_rocauc,
        }

        # 保存结果
        self._save_results(results, save_dir)

        print(f"Results saved to {save_dir}")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

        return results
    
    def _save_segmentation_map(self, segm_map, filename):
        """保存分割图为灰度图像"""
        segm_map_np = segm_map.squeeze().cpu().numpy()
        segm_map_normalized = ((segm_map_np - segm_map_np.min()) / 
                            (segm_map_np.max() - segm_map_np.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(filename, segm_map_normalized)

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
    
    def _save_heatmap(self, segm_map, sample, heatmap_filenam, overlay_filename):
        """生成并保存热力图"""
        segm_map_np = segm_map.squeeze().cpu().numpy()
        segm_map_normalized = ((segm_map_np - segm_map_np.min()) / 
                            (segm_map_np.max() - segm_map_np.min() + 1e-8) * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(segm_map_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(heatmap_filenam, heatmap)
        
        # 生成叠加图
        sample = self._denormalize_tensor(sample)
        sample_np = (sample[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sample_bgr = cv2.cvtColor(sample_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(sample_bgr, 0.5, heatmap, 0.5, 0)
        cv2.imwrite(overlay_filename, overlay)
    
    def _save_results(self, results, save_dir):
        """保存结果到文件"""
        with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
            f.write("=== Pixel-level Results ===\n")
            for key, value in results.items():
                if 'pixel' in key:
                    f.write(f"{key}: {value:.4f}\n")
            
            f.write("\n=== Image-level Results ===\n")
            for key, value in results.items():
                if 'image' in key:
                    f.write(f"{key}: {value:.4f}\n")