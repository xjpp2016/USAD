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


        # print(f"Net Used: {backbone}")

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
            Initialize self.features and let the input sample passing
            throught the backbone net self.model.
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
            Creates memory bank from train dataset and apply greedy coreset subsampling.
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
            patch = patch.reshape(patch.shape[1], -1).T   # Craete a column tensor

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

            score, segm_map = self.predict(sample)  # Anomaly Detection

            image_preds.append(score.numpy())
            pixel_preds.extend(segm_map.flatten().numpy())

            # 保存分割图为灰度图像
            segm_map_np = segm_map.squeeze().numpy()

            # 归一化到0-255
            segm_map_255 = (segm_map_np * 255).astype(np.uint8)
            
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

    
    def predict(self, sample: tensor):
            """
            对测试样本进行异常检测
            主要步骤：
            1. 创建测试样本的局部感知块特征
            2. 通过比较测试块与记忆库中最近邻块的差异，计算图像级异常检测分数
            3. 根据各自的空间位置重新排列计算出的块异常分数，生成分割图。
            然后通过双线性插值上采样分割图，并使用高斯模糊平滑结果
            
            参数：
            - sample: 测试样本
            
            返回：
            - 分割分数（异常得分）
            - 分割图
            """
            
            # ========== 块特征提取 ==========
            # 通过神经网络前向传播获取特征图
            feature_maps = self(sample)
            
            # 创建平均池化层，用于平滑特征图（3x3窗口，步长为1）
            self.avg = torch.nn.AvgPool2d(3, stride=1)
            
            # 获取第一个特征图的高度和宽度（假设所有特征图尺寸相同）
            fmap_size = feature_maps[0].shape[-2]
            
            # 创建自适应平均池化层，将所有特征图调整到相同尺寸
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
            
            # 对每个特征图进行平滑和尺寸调整
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            
            # 在通道维度上拼接所有处理后的特征图
            patch = torch.cat(resized_maps, 1)
            
            # 将特征图重构为块表示：形状变为 (块数量, 块特征维度)
            # 其中块数量 = 特征图高度 × 特征图宽度
            patch = patch.reshape(patch.shape[1], -1).T

            # ========== 计算最大距离分数 s*（论文中的公式6） ==========
            # 计算测试样本中每个块与记忆库中所有块的L2距离
            device = patch.device
            self.memory_bank = self.memory_bank.to(device)
            distances = torch.cdist(patch, self.memory_bank, p=2.0)
            
            # 找到每个测试块到记忆库的最小距离及其索引
            dist_score, dist_score_idxs = torch.min(distances, dim=1)
            
            # 找到所有最小距离中的最大值（最异常的块）
            s_idx = torch.argmax(dist_score)                                # 异常候选块的索引
            s_star = torch.max(dist_score)                                  # 最大距离分数 s*
            m_test_star = torch.unsqueeze(patch[s_idx], dim=0)              # 异常候选块特征
            m_star = self.memory_bank[dist_score_idxs[s_idx]].unsqueeze(0)  # 记忆库中与异常候选块最接近的邻居块

            # ========== K近邻搜索 ==========
            # 计算异常候选块的最近邻居在记忆库中的距离
            knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)
            
            # 找到k个最近邻居的索引（排除自身，所以从第1个开始）
            _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)

            # ========== 计算图像级异常分数 s ==========
            # 获取异常候选块的邻居块（排除第一个，因为第一个是自身）
            m_star_neighbourhood = self.memory_bank[nn_idxs[0, 1:]]
            
            # 计算异常候选块与其邻居块的L2距离（公式7的分母部分）
            w_denominator = torch.linalg.norm(m_test_star - m_star_neighbourhood, dim=1)
            
            # 计算归一化因子，防止指数运算溢出
            norm = torch.sqrt(torch.tensor(patch.shape[1]))
            
            # 计算权重w（论文中的公式7）
            w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
            
            # 计算最终的图像级异常分数
            s = w * s_star

            # ========== 生成分割图 ==========
            # 获取特征图的完整尺寸（高度和宽度）
            fmap_size = feature_maps[0].shape[-2:]
            
            # 将距离分数重新组织为分割图格式 (1, 1, 高度, 宽度)
            segm_map = dist_score.view(1, 1, *fmap_size)
            segm_map = (segm_map - segm_map.min())/(segm_map.max() - segm_map.min() + 1e-8)
            
            # 使用双线性插值将分割图上采样到原始输入图像尺寸
            segm_map = torch.nn.functional.interpolate(
                            segm_map,
                            size=(self.image_size, self.image_size),
                            mode='bilinear'
                        )
            
            # 对分割图应用高斯模糊进行平滑处理
            segm_map = gaussian_blur(segm_map)

            return s, segm_map
    
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
