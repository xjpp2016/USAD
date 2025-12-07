from src.data import CoalDataset, coal_classes
from src.ma_clu import DINOv2AnomalyDetector
from src.ma2patch import DINOv2AnomalyDetector as DINOv2AnomalyDetector_ma2patch
from src.patch_core import PatchCore
from src.fusion import CombinedAnomalyDetector
from src.utils import set_seed
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ALL_CLASSES = coal_classes()
CLU_SIZE = 336
M2P_SIZE = 224
PC_SIZE = 224
STR_F_THRESH = 1.13
RESULT_DIR = 'result/'

def run_combined_model(
        classes: list = ALL_CLASSES,
        threshold: float = STR_F_THRESH,
        clu_backbone: str = 'vit_small_patch14_dinov2.lvd142m',
        m2p_backbone: str = 'vit_base_patch14_dinov2.lvd142m',

) -> None:

    results = {}  # key = class, Value = 包含像素级和图像级ROC AUC的字典

    print(f'Running Combined Anomaly Detection...')
    for cls in classes:
        print(f'\nClass {cls}:')

        # 获取数据加载器
        if CLU_SIZE >= M2P_SIZE and CLU_SIZE >= PC_SIZE:
            _, test_dl = CoalDataset(cls=cls, size=CLU_SIZE).get_dataloaders()
        elif M2P_SIZE >= PC_SIZE:  
            _, test_dl = CoalDataset(cls=cls, size=M2P_SIZE).get_dataloaders()
        else:
            _, test_dl = CoalDataset(cls=cls, size=PC_SIZE).get_dataloaders()

        # 初始化三个检测器
        detector_clu = DINOv2AnomalyDetector(model_name= clu_backbone, img_size=CLU_SIZE, strong_foreground_threshold=threshold)
        detector_clu.load_ma_memory_bank("./saved_models/ma_memory_bank.pkl")
        detector_clu.load_clu_model("./saved_models/confidence_model.pkl")

        detector_m2p = DINOv2AnomalyDetector_ma2patch(model_name=m2p_backbone, img_size=M2P_SIZE)
        detector_m2p.load_ma_memory_bank("./saved_models/ma2ptch_memory_bank.pkl")

        detector_pc = PatchCore(image_size=PC_SIZE)
        detector_pc.load_memory_bank("saved_models/pc_memory_bank.pth")
        
        print('Initializing combined detector...')
        # 初始化合并检测器
        combined_detector = CombinedAnomalyDetector(
            detector_clu=detector_clu,
            detector_m2p=detector_m2p,
            detector_pc=detector_pc,  
            device=None  # 自动选择设备
        )

        print(f'Testing combined model...')
        # 创建目录（如果不存在）
        os.makedirs(f'{RESULT_DIR}/{cls}', exist_ok=True)
        # 评估测试集 - 现在返回包含像素级和图像级的结果
        evaluation_results = combined_detector.evaluate(
            test_dl, 
            save_dir=f'{RESULT_DIR}/{cls}/combined', 
        )

        # 保存结果到文件
        tex_filename = f'{RESULT_DIR}/{cls}/combined/result.txt'
        with open(tex_filename, 'a') as f:  # 使用追加模式
            f.write(f'Class: {cls}\n')
            results[cls] = evaluation_results
            
            # 写入合并结果
            f.write(f'- Combined Image-level ROC AUC = {evaluation_results["combined_image_rocauc"]:.3f}\n')
            f.write(f'- Combined Pixel-level ROC AUC = {evaluation_results["combined_pixel_rocauc"]:.3f}\n')
            
        print(f'Results for {cls}:')
        print(f'- Combined Image-level ROC AUC = {evaluation_results["combined_image_rocauc"]:.3f}')
        print(f'- Combined Pixel-level ROC AUC = {evaluation_results["combined_pixel_rocauc"]:.3f}')
        print()
    


if __name__ == "__main__":
    set_seed(22)
    run_combined_model(threshold=STR_F_THRESH)