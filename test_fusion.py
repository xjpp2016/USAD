from src.data import CoalDataset, coal_classes
from src.ma_clu import DINOv2AnomalyDetector
from src.ma2patch import DINOv2AnomalyDetector as DINOv2AnomalyDetector_ma2patch
from src.patch_core import PatchCore
from src.fusion import CombinedAnomalyDetector
from src.utils import set_seed
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Define all coal classes
ALL_CLASSES = coal_classes()
# Define image sizes for different models
CLU_SIZE = 336
M2P_SIZE = 224
PC_SIZE = 224
# Strong foreground threshold
STR_F_THRESH = 1.13
# Result directory
RESULT_DIR = 'result'
SAVED_MODELS_DIR = 'saved_models'

def run_combined_model(
        classes: list = ALL_CLASSES,
        threshold: float = STR_F_THRESH,
        clu_backbone: str = 'vit_small_patch14_dinov2.lvd142m',
        m2p_backbone: str = 'vit_base_patch14_dinov2.lvd142m',

) -> None:
    """
    Run combined anomaly detection model
    
    Args:
        classes: List of classes to process, defaults to all coal types
        threshold: Strong foreground threshold
        clu_backbone: Backbone network for CLU detector
        m2p_backbone: Backbone network for MA2Patch detector
    """

    results = {}  # key = class, Value = dictionary containing pixel-level and image-level ROC AUC

    print(f'Running Combined Anomaly Detection...')
    for cls in classes:
        print(f'\nClass {cls}:')

        # Get data loader (select based on maximum size)
        if CLU_SIZE >= M2P_SIZE and CLU_SIZE >= PC_SIZE:
            _, test_dl = CoalDataset(cls=cls, size=CLU_SIZE).get_dataloaders()
        elif M2P_SIZE >= PC_SIZE:  
            _, test_dl = CoalDataset(cls=cls, size=M2P_SIZE).get_dataloaders()
        else:
            _, test_dl = CoalDataset(cls=cls, size=PC_SIZE).get_dataloaders()

        # Initialize three detectors

        ma_memory_bank_path = f'{SAVED_MODELS_DIR}/{cls}/ma_memory_bank.pkl'
        confidence_model_path = f'{SAVED_MODELS_DIR}/{cls}/confidence_model.pkl'
        ma2ptch_memory_bank_path = f'{SAVED_MODELS_DIR}/{cls}/ma2ptch_memory_bank.pkl'
        pc_memory_bank_path = f'{SAVED_MODELS_DIR}/{cls}/pc_memory_bank.pth'


        detector_clu = DINOv2AnomalyDetector(model_name=clu_backbone, img_size=CLU_SIZE, strong_foreground_threshold=threshold)
        detector_clu.load_ma_memory_bank(ma_memory_bank_path)
        detector_clu.load_clu_model(confidence_model_path)

        detector_m2p = DINOv2AnomalyDetector_ma2patch(model_name=m2p_backbone, img_size=M2P_SIZE)
        detector_m2p.load_ma_memory_bank(ma2ptch_memory_bank_path)

        detector_pc = PatchCore(image_size=PC_SIZE)
        detector_pc.load_memory_bank(pc_memory_bank_path)
        
        print('Initializing combined detector...')
        # Initialize combined detector
        combined_detector = CombinedAnomalyDetector(
            detector_clu=detector_clu,
            detector_m2p=detector_m2p,
            detector_pc=detector_pc,  
            device=None  # Auto-select device
        )

        print(f'Testing combined model...')
        os.makedirs(f'{RESULT_DIR}/{cls}', exist_ok=True)
        # Evaluate test set - now returns results containing pixel-level and image-level
        evaluation_results = combined_detector.evaluate(
            test_dl, 
            save_dir=f'{RESULT_DIR}/{cls}/combined', 
        )

        # Save results to file
        tex_filename = f'{RESULT_DIR}/{cls}/combined/result.txt'
        with open(tex_filename, 'a') as f:  # Use append mode
            f.write(f'Class: {cls}\n')
            results[cls] = evaluation_results
            
            # Write combined results
            f.write(f'- Combined Image-level ROC AUC = {evaluation_results["combined_image_rocauc"]:.3f}\n')
            f.write(f'- Combined Pixel-level ROC AUC = {evaluation_results["combined_pixel_rocauc"]:.3f}\n')
            
        print(f'Results for {cls}:')
        print(f'- Combined Image-level ROC AUC = {evaluation_results["combined_image_rocauc"]:.3f}')
        print(f'- Combined Pixel-level ROC AUC = {evaluation_results["combined_pixel_rocauc"]:.3f}')
        print()
    


if __name__ == "__main__":
    # Set random seed for reproducible results
    set_seed(22)
    run_combined_model(threshold=STR_F_THRESH)