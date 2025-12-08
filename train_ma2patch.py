import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # set the HuggingFace endpoint to mirror.com
from src.data import CoalDataset, coal_classes
from src.ma2patch import DINOv2AnomalyDetector
from src.utils import set_seed

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


ALL_CLASSES = coal_classes()
VIT_SIZE = 224
RESULT_DIR = 'result'
SAVED_MODELS_DIR = 'trained_models'

def run_model(
        classes: list = ALL_CLASSES,
        backbone: str = "vit_base_patch14_dinov2.lvd142m",
        train_data_ratio: float = 1.0
) -> None:

    size = VIT_SIZE
    results = {}# key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]

    print(f'Running PatchCore...')
    for cls in classes:
        print(f'\nClass {cls}:')

        train_dl, test_dl = CoalDataset(cls, size=size, train_data_ratio=train_data_ratio).get_dataloaders()

        # Train
        model_dir = f'{SAVED_MODELS_DIR}/{cls}'
        os.makedirs(model_dir, exist_ok=True) 
        m2ptch_mem_bank_path = f'{model_dir}/ma2ptch_memory_bank.pkl'

        detector = DINOv2AnomalyDetector(model_name= backbone)
        detector.train(train_dl)
        detector.save_ma_memory_bank(m2ptch_mem_bank_path)

        # Test
        detector_test = DINOv2AnomalyDetector(model_name= backbone)
        detector_test.load_ma_memory_bank("./trained_models/ma2ptch_memory_bank.pkl")

        os.makedirs(f'{RESULT_DIR}/{cls}', exist_ok=True)
        image_rocauc, pixel_rocauc = detector_test.evaluate(test_dl, f'{RESULT_DIR}/{cls}/ma2ptch')

        print(f'Results:')
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        print(f'- Image-level ROC AUC = {image_rocauc:.3f}')
        print(f'- Iixel-level ROC AUC = {pixel_rocauc:.3f}\n')

    # Save global results and statistics
    image_results = [v[0] for k, v in results.items()]
    average_image_rocauc = sum(image_results) / len(image_results)
    pixel_resuts = [v[1] for k, v in results.items()]
    average_pixel_rocauc = sum(pixel_resuts) / len(pixel_resuts)

    print(f'- Average image-level ROC AUC = {average_image_rocauc:.3f}\n')
    print(f'- Average pixel-level ROC AUC = {average_pixel_rocauc:.3f}\n')


if __name__ == "__main__":
    set_seed(22)
    run_model()
