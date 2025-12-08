from src.data import CoalDataset, coal_classes
from src.patch_core import PatchCore
from src.utils import set_seed
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


ALL_CLASSES = coal_classes()
PC_SIZE = 224
RESULT_DIR = 'result'
SAVED_MODELS_DIR = 'trained_models'

def run_model(
        classes: list = ALL_CLASSES,
        train_data_ratio: float = 0.5
) -> None:

    f_coreset = 0.01
    size = PC_SIZE
    results = {}# key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]

    print(f'Running PatchCore...')
    for cls in classes:
        print(f'\nClass {cls}:')

        train_dl, test_dl = CoalDataset(cls, size=size, train_data_ratio=train_data_ratio).get_dataloaders()
        patch_core = PatchCore(f_coreset, image_size=size)

        print(f'Training...')
        model_dir = f'{SAVED_MODELS_DIR}/{cls}'
        os.makedirs(model_dir, exist_ok=True)
        pc_memory_bank_path = f'{model_dir}/pc_memory_bank.pth'

        patch_core.fit(train_dl)

        # 保存memory_bank
        patch_core.save_memory_bank(pc_memory_bank_path)

        # 加载memory_bank
        patch_core.load_memory_bank(pc_memory_bank_path)

        print(f'Testing...')
        os.makedirs(f'{RESULT_DIR}/{cls}', exist_ok=True)
        image_rocauc, pixel_rocauc = patch_core.evaluate(test_dl, f'{RESULT_DIR}/{cls}/pc')

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
