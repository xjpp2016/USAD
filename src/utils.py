import torch
from torch import tensor
from torchvision import transforms

import numpy as np
import PIL
from PIL import ImageFilter
from sklearn import random_projection
from tqdm import tqdm

def get_coreset(
        memory_bank: tensor,
        l: int = 1000,  # Coreset target
        eps: float = 0.09,
) -> tensor:
    """
    Returns l coreset indexes for given memory_bank.

    Args:
    - memory_bank:     Patchcore memory bank tensor
    - l:               Number of patches to select
    - eps:             Sparse Random Projector parameter

    Returns:
    - coreset indexes
    """

    coreset_idx = []  # Returned coreset indexes
    idx = 0

    # Fitting random projections
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        memory_bank = torch.tensor(transformer.fit_transform(memory_bank))
    except ValueError:
        print("Error: could not project vectors. Please increase `eps`.")

    # Coreset subsampling
    print(f'Start Coreset Subsampling...')

    last_item = memory_bank[idx: idx + 1]   # First patch selected = patch on top of memory bank
    coreset_idx.append(torch.tensor(idx))
    min_distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # Norm l2 of distances (tensor)

    # Use GPU if possible
    if torch.cuda.is_available():
        last_item = last_item.to("cuda")
        memory_bank = memory_bank.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(l - 1)):
        distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # L2 norm of distances (tensor)
        min_distances = torch.minimum(distances, min_distances)                         # Vertical tensor of minimum norms
        idx = torch.argmax(min_distances)                                               # Index of maximum related to the minimum of norms

        last_item = memory_bank[idx: idx + 1]   # last_item = maximum patch just found
        min_distances[idx] = 0                  # Zeroing last_item distances
        coreset_idx.append(idx.to("cpu"))       # Save idx inside the coreset

    return torch.stack(coreset_idx)


def gaussian_blur(img: tensor) -> tensor:
    """
    Apply a gaussian smoothing with sigma = 4 over the input image.
    """
    # Setup
    blur_kernel = ImageFilter.GaussianBlur(radius=4)
    tensor_to_pil = transforms.ToPILImage()
    pil_to_tensor = transforms.ToTensor()

    device = img.device

    # Smoothing
    max_value = img.max()   # Maximum value of all elements in the image tensor
    blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
    blurred_tensor = pil_to_tensor(blurred_pil)
    blurred_tensor = blurred_tensor.to(device)
    blurred_map =  blurred_tensor * max_value

    return blurred_map


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        
    return PIL.Image.fromarray(tensor)


def set_seed(seed: int = 42):
    """
    Fix all random seeds to ensure experiment reproducibility
    
    Args:
        seed: Random seed, defaults to 42
    """
    import os
    import random

    # Python built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    
    # PyTorch CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables (optional)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds fixed to {seed}")


def compute_pro(amaps, masks, num_th=200):
    """
    Compute PRO (Per-Region Overlap) AUC
    Args:
        masks: ground truth masks (numpy array, shape: [N, H, W])
        amaps: anomaly maps (numpy array, shape: [N, H, W])
        num_th: number of thresholds
    Returns:
        pro_auc: PRO AUC score
    """
    import pandas as pd
    from skimage import measure
    from statistics import mean
    from sklearn.metrics import auc

    # Ensure inputs are numpy arrays
    masks = np.array(masks)
    amaps = np.array(amaps)
    
    # Ensure masks are binary
    if masks.max() > 1:
        masks = (masks > 128).astype(np.uint8)
    
    # Normalize anomaly maps to range [0, 1]
    if amaps.max() > 1:
        amaps = amaps.astype(np.float32) / 255.0
    else:
        amaps = amaps.astype(np.float32)
    
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    
    print(f"Computing PRO AUC... Threshold range: {min_th:.4f} to {max_th:.4f}")
    
    for th in tqdm(np.arange(min_th, max_th, delta), desc="Scanning thresholds"):
        # Binarize anomaly map
        binary_amaps = np.zeros_like(amaps)
        binary_amaps[amaps > th] = 1
        
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            # Find connected regions in mask
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)
            
            for region in regions:
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
        
        # Skip if no regions detected at current threshold
        if len(pros) == 0:
            continue
        
        # Compute false positive rate
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / (inverse_masks.sum() + 1e-10)  # Avoid division by zero
        
        # Add to dataframe
        new_row = pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Check if there is valid data
    if len(df) == 0:
        print("Warning: No valid PRO data, returning 0")
        return 0.0
    
    # Normalize FPR from 0~1 to 0~0.3
    df = df[df["fpr"] < 0.3]
    if len(df) == 0:
        print("Warning: No data points with FPR < 0.3, returning 0")
        return 0.0
    
    # Ensure maximum FPR is not 0
    fpr_max = df["fpr"].max()
    if fpr_max == 0:
        print("Warning: Maximum FPR is 0, cannot normalize")
        return 0.0
    
    df["fpr"] = df["fpr"] / fpr_max
    
    # Sort by FPR
    df = df.sort_values("fpr")
    
    # Compute AUC
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc