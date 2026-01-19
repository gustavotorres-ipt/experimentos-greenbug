import torch
import random
import torchvision.transforms.functional as TF

def add_gaussian_noise(batch, std=0.01):
    noise = torch.randn_like(batch) * std
    noisy_images = batch + noise
    # Optional: clamp to keep pixel values in [0,1]
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images

# ------------------------------
# 2. Time Masking (mask columns)
# ------------------------------
def time_mask(batch, max_width=20):
    B, C, H, W = batch.shape
    for i in range(B):
        t = random.randint(0, max_width)
        t0 = random.randint(0, max(1, W - t))
        batch[i, :, :, t0:t0+t] = 0
    return batch

# ------------------------------
# 3. Frequency Masking (mask rows)
# ------------------------------
def frequency_mask(batch, max_width=10):
    B, C, H, W = batch.shape
    for i in range(B):
        f = random.randint(0, max_width)
        f0 = random.randint(0, max(1, H - f))
        batch[i, :, f0:f0+f, :] = 0
    return batch

# ------------------------------
# 4. Small Shifts
# ------------------------------
def random_shift(batch, shift_limit=0.1):
    B, C, H, W = batch.shape
    shifted = torch.zeros_like(batch)
    for i in range(B):
        dx = int(random.uniform(-shift_limit, shift_limit) * W)
        dy = int(random.uniform(-shift_limit, shift_limit) * H)
        shifted[i] = TF.affine(batch[i], angle=0, translate=(dx, dy), scale=1.0, shear=0)
    return shifted

# ------------------------------
# 5. Brightness & Contrast
# ------------------------------
def random_brightness_contrast(batch, brightness=0.2, contrast=0.2):
    B, C, H, W = batch.shape
    adjusted = torch.zeros_like(batch)
    for i in range(B):
        b_factor = 1.0 + random.uniform(-brightness, brightness)
        c_factor = 1.0 + random.uniform(-contrast, contrast)
        img = batch[i]
        img = TF.adjust_brightness(img, b_factor)
        img = TF.adjust_contrast(img, c_factor)
        adjusted[i] = img
    return adjusted

# ------------------------------
# 6. Full Augmentation Pipeline
# ------------------------------
def augment_batch(batch):
    batch = add_gaussian_noise(batch, std=0.01)
    batch = time_mask(batch, max_width=20)
    batch = frequency_mask(batch, max_width=10)
    batch = random_shift(batch, shift_limit=0.05)
    batch = random_brightness_contrast(batch, brightness=0.1, contrast=0.1)
    return batch
