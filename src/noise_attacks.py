"""
Noise Attack Resilience Tests — Task T7
=========================================
Tests the encryption scheme's resilience against various noise attacks
applied to the ciphertext before decryption.

Attack types tested:
  1. Gaussian noise — additive white Gaussian noise
  2. Salt & pepper noise — random pixel corruption
  3. Speckle noise — multiplicative noise
  4. JPEG compression — lossy compression artifacts
  5. Cropping attack — data loss (already in evaluation.py as occlusion)
  6. Filtering attack — low-pass/median filtering of ciphertext

For each attack, we:
  1. Encrypt the original image
  2. Apply the noise attack to the ciphertext
  3. Decrypt the attacked ciphertext
  4. Measure PSNR, SSIM, and MSE between original and recovered image

Uses scikit-image for noise generation and SSIM computation.
"""

import numpy as np
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter, median_filter

from .encryption import encrypt, decrypt, EncryptionKey
from .evaluation import compute_mse, compute_psnr


# ─────────────────────────────────────────────────────────────────────────────
# Noise attack functions
# ─────────────────────────────────────────────────────────────────────────────

def apply_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Apply additive Gaussian noise to the image.

    Parameters
    ----------
    image : ndarray, dtype uint8
    sigma : float
        Standard deviation of Gaussian noise (in pixel value units 0-255).

    Returns
    -------
    noisy : ndarray, dtype uint8
    """
    # skimage random_noise expects float in [0,1] and variance in [0,1]
    variance = (sigma / 255.0) ** 2
    noisy = random_noise(image / 255.0, mode='gaussian', var=variance, clip=True)
    return (noisy * 255).astype(np.uint8)


def apply_salt_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """
    Apply salt-and-pepper noise.

    Parameters
    ----------
    image : ndarray, dtype uint8
    amount : float
        Fraction of pixels to corrupt (0 to 1).
    """
    noisy = random_noise(image / 255.0, mode='s&p', amount=amount, clip=True)
    return (noisy * 255).astype(np.uint8)


def apply_speckle_noise(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Apply multiplicative speckle noise.

    Parameters
    ----------
    image : ndarray, dtype uint8
    sigma : float
        Standard deviation of multiplicative noise.
    """
    variance = sigma ** 2
    noisy = random_noise(image / 255.0, mode='speckle', var=variance, clip=True)
    return (noisy * 255).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Simulate JPEG compression attack by saving/loading with quality factor.

    Parameters
    ----------
    image : ndarray, dtype uint8
    quality : int
        JPEG quality factor (1-100). Lower = more compression/artifacts.
    """
    from PIL import Image
    import io

    # Save to JPEG buffer
    pil_img = Image.fromarray(image, mode='L')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    # Reload from buffer
    compressed = np.array(Image.open(buffer).convert('L'), dtype=np.uint8)
    return compressed


def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply Gaussian blur (low-pass filtering attack).

    Parameters
    ----------
    image : ndarray, dtype uint8
    sigma : float
        Standard deviation of Gaussian kernel.
    """
    blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def apply_median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply median filtering attack.

    Parameters
    ----------
    image : ndarray, dtype uint8
    size : int
        Size of the median filter kernel.
    """
    filtered = median_filter(image.astype(np.float64), size=size)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_cropping_attack(image: np.ndarray, crop_percent: float = 25.0) -> np.ndarray:
    """
    Apply cropping attack by zeroing out a portion of the image.

    Parameters
    ----------
    image : ndarray, dtype uint8
    crop_percent : float
        Percentage of image area to zero out.
    """
    H, W = image.shape
    cropped = image.copy()
    crop_rows = int(H * np.sqrt(crop_percent / 100))
    crop_cols = int(W * np.sqrt(crop_percent / 100))
    cropped[:crop_rows, :crop_cols] = 0
    return cropped


# ─────────────────────────────────────────────────────────────────────────────
# SSIM computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Returns
    -------
    ssim_value : float
        SSIM value in [-1, 1]. 1.0 = identical.
    """
    return ssim(original, recovered, data_range=255)


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive noise attack test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_noise_attack_suite(
    original: np.ndarray,
    key: EncryptionKey = None,
    verbose: bool = True
) -> dict:
    """
    Run the full noise attack test suite on an image.

    Encrypts the image, applies various noise attacks to the ciphertext,
    decrypts the attacked ciphertext, and measures recovery quality.

    Parameters
    ----------
    original : ndarray, shape (H, W), dtype uint8
        Original plaintext image.
    key : EncryptionKey, optional
        Encryption key.
    verbose : bool
        Print results for each attack.

    Returns
    -------
    results : dict
        Results for each attack type with PSNR, SSIM, MSE.
    """
    if key is None:
        key = EncryptionKey()

    # Encrypt the original image
    cipher, key, _ = encrypt(original, key)

    # Verify clean decryption works
    clean_decrypted, _ = decrypt(cipher, key)
    assert np.array_equal(original, clean_decrypted), "Clean decryption failed!"

    # Define attack suite
    attacks = {
        'gaussian_sigma10': ('Gaussian (sigma=10)', lambda img: apply_gaussian_noise(img, sigma=10)),
        'gaussian_sigma25': ('Gaussian (sigma=25)', lambda img: apply_gaussian_noise(img, sigma=25)),
        'gaussian_sigma50': ('Gaussian (sigma=50)', lambda img: apply_gaussian_noise(img, sigma=50)),
        'salt_pepper_1pct': ('Salt & Pepper (1%)', lambda img: apply_salt_pepper_noise(img, amount=0.01)),
        'salt_pepper_5pct': ('Salt & Pepper (5%)', lambda img: apply_salt_pepper_noise(img, amount=0.05)),
        'salt_pepper_10pct': ('Salt & Pepper (10%)', lambda img: apply_salt_pepper_noise(img, amount=0.10)),
        'speckle_01': ('Speckle (sigma=0.1)', lambda img: apply_speckle_noise(img, sigma=0.1)),
        'speckle_05': ('Speckle (sigma=0.5)', lambda img: apply_speckle_noise(img, sigma=0.5)),
        'jpeg_q50': ('JPEG (Q=50)', lambda img: apply_jpeg_compression(img, quality=50)),
        'jpeg_q20': ('JPEG (Q=20)', lambda img: apply_jpeg_compression(img, quality=20)),
        'jpeg_q10': ('JPEG (Q=10)', lambda img: apply_jpeg_compression(img, quality=10)),
        'blur_sigma1': ('Gaussian Blur (sigma=1)', lambda img: apply_gaussian_blur(img, sigma=1.0)),
        'blur_sigma2': ('Gaussian Blur (sigma=2)', lambda img: apply_gaussian_blur(img, sigma=2.0)),
        'median_3x3': ('Median Filter (3x3)', lambda img: apply_median_filter(img, size=3)),
        'crop_10pct': ('Cropping (10%)', lambda img: apply_cropping_attack(img, crop_percent=10)),
        'crop_25pct': ('Cropping (25%)', lambda img: apply_cropping_attack(img, crop_percent=25)),
        'crop_50pct': ('Cropping (50%)', lambda img: apply_cropping_attack(img, crop_percent=50)),
    }

    results = {}

    if verbose:
        print(f"\n{'Attack Type':<30} {'PSNR (dB)':>10} {'SSIM':>8} {'MSE':>10}")
        print("-" * 62)

    for attack_key, (attack_name, attack_fn) in attacks.items():
        try:
            # Apply attack to ciphertext
            attacked_cipher = attack_fn(cipher)

            # Decrypt the attacked ciphertext
            recovered, _ = decrypt(attacked_cipher, key)

            # Compute quality metrics
            mse_val = compute_mse(original, recovered)
            psnr_val = compute_psnr(original, recovered)
            ssim_val = compute_ssim(original, recovered)

            results[attack_key] = {
                'name': attack_name,
                'mse': float(mse_val),
                'psnr': float(psnr_val),
                'ssim': float(ssim_val),
                'attacked_cipher': attacked_cipher,
                'recovered': recovered,
            }

            if verbose:
                print(f"  {attack_name:<28} {psnr_val:>10.2f} {ssim_val:>8.4f} {mse_val:>10.1f}")

        except Exception as e:
            results[attack_key] = {
                'name': attack_name,
                'error': str(e),
            }
            if verbose:
                print(f"  {attack_name:<28} {'ERROR':>10} - {str(e)}")

    return results


def run_noise_attack_comparison(
    original: np.ndarray,
    key: EncryptionKey = None,
) -> dict:
    """
    Run a simplified noise attack comparison suitable for plotting.

    Returns a structured dict with attack categories and parameter sweeps
    for generating comparison charts.
    """
    if key is None:
        key = EncryptionKey()

    cipher, key, _ = encrypt(original, key)

    comparison = {}

    # Gaussian noise sweep
    gaussian_results = []
    for sigma in [5, 10, 15, 20, 25, 30, 40, 50]:
        attacked = apply_gaussian_noise(cipher, sigma=sigma)
        recovered, _ = decrypt(attacked, key)
        gaussian_results.append({
            'sigma': sigma,
            'psnr': float(compute_psnr(original, recovered)),
            'ssim': float(compute_ssim(original, recovered)),
        })
    comparison['gaussian_sweep'] = gaussian_results

    # Salt & pepper sweep
    sp_results = []
    for amount in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        attacked = apply_salt_pepper_noise(cipher, amount=amount)
        recovered, _ = decrypt(attacked, key)
        sp_results.append({
            'amount_pct': amount * 100,
            'psnr': float(compute_psnr(original, recovered)),
            'ssim': float(compute_ssim(original, recovered)),
        })
    comparison['salt_pepper_sweep'] = sp_results

    # JPEG quality sweep
    jpeg_results = []
    for quality in [5, 10, 20, 30, 50, 70, 90]:
        attacked = apply_jpeg_compression(cipher, quality=quality)
        recovered, _ = decrypt(attacked, key)
        jpeg_results.append({
            'quality': quality,
            'psnr': float(compute_psnr(original, recovered)),
            'ssim': float(compute_ssim(original, recovered)),
        })
    comparison['jpeg_sweep'] = jpeg_results

    # Cropping sweep
    crop_results = []
    for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        attacked = apply_cropping_attack(cipher, crop_percent=pct)
        recovered, _ = decrypt(attacked, key)
        crop_results.append({
            'crop_pct': pct,
            'psnr': float(compute_psnr(original, recovered)),
            'ssim': float(compute_ssim(original, recovered)),
        })
    comparison['cropping_sweep'] = crop_results

    return comparison
