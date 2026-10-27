"""
Security Evaluation Metrics
============================
Implements the evaluation metrics from Section V of the paper:

1. Histogram Analysis — visual comparison of plaintext vs ciphertext histograms
2. Correlation Analysis — adjacent pixel correlation (Eq. 33)
3. Entropy Analysis — Shannon entropy (Eq. 34-36), ideal = 8.0 for 8-bit
4. NPCR — Number of Pixels Change Rate (Eq. 37), ideal ≈ 99.6%
5. UACI — Unified Average Changing Intensity, ideal ≈ 33.46%
6. Key Sensitivity — effect of tiny key changes on ciphertext
7. Occlusion Attack Resilience — recovery after data loss
8. Execution Time — timing analysis
"""

import numpy as np
from typing import Optional
import os


def compute_entropy(image: np.ndarray) -> float:
    """
    Compute Shannon entropy of an image (Eq. 34).
    
    H(X) = -Σ p(x) * log2(p(x))
    
    Ideal for 8-bit grayscale: 8.0 bits/pixel.
    """
    # Calculate histogram (probability distribution)
    hist = np.histogram(image, bins=256, range=(0, 255))[0]
    hist = hist.astype(np.float64)
    total = hist.sum()

    if total == 0:
        return 0.0

    probs = hist / total
    # Remove zero entries to avoid log2(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def compute_correlation(image: np.ndarray, direction: str = 'horizontal',
                        n_samples: int = 5000, seed: int = 42) -> float:
    """
    Compute adjacent pixel correlation coefficient (Eq. 33).
    
    Parameters
    ----------
    image : ndarray, shape (H, W)
        Input image.
    direction : str
        'horizontal', 'vertical', or 'diagonal'.
    n_samples : int
        Number of random pixel pairs to sample.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    corr : float
        Pearson correlation coefficient. Ideal for cipher: ≈ 0.
    """
    rng = np.random.RandomState(seed)
    H, W = image.shape

    if direction == 'horizontal':
        max_x = W - 1
        max_y = H
        dy, dx = 0, 1
    elif direction == 'vertical':
        max_x = W
        max_y = H - 1
        dy, dx = 1, 0
    elif direction == 'diagonal':
        max_x = W - 1
        max_y = H - 1
        dy, dx = 1, 1
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Sample random pixel positions
    n_samples = min(n_samples, max_x * max_y)
    ys = rng.randint(0, max_y, n_samples)
    xs = rng.randint(0, max_x, n_samples)

    p1 = image[ys, xs].astype(np.float64)
    p2 = image[ys + dy, xs + dx].astype(np.float64)

    # Pearson correlation coefficient
    mean1, mean2 = p1.mean(), p2.mean()
    diff1 = p1 - mean1
    diff2 = p2 - mean2

    numerator = np.sum(diff1 * diff2)
    denominator = np.sqrt(np.sum(diff1 ** 2) * np.sum(diff2 ** 2))

    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def compute_npcr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Number of Pixels Change Rate (NPCR) between two images (Eq. 37).
    
    NPCR = (number of differing pixels / total pixels) × 100%
    
    Ideal: ≈ 99.6%
    """
    diff = (image1.astype(np.int16) != image2.astype(np.int16))
    return np.sum(diff) / diff.size * 100.0


def compute_uaci(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Unified Average Changing Intensity (UACI) between two images.
    
    UACI = (1/(H×W)) × Σ |C1(p,q) - C2(p,q)| / (L-1) × 100%
    
    where L = 256 for 8-bit images.
    Ideal: ≈ 33.46%
    """
    diff = np.abs(image1.astype(np.float64) - image2.astype(np.float64))
    uaci = np.mean(diff / 255.0) * 100.0
    return uaci


def compute_mse(original: np.ndarray, recovered: np.ndarray) -> float:
    """Compute Mean Squared Error between original and recovered images."""
    return np.mean((original.astype(np.float64) - recovered.astype(np.float64)) ** 2)


def compute_psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (in dB)."""
    mse = compute_mse(original, recovered)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def histogram_uniformity_chi_squared(image: np.ndarray) -> float:
    """
    Compute chi-squared statistic to test histogram uniformity.
    
    For a uniform distribution over 256 bins with N total pixels,
    the expected frequency is N/256.
    Lower values indicate more uniform histograms.
    """
    hist = np.histogram(image, bins=256, range=(0, 255))[0].astype(np.float64)
    expected = image.size / 256.0
    chi2 = np.sum((hist - expected) ** 2 / expected)
    return chi2


def full_evaluation(original: np.ndarray, encrypted: np.ndarray,
                    decrypted: Optional[np.ndarray] = None) -> dict:
    """
    Run the complete security evaluation suite.
    
    Parameters
    ----------
    original : ndarray
        Original plaintext image.
    encrypted : ndarray
        Encrypted (ciphertext) image.
    decrypted : ndarray, optional
        Decrypted image (for recovery verification).
    
    Returns
    -------
    results : dict
        Dictionary of all computed metrics.
    """
    results = {}

    # Entropy Analysis
    results['entropy_original'] = compute_entropy(original)
    results['entropy_encrypted'] = compute_entropy(encrypted)

    # Correlation Analysis (plaintext)
    results['corr_original_horizontal'] = compute_correlation(original, 'horizontal')
    results['corr_original_vertical'] = compute_correlation(original, 'vertical')
    results['corr_original_diagonal'] = compute_correlation(original, 'diagonal')

    # Correlation Analysis (ciphertext)
    results['corr_encrypted_horizontal'] = compute_correlation(encrypted, 'horizontal')
    results['corr_encrypted_vertical'] = compute_correlation(encrypted, 'vertical')
    results['corr_encrypted_diagonal'] = compute_correlation(encrypted, 'diagonal')

    # Histogram Uniformity
    results['chi2_original'] = histogram_uniformity_chi_squared(original)
    results['chi2_encrypted'] = histogram_uniformity_chi_squared(encrypted)

    # Recovery quality (if decrypted image is provided)
    if decrypted is not None:
        results['mse_recovery'] = compute_mse(original, decrypted)
        results['psnr_recovery'] = compute_psnr(original, decrypted)
        results['perfect_recovery'] = bool(np.array_equal(original, decrypted))

    return results


def differential_attack_analysis(original: np.ndarray,
                                 encrypt_func,
                                 key) -> dict:
    """
    Perform differential attack analysis (NPCR and UACI).
    
    Changes a single pixel in the original image by 1 bit and compares
    the resulting ciphertext with the original ciphertext.
    
    Parameters
    ----------
    original : ndarray
        Original plaintext image.
    encrypt_func : callable
        Function that encrypts an image: encrypt_func(image, key) -> cipher
    key : EncryptionKey
        The encryption key to use.
    
    Returns
    -------
    results : dict
        NPCR and UACI values.
    """
    # Encrypt original
    cipher1, _, _ = encrypt_func(original, key)

    # Create 1-bit altered version
    altered = original.copy()
    # Change the LSB of the pixel at (0, 0)
    altered[0, 0] = altered[0, 0] ^ 1

    # Encrypt altered
    cipher2, _, _ = encrypt_func(altered, key)

    npcr = compute_npcr(cipher1, cipher2)
    uaci = compute_uaci(cipher1, cipher2)

    return {
        'npcr': npcr,
        'uaci': uaci,
        'cipher_diff': np.abs(cipher1.astype(np.int16) - cipher2.astype(np.int16)).astype(np.uint8),
    }


def key_sensitivity_analysis(original: np.ndarray,
                              encrypt_func, decrypt_func,
                              key, delta: float = 1e-10) -> dict:
    """
    Analyze sensitivity to small key changes.
    
    Encrypts with the original key, then attempts decryption with a
    slightly modified key. Perfect encryption should produce completely
    different results.
    """
    from .encryption import EncryptionKey

    # Encrypt with original key
    cipher, _, _ = encrypt_func(original, key)

    # Create slightly modified key
    modified_key = EncryptionKey(
        dj1_a=key.dj1_a + delta,
        dj1_b=key.dj1_b, dj1_c=key.dj1_c, dj1_d=key.dj1_d,
        dj1_x0=key.dj1_x0, dj1_y0=key.dj1_y0,
        dj2_a=key.dj2_a, dj2_b=key.dj2_b,
        dj2_c=key.dj2_c, dj2_d=key.dj2_d,
        dj2_x0=key.dj2_x0, dj2_y0=key.dj2_y0,
        vdp_h=key.vdp_h, vdp_mu=key.vdp_mu,
        vdp_x0=key.vdp_x0, vdp_v0=key.vdp_v0,
        block_size=key.block_size,
    )

    # Decrypt with modified key
    wrong_decrypted, _ = decrypt_func(cipher, modified_key)

    # Decrypt with correct key
    correct_decrypted, _ = decrypt_func(cipher, key)

    return {
        'npcr_wrong_key': compute_npcr(original, wrong_decrypted),
        'mse_wrong_key': compute_mse(original, wrong_decrypted),
        'psnr_wrong_key': compute_psnr(original, wrong_decrypted),
        'perfect_with_correct_key': bool(np.array_equal(original, correct_decrypted)),
        'wrong_decrypted': wrong_decrypted,
    }


def occlusion_attack_analysis(original: np.ndarray,
                               cipher: np.ndarray,
                               decrypt_func,
                               key,
                               occlusion_percentages: list = None) -> dict:
    """
    Test resilience against occlusion (data loss) attacks.
    
    Replaces portions of the cipher image with zeros and attempts decryption.
    """
    if occlusion_percentages is None:
        occlusion_percentages = [10, 25, 50]

    results = {}
    H, W = cipher.shape

    for pct in occlusion_percentages:
        occluded = cipher.copy()
        # Occlude from top-left corner
        occ_rows = int(H * np.sqrt(pct / 100))
        occ_cols = int(W * np.sqrt(pct / 100))
        occluded[:occ_rows, :occ_cols] = 0

        recovered, _ = decrypt_func(occluded, key)

        # Compute recovery quality
        mse = compute_mse(original, recovered)
        psnr = compute_psnr(original, recovered)

        results[f'occlusion_{pct}pct'] = {
            'mse': mse,
            'psnr': psnr,
            'occluded_cipher': occluded,
            'recovered': recovered,
        }

    return results
