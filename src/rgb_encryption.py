"""
MCQI RGB Color Image Encryption Extension
==========================================
Extends the grayscale encryption pipeline to RGB color images using the
Multi-Channel Quantum Image (MCQI) representation concept.

The paper's scheme is designed for grayscale images. For RGB extension,
we apply the identical 4-step encryption pipeline independently to each
of the three color channels (R, G, B), then recombine.

This is a classical simulation — the "quantum" aspect is that each channel's
8-bit pixel values are processed by the same CNOT cascade / qubit transforms
that correspond to quantum gate operations in the NEQR framework.

Architecture:
    Input: H×W×3 RGB image (uint8)
    Step 1: Split into R, G, B channels
    Step 2: Encrypt each channel independently with the full pipeline
            (Each channel uses the same key but could use different keys
             for additional security — configurable)
    Step 3: Recombine encrypted channels → encrypted RGB image
    Decryption: reverse of above
"""

import numpy as np
from PIL import Image
import time
from dataclasses import dataclass

from .encryption import encrypt, decrypt, EncryptionKey, load_image, save_image
from .evaluation import (
    compute_entropy, compute_correlation, compute_npcr,
    compute_uaci, compute_mse, compute_psnr,
    histogram_uniformity_chi_squared
)


def load_color_image(path: str, target_size: tuple = None) -> np.ndarray:
    """
    Load an image as an RGB 3D array.

    Parameters
    ----------
    path : str
        Path to the image file.
    target_size : tuple (width, height), optional
        Resize the image to this size.

    Returns
    -------
    image : ndarray, shape (H, W, 3), dtype uint8
        RGB image.
    """
    img = Image.open(path).convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def save_color_image(image: np.ndarray, path: str):
    """Save an RGB image array to a file."""
    Image.fromarray(image, mode='RGB').save(path)


def encrypt_rgb(
    image: np.ndarray,
    key: EncryptionKey = None,
    per_channel_keys: bool = False,
    verbose: bool = False
) -> tuple[np.ndarray, EncryptionKey, dict]:
    """
    Encrypt an RGB image by applying the pipeline to each channel independently.

    Parameters
    ----------
    image : ndarray, shape (H, W, 3), dtype uint8
        Input RGB image.
    key : EncryptionKey, optional
        Base key. If per_channel_keys is False, same key is used for all channels.
    per_channel_keys : bool
        If True, derive slightly different keys for each channel by perturbing
        initial conditions. Increases security but requires storing 3 keys.
    verbose : bool
        Print progress info.

    Returns
    -------
    cipher_rgb : ndarray, shape (H, W, 3), dtype uint8
        Encrypted RGB image.
    key : EncryptionKey
        The base key used.
    timing : dict
        Per-channel and total timing information.
    """
    if key is None:
        key = EncryptionKey()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image.shape}")

    H, W, _ = image.shape
    cipher_rgb = np.zeros_like(image)
    timing = {}
    channel_names = ['Red', 'Green', 'Blue']

    for ch in range(3):
        ch_name = channel_names[ch]
        if verbose:
            print(f"  Encrypting {ch_name} channel...")

        # Optionally derive per-channel keys
        if per_channel_keys and ch > 0:
            ch_key = _derive_channel_key(key, ch)
        else:
            ch_key = key

        t0 = time.perf_counter()
        cipher_ch, _, ch_timing = encrypt(image[:, :, ch], ch_key, verbose=False)
        elapsed = time.perf_counter() - t0

        cipher_rgb[:, :, ch] = cipher_ch
        timing[f'{ch_name.lower()}_time'] = elapsed
        timing[f'{ch_name.lower()}_detail'] = ch_timing

    timing['total'] = sum(v for k, v in timing.items() if k.endswith('_time'))

    if verbose:
        print(f"  RGB encryption complete. Total time: {timing['total']:.3f}s")

    return cipher_rgb, key, timing


def decrypt_rgb(
    cipher: np.ndarray,
    key: EncryptionKey,
    per_channel_keys: bool = False,
    verbose: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Decrypt an RGB encrypted image.

    Parameters
    ----------
    cipher : ndarray, shape (H, W, 3), dtype uint8
        Encrypted RGB image.
    key : EncryptionKey
        The base key used during encryption.
    per_channel_keys : bool
        Must match the value used during encryption.
    verbose : bool
        Print progress info.

    Returns
    -------
    decrypted_rgb : ndarray, shape (H, W, 3), dtype uint8
        Recovered RGB image.
    timing : dict
        Per-channel and total timing information.
    """
    H, W, _ = cipher.shape
    decrypted_rgb = np.zeros_like(cipher)
    timing = {}
    channel_names = ['Red', 'Green', 'Blue']

    for ch in range(3):
        ch_name = channel_names[ch]
        if verbose:
            print(f"  Decrypting {ch_name} channel...")

        if per_channel_keys and ch > 0:
            ch_key = _derive_channel_key(key, ch)
        else:
            ch_key = key

        t0 = time.perf_counter()
        dec_ch, ch_timing = decrypt(cipher[:, :, ch], ch_key, verbose=False)
        elapsed = time.perf_counter() - t0

        decrypted_rgb[:, :, ch] = dec_ch
        timing[f'{ch_name.lower()}_time'] = elapsed

    timing['total'] = sum(v for k, v in timing.items() if k.endswith('_time'))

    if verbose:
        print(f"  RGB decryption complete. Total time: {timing['total']:.3f}s")

    return decrypted_rgb, timing


def _derive_channel_key(base_key: EncryptionKey, channel: int) -> EncryptionKey:
    """
    Derive a per-channel key by slightly perturbing the base key's
    initial conditions. This ensures each channel uses independent
    chaotic sequences while maintaining the same parameter regime.
    """
    # Small perturbation that keeps values in valid range
    delta = channel * 0.0137  # Different offset per channel

    return EncryptionKey(
        lasm1_x0=min(0.99, base_key.lasm1_x0 + delta),
        lasm1_y0=min(0.99, base_key.lasm1_y0 + delta * 0.7),
        lasm1_mu=base_key.lasm1_mu,
        lasm2_x0=min(0.99, base_key.lasm2_x0 + delta),
        lasm2_y0=min(0.99, base_key.lasm2_y0 + delta * 0.5),
        lasm2_mu=base_key.lasm2_mu,
        slmm_x0=min(0.99, base_key.slmm_x0 + delta),
        slmm_y0=min(0.99, base_key.slmm_y0 + delta * 0.3),
        slmm_alpha=base_key.slmm_alpha,
        slmm_beta=base_key.slmm_beta,
        block_size=base_key.block_size,
    )


def evaluate_rgb_encryption(
    original: np.ndarray,
    encrypted: np.ndarray,
    decrypted: np.ndarray = None
) -> dict:
    """
    Compute security metrics for RGB encryption, per-channel and aggregate.

    Parameters
    ----------
    original : ndarray, shape (H, W, 3)
    encrypted : ndarray, shape (H, W, 3)
    decrypted : ndarray, shape (H, W, 3), optional

    Returns
    -------
    results : dict
        Per-channel and aggregate metrics.
    """
    channel_names = ['red', 'green', 'blue']
    results = {}

    for ch, name in enumerate(channel_names):
        orig_ch = original[:, :, ch]
        enc_ch = encrypted[:, :, ch]

        results[f'{name}_entropy_original'] = compute_entropy(orig_ch)
        results[f'{name}_entropy_encrypted'] = compute_entropy(enc_ch)

        for direction in ['horizontal', 'vertical', 'diagonal']:
            results[f'{name}_corr_orig_{direction}'] = compute_correlation(orig_ch, direction)
            results[f'{name}_corr_enc_{direction}'] = compute_correlation(enc_ch, direction)

        results[f'{name}_chi2_original'] = histogram_uniformity_chi_squared(orig_ch)
        results[f'{name}_chi2_encrypted'] = histogram_uniformity_chi_squared(enc_ch)

        if decrypted is not None:
            dec_ch = decrypted[:, :, ch]
            results[f'{name}_mse'] = compute_mse(orig_ch, dec_ch)
            results[f'{name}_psnr'] = compute_psnr(orig_ch, dec_ch)
            results[f'{name}_perfect_recovery'] = bool(np.array_equal(orig_ch, dec_ch))

    # Aggregate metrics
    if decrypted is not None:
        results['rgb_perfect_recovery'] = all(
            results[f'{name}_perfect_recovery'] for name in channel_names
        )

    # Average entropy across channels
    results['avg_entropy_encrypted'] = np.mean([
        results[f'{name}_entropy_encrypted'] for name in channel_names
    ])

    # Average correlation across channels
    for direction in ['horizontal', 'vertical', 'diagonal']:
        results[f'avg_corr_enc_{direction}'] = np.mean([
            results[f'{name}_corr_enc_{direction}'] for name in channel_names
        ])

    return results


def rgb_differential_analysis(
    original: np.ndarray,
    key: EncryptionKey = None
) -> dict:
    """
    Perform NPCR/UACI analysis on RGB image encryption.

    Changes one pixel by 1 bit and measures the change in ciphertext
    across all three channels.
    """
    if key is None:
        key = EncryptionKey()

    cipher1, _, _ = encrypt_rgb(original, key)

    # Alter one pixel in one channel
    altered = original.copy()
    altered[0, 0, 0] ^= 1  # Flip LSB of red channel at (0,0)

    cipher2, _, _ = encrypt_rgb(altered, key)

    channel_names = ['red', 'green', 'blue']
    results = {}

    for ch, name in enumerate(channel_names):
        results[f'{name}_npcr'] = compute_npcr(cipher1[:, :, ch], cipher2[:, :, ch])
        results[f'{name}_uaci'] = compute_uaci(cipher1[:, :, ch], cipher2[:, :, ch])

    return results
