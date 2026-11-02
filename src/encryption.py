"""
Two-Phase Chaotic Quantum Image Encryption Pipeline
=====================================================
Implements the complete encryption/decryption scheme from Fig. 9 of the paper.

Architecture:
    Phase 1:
        PM1_1 (Confusion):  Chaotic Qubit-Level Transformation (Step 1)
        PM2_1 (Diffusion):  Block-Wise Geometric Transformations (Step 2)
    Phase 2:
        PM1_2 (Confusion):  Chaotic Selective Perfect Shuffle (Step 3)
        PM2_2 (Diffusion):  Chaotic Seed Diffusion (Step 4)

The output of Phase 1 feeds into Phase 2, producing the final cipher image.
Decryption reverses the process: Phase 2 inverse → Phase 1 inverse.

The paper uses 256×256 grayscale images as the primary test case.
We support arbitrary sizes but default to 256×256 for consistency.

IMPORTANT DISTINCTION:
This is a *classical simulation* of the quantum operations described in the
paper, identical to the approach the authors used ("simulated and evaluated
in MATLAB using classical equivalents of quantum gates" — Section V).

Key update (T0 fix):
  All chaotic key generation now uses the paper-correct maps:
    - 2D-LASM  → PM1_1 qubit key, PM2_1 geometric perm, PM2_2 diffusion key
    - 2D-SLMM  → PM1_2 selection key
"""

import numpy as np
from dataclasses import dataclass, field
from PIL import Image
import time

from .chaotic_maps import (
    generate_qubit_key_lasm,
    generate_selection_key_slmm,
    generate_diffusion_keystream_lasm,
)
from .qubit_transform import qubit_transform_encrypt, qubit_transform_decrypt
from .geometric_transform import geometric_transform_encrypt, geometric_transform_decrypt
from .selective_shuffle import selective_shuffle_encrypt, selective_shuffle_decrypt
from .seed_diffusion import seed_diffusion_encrypt, seed_diffusion_decrypt


@dataclass
class EncryptionKey:
    """
    Complete secret key configuration for the encryption scheme.

    Contains all parameters for the two paper-correct chaotic maps:
      - 2D-LASM  (used in PM1_1 qubit confusion and PM2_2 seed diffusion)
      - 2D-SLMM  (used in PM1_2 selective perfect shuffle)

    Default values are chosen to lie within the paper's stated hyperchaotic
    parameter regimes (mu in [0.37, 0.9], alpha in [0, 4], beta in [0, 1]).

    The effective keyspace is > 10^105 (see paper Section V-A).
    """
    # ── 2D-LASM parameters for PM1_1 (qubit-level transformation) ──────────
    lasm1_x0: float = 0.12
    lasm1_y0: float = 0.34
    lasm1_mu: float = 0.75      # mu ∈ [0.37, 0.9]

    # ── 2D-LASM parameters for PM2_2 (seed diffusion keystream) ────────────
    # Different initial conditions → independent keystream from PM1_1
    lasm2_x0: float = 0.56
    lasm2_y0: float = 0.78
    lasm2_mu: float = 0.60      # mu ∈ [0.37, 0.9]

    # ── 2D-SLMM parameters for PM1_2 (selective perfect shuffle) ───────────
    slmm_x0: float = 0.23
    slmm_y0: float = 0.67
    slmm_alpha: float = 3.20    # alpha in [0, 4] — verified chaotic at 3.2
    slmm_beta: float  = 0.20    # beta  in [0, 1]

    # ── Block size for geometric transformations (Step 2) ───────────────────
    block_size: int = 8

    def to_dict(self) -> dict:
        """Serialize key to dictionary."""
        return {
            'lasm1': {'x0': self.lasm1_x0, 'y0': self.lasm1_y0, 'mu': self.lasm1_mu},
            'lasm2': {'x0': self.lasm2_x0, 'y0': self.lasm2_y0, 'mu': self.lasm2_mu},
            'slmm':  {'x0': self.slmm_x0,  'y0': self.slmm_y0,
                      'alpha': self.slmm_alpha, 'beta': self.slmm_beta},
            'block_size': self.block_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EncryptionKey':
        """Deserialize key from dictionary."""
        return cls(
            lasm1_x0=d['lasm1']['x0'], lasm1_y0=d['lasm1']['y0'],
            lasm1_mu=d['lasm1']['mu'],
            lasm2_x0=d['lasm2']['x0'], lasm2_y0=d['lasm2']['y0'],
            lasm2_mu=d['lasm2']['mu'],
            slmm_x0=d['slmm']['x0'],   slmm_y0=d['slmm']['y0'],
            slmm_alpha=d['slmm']['alpha'], slmm_beta=d['slmm']['beta'],
            block_size=d.get('block_size', 8),
        )


def _generate_all_keys(key: EncryptionKey, height: int, width: int):
    """Pre-generate all chaotic key matrices needed for encryption/decryption."""

    # Step 1 (PM1_1): 2D-LASM qubit confusion key
    qubit_key = generate_qubit_key_lasm(
        key.lasm1_x0, key.lasm1_y0, key.lasm1_mu, height, width
    )

    # Step 3 (PM1_2): 2D-SLMM selection key → {0, 1, 2}
    selection_key = generate_selection_key_slmm(
        key.slmm_x0, key.slmm_y0, key.slmm_alpha, key.slmm_beta, height, width
    )

    # Step 4 (PM2_2): 2D-LASM diffusion keystream (independent LASM instance)
    diffusion_matrix = generate_diffusion_keystream_lasm(
        key.lasm2_x0, key.lasm2_y0, key.lasm2_mu, height, width
    )

    return qubit_key, selection_key, diffusion_matrix


def load_image(path: str, target_size: tuple = None) -> np.ndarray:
    """
    Load an image as a grayscale 2D array.

    Parameters
    ----------
    path : str
        Path to the image file.
    target_size : tuple (width, height), optional
        Resize the image to this size. Default is None (keep original).

    Returns
    -------
    image : ndarray, shape (H, W), dtype uint8
    """
    img = Image.open(path).convert('L')  # Convert to grayscale
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: str):
    """Save a 2D grayscale array as an image file."""
    Image.fromarray(image, mode='L').save(path)


def encrypt(image: np.ndarray, key: EncryptionKey = None,
            verbose: bool = False) -> tuple[np.ndarray, EncryptionKey, dict]:
    """
    Encrypt a grayscale image using the two-phase confusion-diffusion scheme.

    Parameters
    ----------
    image : ndarray, shape (H, W), dtype uint8
        Input grayscale image.
    key : EncryptionKey, optional
        Secret key. If None, uses default parameters.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    cipher : ndarray, shape (H, W), dtype uint8
        Encrypted image.
    key : EncryptionKey
        The key used (needed for decryption).
    timing : dict
        Execution times for each step.
    """
    if key is None:
        key = EncryptionKey()

    H, W = image.shape
    timing = {}

    if verbose:
        print(f"Encrypting {W}×{H} image...")
        print(f"  Generating chaotic keys (2D-LASM + 2D-SLMM)...")

    t0 = time.perf_counter()
    qubit_key, selection_key, diffusion_matrix = _generate_all_keys(key, H, W)
    timing['key_generation'] = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Confusion-Diffusion
    # ═══════════════════════════════════════════════════════════════════

    # Step 1 (PM1_1): Chaotic Qubit-Level Transformation (Confusion)
    if verbose:
        print(f"  Phase 1, Step 1: Qubit-Level Transformation (PM1_1)...")
    t0 = time.perf_counter()
    step1_output = qubit_transform_encrypt(image, qubit_key)
    timing['step1_qubit_transform'] = time.perf_counter() - t0

    # Step 2 (PM2_1): Block-Wise Geometric Transformations (Diffusion)
    if verbose:
        print(f"  Phase 1, Step 2: Geometric Transformations (PM2_1)...")
    t0 = time.perf_counter()
    step2_output = geometric_transform_encrypt(step1_output, key.block_size)
    timing['step2_geometric'] = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Confusion-Diffusion
    # ═══════════════════════════════════════════════════════════════════

    # Step 3 (PM1_2): Chaotic Selective Shuffle (Confusion)
    if verbose:
        print(f"  Phase 2, Step 3: Selective Shuffle (PM1_2)...")
    t0 = time.perf_counter()
    step3_output = selective_shuffle_encrypt(step2_output, selection_key)
    timing['step3_selective_shuffle'] = time.perf_counter() - t0

    # Step 4 (PM2_2): Chaotic Seed Diffusion (Diffusion)
    if verbose:
        print(f"  Phase 2, Step 4: Seed Diffusion (PM2_2)...")
    t0 = time.perf_counter()
    cipher = seed_diffusion_encrypt(step3_output, diffusion_matrix)
    timing['step4_seed_diffusion'] = time.perf_counter() - t0

    timing['total'] = sum(timing.values())

    if verbose:
        print(f"  Encryption complete. Total time: {timing['total']:.3f}s")

    return cipher, key, timing


def decrypt(cipher: np.ndarray, key: EncryptionKey,
            verbose: bool = False) -> tuple[np.ndarray, dict]:
    """
    Decrypt an encrypted image using the inverse two-phase scheme.

    Decryption reverses Phase 2, then reverses Phase 1:
        Step 4^{-1} → Step 3^{-1} → Step 2^{-1} → Step 1^{-1}

    Parameters
    ----------
    cipher : ndarray, shape (H, W), dtype uint8
        Encrypted image.
    key : EncryptionKey
        The secret key used during encryption.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    decrypted : ndarray, shape (H, W), dtype uint8
        Recovered plaintext image.
    timing : dict
        Execution times for each step.
    """
    H, W = cipher.shape
    timing = {}

    if verbose:
        print(f"Decrypting {W}×{H} image...")
        print(f"  Generating chaotic keys (2D-LASM + 2D-SLMM)...")

    t0 = time.perf_counter()
    qubit_key, selection_key, diffusion_matrix = _generate_all_keys(key, H, W)
    timing['key_generation'] = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # INVERSE PHASE 2
    # ═══════════════════════════════════════════════════════════════════

    # Step 4^{-1}: Inverse Seed Diffusion
    if verbose:
        print(f"  Inverse Phase 2, Step 4: Seed Diffusion...")
    t0 = time.perf_counter()
    step4_inv = seed_diffusion_decrypt(cipher, diffusion_matrix)
    timing['step4_inv'] = time.perf_counter() - t0

    # Step 3^{-1}: Inverse Selective Shuffle
    if verbose:
        print(f"  Inverse Phase 2, Step 3: Selective Shuffle...")
    t0 = time.perf_counter()
    step3_inv = selective_shuffle_decrypt(step4_inv, selection_key)
    timing['step3_inv'] = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # INVERSE PHASE 1
    # ═══════════════════════════════════════════════════════════════════

    # Step 2^{-1}: Inverse Geometric Transformations
    if verbose:
        print(f"  Inverse Phase 1, Step 2: Geometric Transformations...")
    t0 = time.perf_counter()
    step2_inv = geometric_transform_decrypt(step3_inv, key.block_size)
    timing['step2_inv'] = time.perf_counter() - t0

    # Step 1^{-1}: Inverse Qubit-Level Transformation
    if verbose:
        print(f"  Inverse Phase 1, Step 1: Qubit-Level Transformation...")
    t0 = time.perf_counter()
    decrypted = qubit_transform_decrypt(step2_inv, qubit_key)
    timing['step1_inv'] = time.perf_counter() - t0

    timing['total'] = sum(timing.values())

    if verbose:
        print(f"  Decryption complete. Total time: {timing['total']:.3f}s")

    return decrypted, timing
