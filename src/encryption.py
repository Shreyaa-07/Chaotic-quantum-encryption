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
"""

import numpy as np
from dataclasses import dataclass
from PIL import Image
import time

from .chaotic_maps import (
    generate_chaotic_key_dejong,
    generate_selection_key_dejong,
    generate_diffusion_matrix_vanderpol,
)
from .qubit_transform import qubit_transform_encrypt, qubit_transform_decrypt
from .geometric_transform import geometric_transform_encrypt, geometric_transform_decrypt
from .selective_shuffle import selective_shuffle_encrypt, selective_shuffle_decrypt
from .seed_diffusion import seed_diffusion_encrypt, seed_diffusion_decrypt


@dataclass
class EncryptionKey:
    """
    Complete secret key configuration for the encryption scheme.
    
    Contains all parameters needed for the chaotic maps.
    Anyone who has these parameters can decrypt the image.
    
    Default values are chosen from the paper's examples:
    - De Jong parameters from Fig. 7: a=1.645, b=1.905, c=0.320, d=1.530
    - Van der Pol parameters from Fig. 8: h=0.29, μ=0.06
    """
    # De Jong Map parameters for Step 1 (qubit-level transformation)
    dj1_a: float = 1.645
    dj1_b: float = 1.905
    dj1_c: float = 0.320
    dj1_d: float = 1.530
    dj1_x0: float = 0.1
    dj1_y0: float = 0.1

    # De Jong Map parameters for Step 3 (selective shuffle selection key)
    # Using the second set of parameters from Fig. 7(b)
    dj2_a: float = 1.42
    dj2_b: float = -2.28
    dj2_c: float = 2.38
    dj2_d: float = -2.08
    dj2_x0: float = 0.1
    dj2_y0: float = 0.1

    # Van der Pol Oscillator parameters for Step 4 (seed diffusion)
    vdp_h: float = 0.29
    vdp_mu: float = 0.06
    vdp_x0: float = 0.5
    vdp_v0: float = 0.5

    # Block size for geometric transformations (Step 2)
    block_size: int = 8

    def to_dict(self) -> dict:
        """Serialize key to dictionary."""
        return {
            'dj1': {'a': self.dj1_a, 'b': self.dj1_b, 'c': self.dj1_c,
                    'd': self.dj1_d, 'x0': self.dj1_x0, 'y0': self.dj1_y0},
            'dj2': {'a': self.dj2_a, 'b': self.dj2_b, 'c': self.dj2_c,
                    'd': self.dj2_d, 'x0': self.dj2_x0, 'y0': self.dj2_y0},
            'vdp': {'h': self.vdp_h, 'mu': self.vdp_mu,
                    'x0': self.vdp_x0, 'v0': self.vdp_v0},
            'block_size': self.block_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EncryptionKey':
        """Deserialize key from dictionary."""
        return cls(
            dj1_a=d['dj1']['a'], dj1_b=d['dj1']['b'],
            dj1_c=d['dj1']['c'], dj1_d=d['dj1']['d'],
            dj1_x0=d['dj1']['x0'], dj1_y0=d['dj1']['y0'],
            dj2_a=d['dj2']['a'], dj2_b=d['dj2']['b'],
            dj2_c=d['dj2']['c'], dj2_d=d['dj2']['d'],
            dj2_x0=d['dj2']['x0'], dj2_y0=d['dj2']['y0'],
            vdp_h=d['vdp']['h'], vdp_mu=d['vdp']['mu'],
            vdp_x0=d['vdp']['x0'], vdp_v0=d['vdp']['v0'],
            block_size=d.get('block_size', 8),
        )


def _generate_all_keys(key: EncryptionKey, height: int, width: int):
    """Pre-generate all chaotic key matrices needed for encryption/decryption."""
    # Step 1: De Jong key for qubit transformation
    qubit_key = generate_chaotic_key_dejong(
        key.dj1_a, key.dj1_b, key.dj1_c, key.dj1_d,
        key.dj1_x0, key.dj1_y0, height, width
    )

    # Step 3: De Jong selection key for selective shuffle
    selection_key = generate_selection_key_dejong(
        key.dj2_a, key.dj2_b, key.dj2_c, key.dj2_d,
        key.dj2_x0, key.dj2_y0, height, width
    )

    # Step 4: Van der Pol diffusion matrix
    diffusion_matrix = generate_diffusion_matrix_vanderpol(
        key.vdp_h, key.vdp_mu, key.vdp_x0, key.vdp_v0,
        height, width
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
        print(f"  Generating chaotic keys...")

    t0 = time.perf_counter()
    qubit_key, selection_key, diffusion_matrix = _generate_all_keys(key, H, W)
    timing['key_generation'] = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Confusion-Diffusion
    # ═══════════════════════════════════════════════════════════════════

    # Step 1 (PM1_1): Chaotic Qubit-Level Transformation (Confusion)
    if verbose:
        print(f"  Phase 1, Step 1: Qubit-Level Transformation...")
    t0 = time.perf_counter()
    step1_output = qubit_transform_encrypt(image, qubit_key)
    timing['step1_qubit_transform'] = time.perf_counter() - t0

    # Step 2 (PM2_1): Block-Wise Geometric Transformations (Diffusion)
    if verbose:
        print(f"  Phase 1, Step 2: Geometric Transformations...")
    t0 = time.perf_counter()
    step2_output = geometric_transform_encrypt(step1_output, key.block_size)
    timing['step2_geometric'] = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Confusion-Diffusion
    # ═══════════════════════════════════════════════════════════════════

    # Step 3 (PM1_2): Chaotic Selective Shuffle (Confusion)
    if verbose:
        print(f"  Phase 2, Step 3: Selective Shuffle...")
    t0 = time.perf_counter()
    step3_output = selective_shuffle_encrypt(step2_output, selection_key)
    timing['step3_selective_shuffle'] = time.perf_counter() - t0

    # Step 4 (PM2_2): Chaotic Seed Diffusion (Diffusion)
    if verbose:
        print(f"  Phase 2, Step 4: Seed Diffusion...")
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
        print(f"  Generating chaotic keys...")

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
