"""
Chaotic Qubit-Level Transformation (Step 1 / PM1_1)
====================================================
Implements the first confusion module of Phase 1 as described in Section IV-B.

Each pixel's 8-bit grayscale value is treated as 8 qubits.
A chaotic key (from De Jong map) determines the order of CNOT (XOR) operations.
The paper provides two specific circuit examples (Eqs. 18-19, Fig. 11).

This is a VECTORIZED implementation for performance.
"""

import numpy as np


def _extract_bits(image: np.ndarray) -> np.ndarray:
    """
    Extract all 8 bit planes from an image.
    Returns shape (H, W, 8) where [:,:,0] = p0 (LSB), [:,:,7] = p7 (MSB).
    """
    H, W = image.shape
    bits = np.zeros((H, W, 8), dtype=np.uint8)
    for i in range(8):
        bits[:, :, i] = (image >> i) & 1
    return bits


def _combine_bits(bits: np.ndarray) -> np.ndarray:
    """
    Combine 8 bit planes back to pixel values.
    bits: shape (H, W, 8) where [:,:,0] = p0 (LSB).
    """
    result = np.zeros(bits.shape[:2], dtype=np.uint8)
    for i in range(8):
        result += (bits[:, :, i].astype(np.uint8) << i)
    return result


def _apply_circuit_a_vectorized(bits: np.ndarray) -> np.ndarray:
    """
    Apply Circuit A (Eq. 18) to all pixels simultaneously.
    bits: shape (H, W, 8) indexed [y, x, bit_index], bit 0 = LSB.
    
    From Eq. 18:
        p'7 = p0 ^ p7
        p'2 = p'7 ^ p2
        p'6 = p'2 ^ p6
        p'1 = p'6 ^ p1
        p'3 = p'1 ^ p3
        p'5 = p'3 ^ p5
        p'4 = p'5 ^ p4
        p'0 = p0
    """
    result = np.zeros_like(bits)
    
    result[:, :, 0] = bits[:, :, 0]                    # p'0 = p0
    result[:, :, 7] = bits[:, :, 0] ^ bits[:, :, 7]   # p'7 = p0 ^ p7
    result[:, :, 2] = result[:, :, 7] ^ bits[:, :, 2] # p'2 = p'7 ^ p2
    result[:, :, 6] = result[:, :, 2] ^ bits[:, :, 6] # p'6 = p'2 ^ p6
    result[:, :, 1] = result[:, :, 6] ^ bits[:, :, 1] # p'1 = p'6 ^ p1
    result[:, :, 3] = result[:, :, 1] ^ bits[:, :, 3] # p'3 = p'1 ^ p3
    result[:, :, 5] = result[:, :, 3] ^ bits[:, :, 5] # p'5 = p'3 ^ p5
    result[:, :, 4] = result[:, :, 5] ^ bits[:, :, 4] # p'4 = p'5 ^ p4
    
    return result


def _apply_circuit_a_inverse_vectorized(bits: np.ndarray) -> np.ndarray:
    """Inverse of Circuit A, vectorized. Reverse the CNOT cascade."""
    result = np.zeros_like(bits)
    
    result[:, :, 4] = bits[:, :, 5] ^ bits[:, :, 4]   # p4 = p'5 ^ p'4
    result[:, :, 5] = bits[:, :, 3] ^ bits[:, :, 5]   # p5 = p'3 ^ p'5
    result[:, :, 3] = bits[:, :, 1] ^ bits[:, :, 3]   # p3 = p'1 ^ p'3
    result[:, :, 1] = bits[:, :, 6] ^ bits[:, :, 1]   # p1 = p'6 ^ p'1
    result[:, :, 6] = bits[:, :, 2] ^ bits[:, :, 6]   # p6 = p'2 ^ p'6
    result[:, :, 2] = bits[:, :, 7] ^ bits[:, :, 2]   # p2 = p'7 ^ p'2
    result[:, :, 7] = bits[:, :, 0] ^ bits[:, :, 7]   # p7 = p0 ^ p'7
    result[:, :, 0] = bits[:, :, 0]                    # p0 = p'0
    
    return result


def _apply_circuit_b_vectorized(bits: np.ndarray) -> np.ndarray:
    """
    Apply Circuit B (Eq. 19) to all pixels simultaneously.
    
    From Eq. 19:
        p'6 = p6
        p'3 = p6 ^ p3
        p'0 = p'3 ^ p0
        p'2 = p'0 ^ p2
        p'5 = p'2 ^ p5
        p'7 = p'5 ^ p7
        p'4 = p'7 ^ p4
        p'1 = p'4 ^ p1
    """
    result = np.zeros_like(bits)
    
    result[:, :, 6] = bits[:, :, 6]                    # p'6 = p6
    result[:, :, 3] = bits[:, :, 6] ^ bits[:, :, 3]   # p'3 = p6 ^ p3
    result[:, :, 0] = result[:, :, 3] ^ bits[:, :, 0] # p'0 = p'3 ^ p0
    result[:, :, 2] = result[:, :, 0] ^ bits[:, :, 2] # p'2 = p'0 ^ p2
    result[:, :, 5] = result[:, :, 2] ^ bits[:, :, 5] # p'5 = p'2 ^ p5
    result[:, :, 7] = result[:, :, 5] ^ bits[:, :, 7] # p'7 = p'5 ^ p7
    result[:, :, 4] = result[:, :, 7] ^ bits[:, :, 4] # p'4 = p'7 ^ p4
    result[:, :, 1] = result[:, :, 4] ^ bits[:, :, 1] # p'1 = p'4 ^ p1
    
    return result


def _apply_circuit_b_inverse_vectorized(bits: np.ndarray) -> np.ndarray:
    """Inverse of Circuit B, vectorized."""
    result = np.zeros_like(bits)
    
    result[:, :, 1] = bits[:, :, 4] ^ bits[:, :, 1]   # p1 = p'4 ^ p'1
    result[:, :, 4] = bits[:, :, 7] ^ bits[:, :, 4]   # p4 = p'7 ^ p'4
    result[:, :, 7] = bits[:, :, 5] ^ bits[:, :, 7]   # p7 = p'5 ^ p'7
    result[:, :, 5] = bits[:, :, 2] ^ bits[:, :, 5]   # p5 = p'2 ^ p'5
    result[:, :, 2] = bits[:, :, 0] ^ bits[:, :, 2]   # p2 = p'0 ^ p'2
    result[:, :, 0] = bits[:, :, 3] ^ bits[:, :, 0]   # p0 = p'3 ^ p'0
    result[:, :, 3] = bits[:, :, 6] ^ bits[:, :, 3]   # p3 = p6 ^ p'3
    result[:, :, 6] = bits[:, :, 6]                    # p6 = p'6
    
    return result


def qubit_transform_encrypt(image: np.ndarray, key_matrix: np.ndarray) -> np.ndarray:
    """
    Apply chaotic qubit-level transformation (encryption) to each pixel.
    
    The LSB of each key entry selects Circuit A (0) or Circuit B (1).
    Vectorized implementation for performance.
    """
    H, W = image.shape
    bits = _extract_bits(image)
    
    # Create mask for circuit selection
    mask_a = (key_matrix & 1) == 0  # shape (H, W)
    mask_b = ~mask_a
    
    # Apply Circuit A to pixels where mask_a is True
    result = np.zeros_like(bits)
    
    if np.any(mask_a):
        bits_a = bits.copy()
        # Zero out pixels not selected for circuit A
        for i in range(8):
            bits_a[:, :, i] *= mask_a
        transformed_a = _apply_circuit_a_vectorized(bits_a)
        result += transformed_a
    
    if np.any(mask_b):
        bits_b = bits.copy()
        for i in range(8):
            bits_b[:, :, i] *= mask_b
        transformed_b = _apply_circuit_b_vectorized(bits_b)
        result += transformed_b
    
    return _combine_bits(result)


def qubit_transform_decrypt(image: np.ndarray, key_matrix: np.ndarray) -> np.ndarray:
    """
    Apply inverse chaotic qubit-level transformation (decryption).
    Vectorized implementation.
    """
    H, W = image.shape
    bits = _extract_bits(image)
    
    mask_a = (key_matrix & 1) == 0
    mask_b = ~mask_a
    
    result = np.zeros_like(bits)
    
    if np.any(mask_a):
        bits_a = bits.copy()
        for i in range(8):
            bits_a[:, :, i] *= mask_a
        restored_a = _apply_circuit_a_inverse_vectorized(bits_a)
        result += restored_a
    
    if np.any(mask_b):
        bits_b = bits.copy()
        for i in range(8):
            bits_b[:, :, i] *= mask_b
        restored_b = _apply_circuit_b_inverse_vectorized(bits_b)
        result += restored_b
    
    return _combine_bits(result)
