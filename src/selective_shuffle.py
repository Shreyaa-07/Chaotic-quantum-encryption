"""
Chaotic Selective Perfect Shuffle Operation (Step 3 / PM1_2)
=============================================================
Implements the confusion module of Phase 2 as described in Section IV-D.

Three bit-shift operations selected per-pixel by a chaotic key:
    0: 2-bit left circular shift with bit reversal prefix
    1: 2-bit right circular shift with bit reversal prefix  
    2: 4-bit left circular shift

VECTORIZED implementation for performance.
"""

import numpy as np


def _extract_bits(image: np.ndarray) -> np.ndarray:
    """Extract 8 bit planes. Returns shape (H, W, 8), index 0 = LSB."""
    bits = np.zeros((*image.shape, 8), dtype=np.uint8)
    for i in range(8):
        bits[:, :, i] = (image >> i) & 1
    return bits


def _combine_bits(bits: np.ndarray) -> np.ndarray:
    """Combine 8 bit planes back to pixel values."""
    result = np.zeros(bits.shape[:2], dtype=np.uint8)
    for i in range(8):
        result += (bits[:, :, i].astype(np.uint8) << i)
    return result


def _apply_op0_forward(bits: np.ndarray) -> np.ndarray:
    """
    Operation 0: Bit reversal + 2-bit left circular shift.
    Input bits[i] indexed 0..7 (LSB..MSB).
    
    Step 1 - Reverse: [p0,p1,p2,p3,p4,p5,p6,p7] -> [p7,p6,p5,p4,p3,p2,p1,p0]
    Step 2 - 2-bit left shift: [p7,p6,p5,p4,p3,p2,p1,p0] -> [p5,p4,p3,p2,p1,p0,p7,p6]
    
    So the new bit at position i comes from:
    reversed = [7,6,5,4,3,2,1,0]  (bit index i -> bit index 7-i)
    After 2-bit left circular shift of this reversed array:
    result[i] = reversed[(i+2) % 8] = bit at position 7-((i+2)%8)
    """
    result = np.zeros_like(bits)
    for i in range(8):
        src = 7 - ((i + 2) % 8)
        result[:, :, i] = bits[:, :, src]
    return result


def _apply_op0_inverse(bits: np.ndarray) -> np.ndarray:
    """Inverse of operation 0: 2-bit right shift + undo reversal."""
    result = np.zeros_like(bits)
    for i in range(8):
        src = 7 - ((i + 2) % 8)
        result[:, :, src] = bits[:, :, i]
    return result


def _apply_op1_forward(bits: np.ndarray) -> np.ndarray:
    """
    Operation 1: Bit reversal + 2-bit right circular shift.
    
    Step 1 - Reverse: bit i -> bit 7-i
    Step 2 - 2-bit right shift: position i gets from position (i-2)%8
    result[i] = reversed[(i-2+8) % 8] = bit at position 7-((i+6)%8)
    """
    result = np.zeros_like(bits)
    for i in range(8):
        src = 7 - ((i + 6) % 8)
        result[:, :, i] = bits[:, :, src]
    return result


def _apply_op1_inverse(bits: np.ndarray) -> np.ndarray:
    """Inverse of operation 1."""
    result = np.zeros_like(bits)
    for i in range(8):
        src = 7 - ((i + 6) % 8)
        result[:, :, src] = bits[:, :, i]
    return result


def _apply_op2_forward(bits: np.ndarray) -> np.ndarray:
    """
    Operation 2: 4-bit left circular shift.
    result[i] = bits[(i+4) % 8]
    """
    result = np.zeros_like(bits)
    for i in range(8):
        result[:, :, i] = bits[:, :, (i + 4) % 8]
    return result


def _apply_op2_inverse(bits: np.ndarray) -> np.ndarray:
    """Inverse: 4-bit right circular shift."""
    result = np.zeros_like(bits)
    for i in range(8):
        result[:, :, i] = bits[:, :, (i - 4 + 8) % 8]
    return result


def selective_shuffle_encrypt(image: np.ndarray,
                               selection_key: np.ndarray) -> np.ndarray:
    """
    Apply chaotic selective perfect shuffle (encryption), vectorized.
    
    Each pixel is processed with one of three operations based on
    the selection key value {0, 1, 2}.
    """
    bits = _extract_bits(image)
    result = np.zeros_like(bits)
    
    # Apply each operation to the corresponding subset of pixels
    for op_idx, apply_fn in enumerate([_apply_op0_forward, _apply_op1_forward, _apply_op2_forward]):
        mask = (selection_key % 3) == op_idx
        if not np.any(mask):
            continue
        
        # Extract only the relevant pixels
        bits_subset = bits.copy()
        for i in range(8):
            bits_subset[:, :, i] *= mask
        
        transformed = apply_fn(bits_subset)
        result += transformed
    
    return _combine_bits(result)


def selective_shuffle_decrypt(image: np.ndarray,
                               selection_key: np.ndarray) -> np.ndarray:
    """
    Apply inverse chaotic selective perfect shuffle (decryption), vectorized.
    """
    bits = _extract_bits(image)
    result = np.zeros_like(bits)
    
    for op_idx, inv_fn in enumerate([_apply_op0_inverse, _apply_op1_inverse, _apply_op2_inverse]):
        mask = (selection_key % 3) == op_idx
        if not np.any(mask):
            continue
        
        bits_subset = bits.copy()
        for i in range(8):
            bits_subset[:, :, i] *= mask
        
        restored = inv_fn(bits_subset)
        result += restored
    
    return _combine_bits(result)
