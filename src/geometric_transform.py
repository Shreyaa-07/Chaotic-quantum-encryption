"""
Block-Wise Quantum Geometric Transformations (Step 2 / PM2_1)
==============================================================
Implements the diffusion module of Phase 1 as described in Section IV-C.

The image is divided into blocks of 8×8 pixels. Three geometric transformations
are applied to each block:
1. Vertical Flip (Γ_I^VF) — flip along the y-axis (Eq. 21)
2. Horizontal Flip (Γ_I^HF) — flip along the x-axis (Eq. 22)
3. Coordinate Swap (Γ_I^CS) — transpose the block (Eq. 23)

These transformations correspond to quantum circuits operating on position qubits
(Fig. 12). In our classical simulation, they are standard array operations.

The paper applies all three transformations sequentially to each block.
"""

import numpy as np


def _vertical_flip_block(block: np.ndarray) -> np.ndarray:
    """Flip the block vertically (reverse row order). Eq. 21."""
    return block[::-1, :].copy()


def _horizontal_flip_block(block: np.ndarray) -> np.ndarray:
    """Flip the block horizontally (reverse column order). Eq. 22."""
    return block[:, ::-1].copy()


def _coordinate_swap_block(block: np.ndarray) -> np.ndarray:
    """Transpose the block (swap rows and columns). Eq. 23."""
    return block.T.copy()


def geometric_transform_encrypt(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Apply block-wise geometric transformations for pixel-level scrambling.
    
    All three operations (vertical flip, horizontal flip, coordinate swap)
    are applied sequentially to each 8×8 block as described in the paper.
    
    Parameters
    ----------
    image : ndarray, shape (H, W), dtype uint8
        Input image (output of Step 1).
    block_size : int
        Size of each block (default 8 for 8×8 blocks as in the paper).
    
    Returns
    -------
    scrambled : ndarray, shape (H, W), dtype uint8
        Scrambled image.
    """
    H, W = image.shape
    scrambled = image.copy()

    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            # Handle edge blocks that may be smaller
            y_end = min(y + block_size, H)
            x_end = min(x + block_size, W)
            block = scrambled[y:y_end, x:x_end]

            # Only apply to square blocks (skip edge blocks if not square)
            if block.shape[0] == block.shape[1]:
                # Apply all three transforms sequentially
                block = _vertical_flip_block(block)
                block = _horizontal_flip_block(block)
                block = _coordinate_swap_block(block)
                scrambled[y:y_end, x:x_end] = block

    return scrambled


def geometric_transform_decrypt(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Apply inverse block-wise geometric transformations.
    
    Since we applied VF → HF → CS in encryption, the inverse is:
    CS^{-1} → HF^{-1} → VF^{-1}
    
    Note: transpose is self-inverse, and both flips are self-inverse,
    so we just reverse the order of operations.
    """
    H, W = image.shape
    restored = image.copy()

    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            y_end = min(y + block_size, H)
            x_end = min(x + block_size, W)
            block = restored[y:y_end, x:x_end]

            if block.shape[0] == block.shape[1]:
                # Inverse order: CS^{-1} → HF^{-1} → VF^{-1}
                block = _coordinate_swap_block(block)   # CS is self-inverse
                block = _horizontal_flip_block(block)    # HF is self-inverse
                block = _vertical_flip_block(block)      # VF is self-inverse
                restored[y:y_end, x:x_end] = block

    return restored
