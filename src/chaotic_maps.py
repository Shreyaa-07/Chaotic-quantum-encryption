"""
Chaotic Map Key Generators — Paper-Correct Implementation
==========================================================
Implements the exact chaotic maps specified in the paper (Section IV-A):

  2D-LASM (2D Logistic-Adjusted-Sine Map) — Eqs. (14)–(15):
      X_{i+1} = sin(pi * mu * (Y_i + 3) * X_i * (1 - X_i))
      Y_{i+1} = sin(pi * mu * (X_{i+1} + 3) * Y_i * (1 - Y_i))
      Parameter: mu in [0.37, 0.9]  (hyperchaotic regime)

  2D-SLMM (2D Sine-Logistic-Modulation Map) — Eqs. (16)–(17):
      X_{i+1} = alpha * (sin(pi * Y_i) + beta) * X_i * (1 - X_i)
      Y_{i+1} = alpha * (sin(pi * X_{i+1}) + beta) * Y_i * (1 - Y_i)
      Parameters: alpha in [0, 4], beta in [0, 1]

Usage in encryption pipeline:
  - 2D-LASM → PM1_1 (qubit key matrix) and PM2_2 (seed diffusion keystream)
  - 2D-SLMM → PM1_2 (selective shuffle selection key)
  - 2D-LASM → PM2_1 (geometric permutation index vectors)

NOTE: The previous implementation used De Jong Fractal Map and Van der Pol
Oscillator, which are NOT part of the paper. This is the corrected version.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core chaotic map iterators
# ─────────────────────────────────────────────────────────────────────────────

def lasm_2d(x0: float, y0: float, mu: float,
            n_iterations: int, n_transient: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate chaotic sequences using the 2D Logistic-Adjusted-Sine Map (2D-LASM).

    From the paper (Section IV-A, Eqs. 14–15):
        X_{i+1} = sin(pi * mu * (Y_i + 3) * X_i * (1 - X_i))
        Y_{i+1} = sin(pi * mu * (X_{i+1} + 3) * Y_i * (1 - Y_i))

    Parameters
    ----------
    x0 : float
        Initial X value. Must be in (0, 1).
    y0 : float
        Initial Y value. Must be in (0, 1).
    mu : float
        Control parameter. Paper specifies mu in [0.37, 0.9] for hyperchaotic behavior
        with positive Lyapunov exponents.
    n_iterations : int
        Number of useful output samples (after discarding transients).
    n_transient : int
        Number of warm-up iterations to discard (reach the attractor).

    Returns
    -------
    x_seq, y_seq : ndarray of shape (n_iterations,)
        Chaotic sequences. Values lie in (-1, 1) due to the sine nonlinearity.
    """
    if not (0.37 <= mu <= 0.9):
        raise ValueError(f"mu={mu} is outside the hyperchaotic range [0.37, 0.9].")
    if not (0.0 < x0 < 1.0) or not (0.0 < y0 < 1.0):
        raise ValueError("Initial conditions x0 and y0 must be strictly in (0, 1).")

    total = n_iterations + n_transient
    x = np.empty(total)
    y = np.empty(total)
    x[0] = x0
    y[0] = y0

    for i in range(total - 1):
        xi = np.sin(np.pi * mu * (y[i] + 3.0) * x[i] * (1.0 - x[i]))
        yi = np.sin(np.pi * mu * (xi + 3.0) * y[i] * (1.0 - y[i]))
        # Guard: sin output is always in [-1, 1] — no overflow possible, but NaN-guard
        x[i + 1] = xi if np.isfinite(xi) else 0.5
        y[i + 1] = yi if np.isfinite(yi) else 0.5

    return x[n_transient:], y[n_transient:]


def slmm_2d(x0: float, y0: float, alpha: float, beta: float,
            n_iterations: int, n_transient: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate chaotic sequences using the 2D Sine-Logistic-Modulation Map (2D-SLMM).

    From the paper (Section IV-A, Eqs. 16–17):
        X_{i+1} = alpha * (sin(pi * Y_i) + beta) * X_i * (1 - X_i)
        Y_{i+1} = alpha * (sin(pi * X_{i+1}) + beta) * Y_i * (1 - Y_i)

    Parameters
    ----------
    x0 : float
        Initial X value. Must be in (0, 1).
    y0 : float
        Initial Y value. Must be in (0, 1).
    alpha : float
        Scaling parameter. Paper range: [0, 4].
    beta : float
        Offset parameter. Paper range: [0, 1].
    n_iterations : int
        Number of useful output samples.
    n_transient : int
        Warm-up iterations to discard.

    Returns
    -------
    x_seq, y_seq : ndarray of shape (n_iterations,)
        Chaotic sequences.
    """
    if not (0.0 < x0 < 1.0) or not (0.0 < y0 < 1.0):
        raise ValueError("Initial conditions x0 and y0 must be strictly in (0, 1).")

    total = n_iterations + n_transient
    x = np.empty(total)
    y = np.empty(total)
    x[0] = x0
    y[0] = y0

    for i in range(total - 1):
        xi = alpha * (np.sin(np.pi * y[i]) + beta) * x[i] * (1.0 - x[i])
        yi = alpha * (np.sin(np.pi * xi) + beta) * y[i] * (1.0 - y[i])
        x[i + 1] = xi if np.isfinite(xi) else 0.5
        y[i + 1] = yi if np.isfinite(yi) else 0.5

    return x[n_transient:], y[n_transient:]


# ─────────────────────────────────────────────────────────────────────────────
# Key-matrix generators used by the encryption pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_qubit_key_lasm(x0: float, y0: float, mu: float,
                             height: int, width: int) -> np.ndarray:
    """
    Generate an 8-bit key matrix using 2D-LASM for PM1_1 (qubit transform).

    The LSB of each entry selects Circuit A (0) or Circuit B (1) in the
    CNOT cascade. Sequences are quantized to [0, 255].

    Returns
    -------
    key_matrix : ndarray, shape (height, width), dtype uint8
    """
    n_pixels = height * width
    x_seq, y_seq = lasm_2d(x0, y0, mu, n_pixels)

    # Combine both sequences and normalize to [0, 255]
    combined = x_seq + y_seq          # range roughly in (-2, 2)
    c_min, c_max = combined.min(), combined.max()
    if c_max - c_min < 1e-12:
        normalized = np.zeros_like(combined)
    else:
        normalized = (combined - c_min) / (c_max - c_min)  # → [0, 1]

    key_matrix = (normalized * 255.0).astype(np.uint8)
    return key_matrix.reshape(height, width)


def generate_selection_key_slmm(x0: float, y0: float, alpha: float, beta: float,
                                 height: int, width: int) -> np.ndarray:
    """
    Generate a selection key matrix in {0, 1, 2} using 2D-SLMM for PM1_2
    (selective perfect shuffle).

    Values:
        0 → Perfect Shuffle (PS)
        1 → Bit Inversion (BI)
        2 → Circular Shift (CS)

    Returns
    -------
    selection_key : ndarray, shape (height, width), dtype uint8
    """
    n_pixels = height * width
    x_seq, y_seq = slmm_2d(x0, y0, alpha, beta, n_pixels)

    # Use y_seq (as in the paper's threshold selection) → quantize to {0, 1, 2}
    y_min, y_max = y_seq.min(), y_seq.max()
    if y_max - y_min < 1e-12:
        normalized = np.zeros_like(y_seq)
    else:
        normalized = (y_seq - y_min) / (y_max - y_min)   # → [0, 1]

    # Partition into three equal thirds → {0, 1, 2}
    selection = (normalized * 2.9999).astype(np.uint8)
    return selection.reshape(height, width)


def generate_diffusion_keystream_lasm(x0: float, y0: float, mu: float,
                                       height: int, width: int) -> np.ndarray:
    """
    Generate an 8-bit diffusion keystream using 2D-LASM for PM2_2 (seed diffusion).

    This corresponds to K_i in the dynamic feedback equation:
        C_i = P_i XOR K_i XOR C_{i-1}

    A separate LASM instance (different initial conditions from qubit key) is used
    so that the two keystreams are independent.

    Returns
    -------
    diffusion_matrix : ndarray, shape (height, width), dtype uint8
    """
    n_pixels = height * width
    x_seq, y_seq = lasm_2d(x0, y0, mu, n_pixels)

    # Use x_seq only for keystream (paper uses a single sequence per module)
    x_min, x_max = x_seq.min(), x_seq.max()
    if x_max - x_min < 1e-12:
        normalized = np.zeros_like(x_seq)
    else:
        normalized = (x_seq - x_min) / (x_max - x_min)   # → [0, 1]

    diffusion_matrix = (normalized * 255.0).astype(np.uint8)
    return diffusion_matrix.reshape(height, width)


def generate_geometric_permutation_lasm(x0: float, y0: float, mu: float,
                                         n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate row and column permutation index vectors using 2D-LASM for PM2_1
    (geometric coordinate permutation).

    Sorting the chaotic sequence produces a pseudo-random permutation of indices.

    Parameters
    ----------
    n : int
        Number of pixels along one dimension (image is n×n, or use H/W separately).

    Returns
    -------
    row_perm, col_perm : ndarray of shape (n,), dtype int64
        Permutation indices for rows and columns respectively.
    """
    x_seq, y_seq = lasm_2d(x0, y0, mu, 2 * n)
    row_perm = np.argsort(x_seq[:n])
    col_perm = np.argsort(y_seq[n:2 * n])
    return row_perm, col_perm
