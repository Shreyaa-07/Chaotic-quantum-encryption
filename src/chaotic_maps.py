"""
Chaotic Map Key Generators
==========================
Implements the De Jong Fractal Map and the discretized Van der Pol Oscillator
used for chaotic key generation in the paper (Section IV-A).

De Jong Fractal Map (Eqs. 14-15):
    x_{γ+1} = sin(a·y_γ) − cos(b·x_γ)
    y_{γ+1} = sin(c·x_γ) − cos(d·y_γ)

Van der Pol Oscillator (Eqs. 16-17):
    x_{γ+1} = x_γ + h·v_γ
    v_{γ+1} = v_γ + h·[μ(1 − x_γ²)v_γ − x_γ]
"""

import numpy as np


def de_jong_map(a: float, b: float, c: float, d: float,
                x0: float, y0: float, n_iterations: int,
                n_transient: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate chaotic sequences using the De Jong Fractal Map.

    Parameters
    ----------
    a, b, c, d : float
        Control parameters for the De Jong map.
    x0, y0 : float
        Initial conditions.
    n_iterations : int
        Number of output samples to generate (after transient).
    n_transient : int
        Number of initial iterations to discard (to reach the attractor).

    Returns
    -------
    x_seq, y_seq : ndarray
        Chaotic sequences of length n_iterations.
    """
    total = n_iterations + n_transient
    x = np.empty(total)
    y = np.empty(total)
    x[0] = x0
    y[0] = y0

    for i in range(total - 1):
        x[i + 1] = np.sin(a * y[i]) - np.cos(b * x[i])
        y[i + 1] = np.sin(c * x[i]) - np.cos(d * y[i])

    return x[n_transient:], y[n_transient:]


def van_der_pol_oscillator(h: float, mu: float,
                           x0: float, v0: float,
                           n_iterations: int,
                           n_transient: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate chaotic sequences using the discretized Van der Pol Oscillator.

    Parameters
    ----------
    h : float
        Timestep parameter.
    mu : float
        Nonlinearity parameter.
    x0, v0 : float
        Initial conditions.
    n_iterations : int
        Number of output samples to generate (after transient).
    n_transient : int
        Number of initial iterations to discard.

    Returns
    -------
    x_seq, v_seq : ndarray
        Chaotic sequences of length n_iterations.
    """
    total = n_iterations + n_transient
    x = np.empty(total)
    v = np.empty(total)
    x[0] = x0
    v[0] = v0

    for i in range(total - 1):
        x[i + 1] = x[i] + h * v[i]
        v[i + 1] = v[i] + h * (mu * (1 - x[i] ** 2) * v[i] - x[i])

    return x[n_transient:], v[n_transient:]


def generate_chaotic_key_dejong(a: float, b: float, c: float, d: float,
                                x0: float, y0: float,
                                height: int, width: int) -> np.ndarray:
    """
    Generate an 8-bit key matrix of shape (height, width) using the De Jong map.
    
    The chaotic float sequences are quantized to integers in [0, 255]
    by mapping the range [-2, 2] (the typical De Jong output range) to [0, 255].
    """
    n_pixels = height * width
    x_seq, y_seq = de_jong_map(a, b, c, d, x0, y0, n_pixels)

    # Combine x and y sequences, normalize to [0, 1], then quantize to [0, 255]
    combined = x_seq + y_seq  # range roughly [-4, 4]
    # Normalize using the actual min/max to [0, 1]
    cmin, cmax = combined.min(), combined.max()
    if cmax - cmin < 1e-12:
        normalized = np.zeros_like(combined)
    else:
        normalized = (combined - cmin) / (cmax - cmin)

    key_matrix = np.floor(normalized * 255).astype(np.uint8)
    return key_matrix.reshape(height, width)


def generate_selection_key_dejong(a: float, b: float, c: float, d: float,
                                  x0: float, y0: float,
                                  height: int, width: int) -> np.ndarray:
    """
    Generate a selection key matrix of shape (height, width) with values in {0, 1, 2}
    for the chaotic selective shuffle operation (Step 3 / PM1_2).
    
    Each entry indicates which bit-shift operation to apply:
        0 → 2-bit left shift (with bit reversal)
        1 → 2-bit right shift (with bit reversal)
        2 → 4-bit left shift
    """
    n_pixels = height * width
    x_seq, y_seq = de_jong_map(a, b, c, d, x0, y0, n_pixels)

    # Use the y_seq to generate selection keys
    # Normalize and quantize to {0, 1, 2}
    ymin, ymax = y_seq.min(), y_seq.max()
    if ymax - ymin < 1e-12:
        normalized = np.zeros_like(y_seq)
    else:
        normalized = (y_seq - ymin) / (ymax - ymin)

    selection = (normalized * 2.999).astype(np.uint8)  # values in {0, 1, 2}
    return selection.reshape(height, width)


def generate_diffusion_matrix_vanderpol(h: float, mu: float,
                                        x0: float, v0: float,
                                        height: int, width: int) -> np.ndarray:
    """
    Generate a 256x256 random matrix using the Van der Pol oscillator
    for the seed diffusion module (Step 4 / PM2_2, Eq. 32).
    
    Returns an 8-bit matrix of shape (height, width).
    """
    n_pixels = height * width
    x_seq, v_seq = van_der_pol_oscillator(h, mu, x0, v0, n_pixels)

    # Normalize the x sequence to [0, 255]
    xmin, xmax = x_seq.min(), x_seq.max()
    if xmax - xmin < 1e-12:
        normalized = np.zeros_like(x_seq)
    else:
        normalized = (x_seq - xmin) / (xmax - xmin)

    diffusion_matrix = np.floor(normalized * 255).astype(np.uint8)
    return diffusion_matrix.reshape(height, width)
