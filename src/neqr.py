"""
NEQR Quantum Image Representation — Qiskit Implementation (Task T1)
=====================================================================
Implements the Novel Enhanced Quantum Image Representation (NEQR) described
in Section III of the paper (Fig. 4).

NEQR encodes a 2^n x 2^n grayscale image into (2n + 8) qubits:

    |I> = (1/2^n) * SUM_{Y=0}^{2^n-1} SUM_{X=0}^{2^n-1}  |f(Y,X)> ⊗ |Y> ⊗ |X>

where:
  - |f(Y,X)> = |C0 C1 ... C7>  is the 8-qubit grayscale pixel value
  - |Y> uses n qubits for the row index
  - |X> uses n qubits for the column index

Total qubits = 8 (pixel value) + n (row) + n (column) = 2n + 8.

For practical simulation:
  - n=1: 2×2 image  → 10 qubits  (1024-dim Hilbert space)
  - n=2: 4×4 image  → 12 qubits  (4096-dim Hilbert space)
  - n=3: 8×8 image  → 14 qubits  (16384-dim Hilbert space)  ← default demo
  - n=4: 16×16 image → 16 qubits (65536-dim Hilbert space)  ← AerSimulator limit

Qubit register layout (LSB at index 0):
  [0..7]     : pixel intensity register (q0=LSB, q7=MSB)
  [8..8+n-1] : X-coordinate register (column)
  [8+n..2n+7]: Y-coordinate register (row)

NOTE: For n > 4, the state vector becomes too large to simulate on a laptop.
      The circuit is still correct — just run on IBM hardware or with shots-mode.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Utility: integer → fixed-width binary string
# ─────────────────────────────────────────────────────────────────────────────
def _int_to_bits(value: int, n_bits: int) -> str:
    """Return a binary string of exactly n_bits (MSB-first, no '0b' prefix)."""
    return format(value, f'0{n_bits}b')


# ─────────────────────────────────────────────────────────────────────────────
# NEQR circuit builder
# ─────────────────────────────────────────────────────────────────────────────
def build_neqr_circuit(image: np.ndarray) -> QuantumCircuit:
    """
    Build a Qiskit QuantumCircuit that prepares the NEQR state for the given image.

    Parameters
    ----------
    image : ndarray, shape (2^n, 2^n), dtype uint8
        Grayscale image to encode. Dimensions must be a power of 2.

    Returns
    -------
    qc : QuantumCircuit
        Circuit of (2n + 8) qubits whose statevector encodes the NEQR state.
        The circuit is NOT measured — run with statevector_simulator to read
        the amplitudes, or add measurements for shot-based simulation.

    Circuit Structure
    -----------------
    Step 1: Apply H to all 2n position qubits to create uniform superposition:
            (1/2^n) * SUM_{i=0}^{2^{2n}-1} |i>  ⊗  |0>^8
    Step 2: For each pixel (y, x), apply multi-controlled X gates on the
            intensity qubits conditioned on |Y=y> ⊗ |X=x>, encoding the
            pixel value into the corresponding amplitude.

    Gate count: O(2^{2n} * 8) multi-controlled X gates.
    Pixel qubits: indices 0..7  (LSB=q0)
    X-coord qubits: indices 8..8+n-1
    Y-coord qubits: indices 8+n..2n+7
    """
    H, W = image.shape
    if H != W:
        raise ValueError(f"Image must be square, got {H}×{W}.")
    n = int(np.log2(H))
    if 2 ** n != H:
        raise ValueError(f"Image dimension {H} is not a power of 2.")
    if n < 1 or n > 4:
        warnings.warn(
            f"n={n} (image size 2^n={H}): n>4 requires >16 qubits and may be "
            f"too large for statevector simulation."
        )

    n_total = 2 * n + 8  # total qubits

    # Register names (for readability in circuit diagrams)
    q_pixel = QuantumRegister(8, 'pixel')   # qubits 0-7: intensity bits
    q_x     = QuantumRegister(n, 'x')       # qubits 8..8+n-1: column index
    q_y     = QuantumRegister(n, 'y')       # qubits 8+n..2n+7: row index

    qc = QuantumCircuit(q_pixel, q_x, q_y,
                        name=f'NEQR_{H}x{W}')

    # ── Step 1: Hadamard on all position qubits ────────────────────────────
    # Creates: (1/2^n) SUM |Y> ⊗ |X> with pixel register = |0>^8
    for i in range(n):
        qc.h(q_x[i])
        qc.h(q_y[i])

    qc.barrier(label='H done')

    # ── Step 2: Controlled pixel encoding ─────────────────────────────────
    # For each pixel (y, x):
    #   1. Determine the binary addresses of x and y
    #   2. For each bit k of f(y,x) that is 1, apply a multi-controlled-X
    #      on pixel qubit k, controlled on the position qubits matching (y,x).
    #
    # We use X-gates to flip position qubits to |1> where the address bit = 0
    # (so the MCX fires only when address == (y,x)), then undo those flips.

    for y in range(H):
        y_bits = _int_to_bits(y, n)  # MSB first
        for x in range(W):
            x_bits = _int_to_bits(x, n)  # MSB first
            pixel_val = int(image[y, x])
            if pixel_val == 0:
                continue  # |0> is already the default state — skip

            # Collect all 2n position qubits (x first, then y)
            pos_qubits = list(q_x) + list(q_y)  # length 2n

            # Build the combined address bits string.
            # IMPORTANT: Qiskit register q[0] = LSB of statevector index.
            # So addr_bits must be LSB-first (reversed vs MSB-first binary string).
            # x_bits[::-1] puts x's LSB at position 0 → q_x[0], etc.
            addr_bits = x_bits[::-1] + y_bits[::-1]  # length 2n, LSB first per register

            # Flip position qubits where address bit = 0 (so MCX fires on this addr)
            flip_indices = [i for i, b in enumerate(addr_bits) if b == '0']
            for i in flip_indices:
                qc.x(pos_qubits[i])

            # For each bit of the pixel value that is 1, apply MCX on pixel reg
            pixel_bits = _int_to_bits(pixel_val, 8)  # MSB first (bit 7..0)
            for bit_pos in range(8):
                # bit_pos 0 in our pixel register = LSB = pixel_bits[-1]
                if pixel_bits[7 - bit_pos] == '1':
                    # Multi-controlled X: controls = all 2n position qubits
                    # target = pixel qubit at bit_pos
                    if 2 * n == 1:
                        qc.cx(pos_qubits[0], q_pixel[bit_pos])
                    elif 2 * n == 2:
                        qc.ccx(pos_qubits[0], pos_qubits[1], q_pixel[bit_pos])
                    else:
                        ctrl_list = pos_qubits  # list of 2n control qubits
                        qc.append(
                            MCXGate(len(ctrl_list)),
                            ctrl_list + [q_pixel[bit_pos]]
                        )

            # Undo the address-selection flips
            for i in flip_indices:
                qc.x(pos_qubits[i])

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# NEQR statevector decoder: extract pixel values from the statevector
# ─────────────────────────────────────────────────────────────────────────────
def decode_neqr_statevector(statevector: np.ndarray, n: int) -> np.ndarray:
    """
    Decode the NEQR statevector back into a 2^n × 2^n pixel array.

    The NEQR statevector has amplitudes of magnitude (1/2^n) at basis states
    corresponding to each (Y, X, pixel_value) combination. We extract pixel
    values by scanning the statevector for the largest-amplitude basis states.

    Parameters
    ----------
    statevector : ndarray, complex, shape (2^(2n+8),)
        The statevector output from the Qiskit statevector simulator.
    n : int
        The image dimension parameter (image is 2^n × 2^n).

    Returns
    -------
    image : ndarray, shape (2^n, 2^n), dtype uint8
        Decoded pixel values.
    """
    size = 2 ** n
    image = np.zeros((size, size), dtype=np.uint8)
    expected_amp = 1.0 / (2 ** n)  # Each superposed state has this amplitude

    # Qubit layout (Qiskit uses little-endian bit ordering for statevector):
    # statevector index bits (from LSB): q_pixel[0], q_pixel[1], ..., q_pixel[7],
    #                                    q_x[0], ..., q_x[n-1],
    #                                    q_y[0], ..., q_y[n-1]
    sv = np.array(statevector)
    n_total = 2 * n + 8

    for idx in range(len(sv)):
        amp = abs(sv[idx])
        if amp < expected_amp * 0.5:
            continue  # skip near-zero amplitudes

        # Decode the basis index into qubit values.
        # Qiskit statevector is little-endian: bit 0 of idx = q[0] = LSB.
        # bits_le[k] = value of qubit k.
        bits_le = format(idx, f'0{n_total}b')[::-1]   # little-endian: bit 0 = q[0]
        pixel_val = int(bits_le[:8][::-1], 2)           # pixel q[0..7]: LSB-first → reverse for int
        # For x: q_x[0] holds LSB of x, q_x[n-1] holds MSB → bits_le[8:8+n] is LSB-first
        x_val = int(bits_le[8:8+n][::-1], 2)             # reverse LSB-first slice -> MSB-first for int()
        y_val = int(bits_le[8+n:2*n+8][::-1], 2)         # same for y

        if 0 <= y_val < size and 0 <= x_val < size:
            image[y_val, x_val] = pixel_val

    return image


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def get_circuit_stats(qc: QuantumCircuit) -> dict:
    """Return a summary of circuit resource metrics."""
    ops = qc.count_ops()
    return {
        'n_qubits': qc.num_qubits,
        'depth': qc.depth(),
        'total_gates': sum(ops.values()),
        'cx_count': ops.get('cx', 0),
        'ccx_count': ops.get('ccx', 0),
        'mcx_count': ops.get('mcx', 0),
        'h_count': ops.get('h', 0),
        'x_count': ops.get('x', 0),
        'gate_breakdown': dict(ops),
    }
