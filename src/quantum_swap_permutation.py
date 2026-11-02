"""
Quantum SWAP Network for Geometric Permutation — Task T3
==========================================================
Implements the geometric coordinate permutation (PM2_1) as a quantum SWAP
network operating on the position qubits of an NEQR-encoded image.

The paper (Section IV-C) describes permuting pixel coordinates using chaotic
index vectors. In a quantum setting, this corresponds to a network of SWAP
gates that rearrange the computational basis states of the position registers.

Feasibility:
  - 4×4 images (n=2): 4 position qubits + 8 pixel qubits = 12 qubits ✓
  - 8×8 images (n=3): 6 position qubits + 8 pixel qubits = 14 qubits ✓
  - Larger images: circuit is correct but simulation is intractable.

Approach:
  A permutation of 2^n elements can be decomposed into a sequence of
  transpositions (2-element swaps). Each transposition on position qubits
  is implemented as a multi-controlled SWAP conditioned on the other
  position qubits distinguishing the two swapped indices.

  For simplicity and circuit efficiency, we use the Qiskit SWAP gate
  decomposition and a sorting-network approach (bubble sort) to decompose
  the arbitrary permutation into adjacent transpositions on the position
  register.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SwapGate

from .chaotic_maps import generate_geometric_permutation_lasm


def _permutation_to_swaps(perm: np.ndarray) -> list[tuple[int, int]]:
    """
    Decompose an arbitrary permutation into a sequence of transpositions
    (pairs of indices to swap) using a simple selection-sort decomposition.

    Parameters
    ----------
    perm : ndarray of shape (n,)
        Permutation array where perm[i] = target index for element i.

    Returns
    -------
    swaps : list of (i, j) tuples
        Sequence of transpositions that realize the permutation.
    """
    perm = list(perm)
    swaps = []
    for i in range(len(perm)):
        if perm[i] != i:
            # Find where i currently is
            j = perm.index(i)
            swaps.append((i, j))
            perm[i], perm[j] = perm[j], perm[i]
    return swaps


def build_swap_permutation_circuit(n: int, perm: np.ndarray,
                                   register_name: str = 'pos') -> QuantumCircuit:
    """
    Build a quantum circuit that permutes the basis states of an n-qubit
    register according to the given permutation.

    For a register |k⟩ of n qubits, the circuit maps:
        |k⟩ → |perm[k]⟩

    This is done by decomposing the permutation into transpositions and
    implementing each transposition as a controlled-SWAP network.

    For small n (2-3), the circuits are tractable. For large n, the depth
    grows polynomially but the state space is too large to simulate.

    Parameters
    ----------
    n : int
        Number of qubits in the register.
    perm : ndarray of shape (2^n,)
        Permutation of range(2^n).
    register_name : str
        Name for the quantum register.

    Returns
    -------
    qc : QuantumCircuit
        Circuit implementing the permutation on n qubits.
    """
    q = QuantumRegister(n, register_name)
    qc = QuantumCircuit(q, name=f'Perm_{register_name}')

    swaps = _permutation_to_swaps(perm)

    for (i, j) in swaps:
        # Each transposition (i, j) swaps basis states |i⟩ and |j⟩
        # We implement this using X gates to create the right control pattern
        # and a multi-controlled SWAP

        # Find bits where i and j differ
        diff_bits = i ^ j
        if diff_bits == 0:
            continue

        # Simple case: if i and j differ in exactly one bit, it's just a
        # controlled-X (CNOT) with the other bits as controls
        n_diff = bin(diff_bits).count('1')

        if n_diff == 1:
            # Single bit difference — use multi-controlled X
            target_bit = int(np.log2(diff_bits))
            ctrl_bits = [b for b in range(n) if b != target_bit]

            # Set up controls: for each control bit, flip if that bit should be 0
            for b in ctrl_bits:
                if not ((i >> b) & 1):
                    qc.x(q[b])

            # Apply multi-controlled X
            if len(ctrl_bits) == 0:
                qc.x(q[target_bit])
            elif len(ctrl_bits) == 1:
                qc.cx(q[ctrl_bits[0]], q[target_bit])
            elif len(ctrl_bits) == 2:
                qc.ccx(q[ctrl_bits[0]], q[ctrl_bits[1]], q[target_bit])
            else:
                from qiskit.circuit.library import MCXGate
                qc.append(MCXGate(len(ctrl_bits)),
                          [q[b] for b in ctrl_bits] + [q[target_bit]])

            # Undo control flips
            for b in ctrl_bits:
                if not ((i >> b) & 1):
                    qc.x(q[b])
        else:
            # Multi-bit difference: decompose into sequence of single-bit swaps
            # Use the Fredkin (controlled-SWAP) approach
            bits_diff = [b for b in range(n) if (diff_bits >> b) & 1]

            # Swap using a sequence of controlled operations
            # First, isolate the two states using ancilla-free approach
            for k in range(len(bits_diff) - 1):
                bit_a = bits_diff[k]
                bit_b = bits_diff[k + 1]
                ctrl_bits = [b for b in range(n) if b not in (bit_a, bit_b)]

                # Set controls to match state i's pattern
                for b in ctrl_bits:
                    if not ((i >> b) & 1):
                        qc.x(q[b])

                # Controlled SWAP between the two differing bits
                if len(ctrl_bits) == 0:
                    qc.swap(q[bit_a], q[bit_b])
                elif len(ctrl_bits) == 1:
                    qc.cswap(q[ctrl_bits[0]], q[bit_a], q[bit_b])
                else:
                    # Multi-controlled SWAP = 3 multi-controlled Toffolis
                    # For simplicity, use Qiskit's built-in decomposition
                    from qiskit.circuit.library import MCXGate
                    # SWAP = 3 CNOTs, so controlled-SWAP = 3 controlled-CNOTs
                    # We use a simplified version for small circuits
                    ctrl_qubits = [q[b] for b in ctrl_bits]
                    qc.append(
                        SwapGate().control(len(ctrl_bits)),
                        ctrl_qubits + [q[bit_a], q[bit_b]]
                    )

                # Undo control flips
                for b in ctrl_bits:
                    if not ((i >> b) & 1):
                        qc.x(q[b])

    return qc


def build_geometric_permutation_circuit(
    n: int,
    lasm_x0: float = 0.12, lasm_y0: float = 0.34, lasm_mu: float = 0.75
) -> tuple[QuantumCircuit, dict]:
    """
    Build the complete PM2_1 geometric permutation circuit for a 2^n × 2^n image.

    This creates SWAP networks on both the X and Y position registers of the
    NEQR encoding, using chaotic permutation indices from 2D-LASM.

    Parameters
    ----------
    n : int
        Image dimension parameter (image is 2^n × 2^n).
    lasm_x0, lasm_y0, lasm_mu : float
        2D-LASM parameters for generating the permutation.

    Returns
    -------
    qc : QuantumCircuit
        Circuit on 2n qubits (n for X-coord, n for Y-coord).
    stats : dict
        Circuit resource statistics.
    """
    size = 2 ** n

    # Generate permutation indices from 2D-LASM
    row_perm, col_perm = generate_geometric_permutation_lasm(
        lasm_x0, lasm_y0, lasm_mu, size
    )

    # Build SWAP circuits for row and column permutations
    q_x = QuantumRegister(n, 'x')
    q_y = QuantumRegister(n, 'y')
    qc = QuantumCircuit(q_x, q_y, name=f'PM2_1_{size}x{size}')

    # Column permutation on X register
    col_circuit = build_swap_permutation_circuit(n, col_perm, 'x')
    qc.compose(col_circuit, qubits=list(range(n)), inplace=True)

    qc.barrier(label='col_perm_done')

    # Row permutation on Y register
    row_circuit = build_swap_permutation_circuit(n, row_perm, 'y')
    qc.compose(row_circuit, qubits=list(range(n, 2 * n)), inplace=True)

    # Resource statistics
    ops = qc.count_ops()
    stats = {
        'n_qubits': qc.num_qubits,
        'image_size': f'{size}x{size}',
        'depth': qc.depth(),
        'total_gates': sum(ops.values()),
        'swap_count': ops.get('swap', 0),
        'cx_count': ops.get('cx', 0),
        'gate_breakdown': dict(ops),
        'row_perm': row_perm.tolist(),
        'col_perm': col_perm.tolist(),
    }

    return qc, stats


def verify_permutation_circuit(n: int) -> dict:
    """
    Verify that the SWAP permutation circuit correctly permutes basis states
    for a 2^n register.

    Returns a dict with verification results.
    """
    from qiskit.quantum_info import Statevector

    size = 2 ** n
    row_perm, col_perm = generate_geometric_permutation_lasm(
        0.12, 0.34, 0.75, size
    )

    # Test column permutation
    col_circuit = build_swap_permutation_circuit(n, col_perm, 'x')
    errors = []

    for k in range(size):
        # Prepare basis state |k⟩
        qc = QuantumCircuit(n)
        for bit in range(n):
            if (k >> bit) & 1:
                qc.x(bit)
        qc = qc.compose(col_circuit)

        sv = Statevector(qc)
        result_state = int(np.argmax(np.abs(np.array(sv))))
        expected = col_perm[k]

        if result_state != expected:
            errors.append(f"Col perm: |{k}⟩ → |{result_state}⟩, expected |{expected}⟩")

    return {
        'n': n,
        'image_size': f'{size}x{size}',
        'col_perm': col_perm.tolist(),
        'errors': errors,
        'passed': len(errors) == 0,
    }
