"""
Quantum CNOT Cascade Circuits for PM1_1 — Qiskit Implementation
================================================================
Implements the two quantum CNOT cascade circuits (Circuit A and Circuit B)
described in Section IV-B of the paper (Eqs. 18-19, Fig. 11).

Each circuit operates on a single 8-qubit pixel register (q0..q7 = LSB..MSB).
The chaotic key selects Circuit A or Circuit B per-pixel.

Circuit A (Eq. 18 — Fig. 11a):
    p'0 = p0                        (q0 unchanged)
    p'7 = p0 XOR p7                 (CNOT: control=q0, target=q7)
    p'2 = p'7 XOR p2                (CNOT: control=q7, target=q2)
    p'6 = p'2 XOR p6                (CNOT: control=q2, target=q6)
    p'1 = p'6 XOR p1                (CNOT: control=q6, target=q1)
    p'3 = p'1 XOR p3                (CNOT: control=q1, target=q3)
    p'5 = p'3 XOR p5                (CNOT: control=q3, target=q5)
    p'4 = p'5 XOR p4                (CNOT: control=q5, target=q4)

Circuit B (Eq. 19 — Fig. 11b):
    p'6 = p6                        (q6 unchanged)
    p'3 = p6 XOR p3                 (CNOT: control=q6, target=q3)
    p'0 = p'3 XOR p0                (CNOT: control=q3, target=q0)
    p'2 = p'0 XOR p2                (CNOT: control=q0, target=q2)
    p'5 = p'2 XOR p5                (CNOT: control=q2, target=q5)
    p'7 = p'5 XOR p7                (CNOT: control=q5, target=q7)
    p'4 = p'7 XOR p4                (CNOT: control=q7, target=q4)
    p'1 = p'4 XOR p1                (CNOT: control=q4, target=q1)

Each circuit uses exactly 7 CNOT gates.
The circuits are self-inverse when applied twice (CNOT ∘ CNOT = I).
The actual inverse is the reverse sequence of the same CNOT gates.

Integration with NEQR:
    In a full quantum implementation, Circuit A or B would be applied to
    each pixel's 8-qubit register within the NEQR encoding. Here we expose:
      1. Standalone 8-qubit circuits for resource analysis
      2. A helper that appends the circuit to an existing NEQR circuit
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


# ─────────────────────────────────────────────────────────────────────────────
# Circuit A — Eq. 18
# ─────────────────────────────────────────────────────────────────────────────

def build_circuit_a(name: str = 'CircuitA') -> QuantumCircuit:
    """
    Build Circuit A (Eq. 18, Fig. 11a) as a standalone 8-qubit QuantumCircuit.

    CNOT sequence (control → target):
        q0→q7, q7→q2, q2→q6, q6→q1, q1→q3, q3→q5, q5→q4

    Returns
    -------
    qc : QuantumCircuit
        8-qubit circuit implementing Circuit A.
        Gate count: 7 CNOT gates.
        Circuit depth: 7 (sequential chain).
    """
    q = QuantumRegister(8, 'q')
    qc = QuantumCircuit(q, name=name)

    # Paper Eq. 18: CNOT cascade
    qc.cx(q[0], q[7])   # p'7 = p0 XOR p7
    qc.cx(q[7], q[2])   # p'2 = p'7 XOR p2
    qc.cx(q[2], q[6])   # p'6 = p'2 XOR p6
    qc.cx(q[6], q[1])   # p'1 = p'6 XOR p1
    qc.cx(q[1], q[3])   # p'3 = p'1 XOR p3
    qc.cx(q[3], q[5])   # p'5 = p'3 XOR p5
    qc.cx(q[5], q[4])   # p'4 = p'5 XOR p4

    return qc


def build_circuit_a_inverse(name: str = 'CircuitA_inv') -> QuantumCircuit:
    """
    Build the inverse of Circuit A (reverse CNOT cascade order).

    Inverse sequence (reverse of forward):
        q5→q4, q3→q5, q1→q3, q6→q1, q2→q6, q7→q2, q0→q7
    """
    q = QuantumRegister(8, 'q')
    qc = QuantumCircuit(q, name=name)

    qc.cx(q[5], q[4])
    qc.cx(q[3], q[5])
    qc.cx(q[1], q[3])
    qc.cx(q[6], q[1])
    qc.cx(q[2], q[6])
    qc.cx(q[7], q[2])
    qc.cx(q[0], q[7])

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Circuit B — Eq. 19
# ─────────────────────────────────────────────────────────────────────────────

def build_circuit_b(name: str = 'CircuitB') -> QuantumCircuit:
    """
    Build Circuit B (Eq. 19, Fig. 11b) as a standalone 8-qubit QuantumCircuit.

    CNOT sequence (control → target):
        q6→q3, q3→q0, q0→q2, q2→q5, q5→q7, q7→q4, q4→q1

    Returns
    -------
    qc : QuantumCircuit
        8-qubit circuit implementing Circuit B.
        Gate count: 7 CNOT gates.
        Circuit depth: 7 (sequential chain).
    """
    q = QuantumRegister(8, 'q')
    qc = QuantumCircuit(q, name=name)

    # Paper Eq. 19: CNOT cascade
    qc.cx(q[6], q[3])   # p'3 = p6 XOR p3
    qc.cx(q[3], q[0])   # p'0 = p'3 XOR p0
    qc.cx(q[0], q[2])   # p'2 = p'0 XOR p2
    qc.cx(q[2], q[5])   # p'5 = p'2 XOR p5
    qc.cx(q[5], q[7])   # p'7 = p'5 XOR p7
    qc.cx(q[7], q[4])   # p'4 = p'7 XOR p4
    qc.cx(q[4], q[1])   # p'1 = p'4 XOR p1

    return qc


def build_circuit_b_inverse(name: str = 'CircuitB_inv') -> QuantumCircuit:
    """Build the inverse of Circuit B (reverse CNOT cascade order)."""
    q = QuantumRegister(8, 'q')
    qc = QuantumCircuit(q, name=name)

    qc.cx(q[4], q[1])
    qc.cx(q[7], q[4])
    qc.cx(q[5], q[7])
    qc.cx(q[2], q[5])
    qc.cx(q[0], q[2])
    qc.cx(q[3], q[0])
    qc.cx(q[6], q[3])

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Combined chaotic circuit selector
# ─────────────────────────────────────────────────────────────────────────────

def build_pm1_1_circuit_for_pixel(key_byte: int) -> QuantumCircuit:
    """
    Build the PM1_1 quantum circuit for a single pixel based on the chaotic key.

    The LSB of the key byte selects Circuit A (0) or Circuit B (1),
    matching the classical simulation in qubit_transform.py.

    Parameters
    ----------
    key_byte : int in [0, 255]
        Chaotic key value for this pixel (from 2D-LASM).

    Returns
    -------
    qc : QuantumCircuit
        8-qubit circuit for this pixel's qubit transformation.
    """
    if key_byte & 1 == 0:
        return build_circuit_a(name=f'PM1_1_CircA(key={key_byte})')
    else:
        return build_circuit_b(name=f'PM1_1_CircB(key={key_byte})')


# ─────────────────────────────────────────────────────────────────────────────
# Verification: simulate circuits classically and compare to NumPy qubit_transform
# ─────────────────────────────────────────────────────────────────────────────

def _circuit_to_unitary(qc: QuantumCircuit) -> np.ndarray:
    """
    Get the 256×256 unitary matrix of an 8-qubit circuit using Qiskit's
    unitary simulator (exact, not shot-based).
    """
    from qiskit.quantum_info import Operator
    return np.array(Operator(qc).data)


def verify_circuits_match_classical():
    """
    Verify that the Qiskit circuits produce the same bit transformation as
    the classical NumPy implementation in qubit_transform.py.

    Tests on all 256 possible 8-bit pixel values for both circuits.
    Returns True if all match.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from qiskit.quantum_info import Statevector

    from src.qubit_transform import (
        _extract_bits, _combine_bits,
        _apply_circuit_a_vectorized, _apply_circuit_b_vectorized,
        _apply_circuit_a_inverse_vectorized, _apply_circuit_b_inverse_vectorized,
    )

    errors = []

    for pixel_val in range(256):
        pixel_img = np.array([[pixel_val]], dtype=np.uint8)
        bits = _extract_bits(pixel_img)  # shape (1,1,8)

        for circuit_fn, classical_fn, label in [
            (build_circuit_a, _apply_circuit_a_vectorized, "Circuit A"),
            (build_circuit_b, _apply_circuit_b_vectorized, "Circuit B"),
        ]:
            # ── Classical result ──────────────────────────────────────────
            classical_bits = classical_fn(bits.copy())
            classical_pixel = _combine_bits(classical_bits)[0, 0]

            # ── Quantum (Qiskit) result ───────────────────────────────────
            # Prepare input: |pixel_val> as an 8-qubit basis state
            # Qiskit bit ordering: q[0] = LSB → index 0 of statevector
            qc = QuantumCircuit(8)
            # Initialize to |pixel_val> by flipping qubits where bit=1
            for bit_idx in range(8):
                if (pixel_val >> bit_idx) & 1:
                    qc.x(bit_idx)  # bit_idx 0 = q[0] = LSB

            # Append the CNOT circuit
            cnot_qc = circuit_fn()
            qc = qc.compose(cnot_qc)

            # Get statevector
            sv = Statevector(qc)
            sv_arr = np.array(sv)

            # Find the basis state with amplitude ≈ 1 (pure state)
            quantum_pixel = int(np.argmax(np.abs(sv_arr)))

            if quantum_pixel != classical_pixel:
                errors.append(
                    f"{label}, pixel={pixel_val}: "
                    f"classical={classical_pixel}, quantum={quantum_pixel}"
                )

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Resource analysis
# ─────────────────────────────────────────────────────────────────────────────

def get_circuit_resource_report() -> dict:
    """
    Return resource metrics for Circuit A and Circuit B.
    Compares against paper Table II if available.
    """
    from qiskit.quantum_info import Operator

    circ_a = build_circuit_a()
    circ_b = build_circuit_b()

    def stats(qc):
        ops = qc.count_ops()
        return {
            'n_qubits': qc.num_qubits,
            'depth': qc.depth(),
            'cnot_count': ops.get('cx', 0),
            'total_gates': sum(ops.values()),
            'gate_breakdown': dict(ops),
        }

    return {
        'circuit_a': stats(circ_a),
        'circuit_b': stats(circ_b),
        'paper_reference': {
            'cnot_per_circuit': 7,
            'depth': 7,
            'note': 'Each circuit uses a sequential cascade of 7 CNOT gates (Table II).'
        }
    }
