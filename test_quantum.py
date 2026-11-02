"""
Tests for T1 (NEQR encoding) and T2 (Quantum CNOT circuits).
Runs with Qiskit statevector simulator — no IBM hardware needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("T1: NEQR QUANTUM IMAGE ENCODING TEST")
print(SEP)

from src.neqr import build_neqr_circuit, decode_neqr_statevector, get_circuit_stats

# Test on a 2x2 image (n=1, 10 qubits)
print("\n[T1.1] NEQR encoding of 2x2 image (n=1, 10 qubits)...")
img_2x2 = np.array([[100, 200],
                     [ 50, 255]], dtype=np.uint8)
qc_2x2 = build_neqr_circuit(img_2x2)
stats_2x2 = get_circuit_stats(qc_2x2)
sv_2x2 = Statevector(qc_2x2)
decoded_2x2 = decode_neqr_statevector(np.array(sv_2x2), n=1)

match = np.array_equal(img_2x2, decoded_2x2)
print(f"  Original : {img_2x2.tolist()}")
print(f"  Decoded  : {decoded_2x2.tolist()}")
print(f"  Match    : {match}")
print(f"  Qubits   : {stats_2x2['n_qubits']}")
print(f"  Depth    : {stats_2x2['depth']}")
print(f"  Gates    : {stats_2x2['total_gates']} (breakdown: {stats_2x2['gate_breakdown']})")
assert match, "T1.1 FAILED: 2x2 NEQR decode mismatch!"
print("  [PASSED]")

# Test on a 4x4 image (n=2, 12 qubits)
print("\n[T1.2] NEQR encoding of 4x4 image (n=2, 12 qubits)...")
rng = np.random.RandomState(42)
img_4x4 = rng.randint(0, 256, (4, 4), dtype=np.uint8)
qc_4x4 = build_neqr_circuit(img_4x4)
stats_4x4 = get_circuit_stats(qc_4x4)
sv_4x4 = Statevector(qc_4x4)
decoded_4x4 = decode_neqr_statevector(np.array(sv_4x4), n=2)

match_4 = np.array_equal(img_4x4, decoded_4x4)
print(f"  Original:\n{img_4x4}")
print(f"  Decoded:\n{decoded_4x4}")
print(f"  Match    : {match_4}")
print(f"  Qubits   : {stats_4x4['n_qubits']}")
print(f"  Depth    : {stats_4x4['depth']}")
print(f"  Total gates: {stats_4x4['total_gates']}")
print(f"  MCX gates: {stats_4x4['mcx_count']}, CCX: {stats_4x4['ccx_count']}, CX: {stats_4x4['cx_count']}")
assert match_4, "T1.2 FAILED: 4x4 NEQR decode mismatch!"
print("  [PASSED]")

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("T2: QUANTUM CNOT CIRCUIT TESTS")
print(SEP)

from src.quantum_cnot_circuits import (
    build_circuit_a, build_circuit_b,
    build_circuit_a_inverse, build_circuit_b_inverse,
    get_circuit_resource_report,
    verify_circuits_match_classical,
)

# [T2.1] Circuit structure check
print("\n[T2.1] Circuit resource metrics (paper Table II comparison)...")
report = get_circuit_resource_report()
for label, key in [("Circuit A", "circuit_a"), ("Circuit B", "circuit_b")]:
    r = report[key]
    paper = report["paper_reference"]
    cnot_ok = r["cnot_count"] == paper["cnot_per_circuit"]
    depth_ok = r["depth"] == paper["depth"]
    print(f"  {label}:")
    print(f"    CNOT count = {r['cnot_count']} (paper: {paper['cnot_per_circuit']}) {'[OK]' if cnot_ok else '[MISMATCH]'}")
    print(f"    Depth      = {r['depth']}  (paper: {paper['depth']}) {'[OK]' if depth_ok else '[MISMATCH]'}")
    print(f"    Qubits     = {r['n_qubits']}")
    assert cnot_ok, f"T2.1 FAILED: {label} CNOT count wrong!"
    assert depth_ok, f"T2.1 FAILED: {label} depth wrong!"
print("  [PASSED]")

# [T2.2] Quantum circuit is self-consistent: A o A^{-1} = I
print("\n[T2.2] Verifying A o A^{-1} = Identity (statevector round-trip)...")
for pixel_val in [0, 1, 100, 127, 200, 255]:
    qc = QuantumCircuit(8)
    for bit_idx in range(8):
        if (pixel_val >> bit_idx) & 1:
            qc.x(bit_idx)
    qc = qc.compose(build_circuit_a())
    qc = qc.compose(build_circuit_a_inverse())
    sv = np.array(Statevector(qc))
    recovered = int(np.argmax(np.abs(sv)))
    assert recovered == pixel_val, f"T2.2 FAILED: CircA roundtrip pixel={pixel_val}, got {recovered}"

for pixel_val in [0, 1, 100, 127, 200, 255]:
    qc = QuantumCircuit(8)
    for bit_idx in range(8):
        if (pixel_val >> bit_idx) & 1:
            qc.x(bit_idx)
    qc = qc.compose(build_circuit_b())
    qc = qc.compose(build_circuit_b_inverse())
    sv = np.array(Statevector(qc))
    recovered = int(np.argmax(np.abs(sv)))
    assert recovered == pixel_val, f"T2.2 FAILED: CircB roundtrip pixel={pixel_val}, got {recovered}"
print("  All Circuit A and B round-trip tests passed [PASSED]")

# [T2.3] Quantum matches classical NumPy implementation
print("\n[T2.3] Verifying Qiskit circuits match classical NumPy simulation (all 256 pixels)...")
errors = verify_circuits_match_classical()
if errors:
    print(f"  [FAILED] {len(errors)} mismatches found:")
    for e in errors[:5]:
        print(f"    {e}")
    raise AssertionError("T2.3 FAILED: Qiskit and classical don't match!")
else:
    print("  All 512 tests (256 pixels x 2 circuits) matched perfectly [PASSED]")

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("ALL T1 + T2 TESTS PASSED")
print(SEP)
