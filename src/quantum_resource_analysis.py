"""
Quantum Resource Analysis — Task T5
=====================================
Analyzes the quantum circuit resources (gate counts, depth, qubit counts)
for all quantum components of the encryption scheme and compares against
the paper's reported values (Table II, Section V-B).

Components analyzed:
  1. NEQR encoding circuits (for various image sizes)
  2. CNOT cascade circuits A and B (PM1_1)
  3. Geometric permutation SWAP networks (PM2_1/T3)
  4. Full pipeline resource estimates

The paper reports:
  - Circuit A & B: 7 CNOT gates each, depth 7
  - NEQR for 2^n x 2^n: O(2^{2n} * 8) gates
  - Total qubits for 256x256: 2*8 + 8 = 24 (but 2^16 basis states)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .neqr import build_neqr_circuit, get_circuit_stats
from .quantum_cnot_circuits import (
    build_circuit_a, build_circuit_b,
    build_circuit_a_inverse, build_circuit_b_inverse,
    get_circuit_resource_report
)


def analyze_cnot_circuits() -> dict:
    """
    Detailed resource analysis of CNOT cascade circuits A and B.

    Returns gate counts, depth, and comparison with paper Table II.
    """
    report = get_circuit_resource_report()

    # Also analyze inverse circuits
    inv_a = build_circuit_a_inverse()
    inv_b = build_circuit_b_inverse()

    def _stats(qc):
        ops = qc.count_ops()
        return {
            'n_qubits': qc.num_qubits,
            'depth': qc.depth(),
            'cx_count': ops.get('cx', 0),
            'total_gates': sum(ops.values()),
            'gate_breakdown': dict(ops),
        }

    report['circuit_a_inverse'] = _stats(inv_a)
    report['circuit_b_inverse'] = _stats(inv_b)

    # Verify: compose forward + inverse should give identity-depth behavior
    identity_a = build_circuit_a().compose(build_circuit_a_inverse())
    identity_b = build_circuit_b().compose(build_circuit_b_inverse())

    report['identity_check'] = {
        'circuit_a_depth': identity_a.depth(),
        'circuit_b_depth': identity_b.depth(),
        'circuit_a_gates': sum(identity_a.count_ops().values()),
        'circuit_b_gates': sum(identity_b.count_ops().values()),
    }

    return report


def analyze_neqr_scaling(max_n: int = 3) -> list[dict]:
    """
    Analyze NEQR circuit resources for different image sizes.

    Parameters
    ----------
    max_n : int
        Maximum n value (image is 2^n × 2^n). Default 3 → 8×8.

    Returns
    -------
    results : list of dict
        Resource metrics for each image size.
    """
    results = []

    for n in range(1, max_n + 1):
        size = 2 ** n
        # Create a test image with varied pixel values
        rng = np.random.RandomState(42)
        test_image = rng.randint(0, 256, (size, size), dtype=np.uint8)

        qc = build_neqr_circuit(test_image)
        stats = get_circuit_stats(qc)

        # Theoretical estimates from paper
        n_pixels = size * size
        theoretical_max_mcx = n_pixels * 8  # worst case: all 8 bits set for all pixels
        # Count non-zero pixels
        nonzero_pixels = np.count_nonzero(test_image)
        nonzero_bits = sum(bin(int(p)).count('1') for p in test_image.flatten())

        stats['image_size'] = f'{size}x{size}',
        stats['n'] = n
        stats['total_pixels'] = n_pixels
        stats['nonzero_pixels'] = int(nonzero_pixels)
        stats['nonzero_bits'] = int(nonzero_bits)
        stats['theoretical_max_mcx'] = theoretical_max_mcx
        stats['position_qubits'] = 2 * n
        stats['pixel_qubits'] = 8
        stats['total_qubits'] = 2 * n + 8

        results.append(stats)

    return results


def analyze_transpiled_resources(max_n: int = 2) -> list[dict]:
    """
    Analyze resources after transpilation to a basis gate set.

    Transpiles circuits to {CX, U1, U2, U3} (IBM hardware-like basis)
    to get realistic gate counts for physical implementation.

    Parameters
    ----------
    max_n : int
        Maximum n (default 2 → 4×4 images for tractable transpilation).

    Returns
    -------
    results : list of dict
        Pre- and post-transpilation resource comparison.
    """
    results = []

    for n in range(1, max_n + 1):
        size = 2 ** n
        rng = np.random.RandomState(42)
        test_image = rng.randint(0, 256, (size, size), dtype=np.uint8)

        qc = build_neqr_circuit(test_image)

        # Pre-transpilation stats
        pre_ops = qc.count_ops()
        pre_stats = {
            'n_qubits': qc.num_qubits,
            'depth': qc.depth(),
            'total_gates': sum(pre_ops.values()),
            'gate_breakdown': dict(pre_ops),
        }

        # Transpile to basis gates
        pm = generate_preset_pass_manager(optimization_level=2, basis_gates=['cx', 'u1', 'u2', 'u3', 'id'])
        transpiled = pm.run(qc)

        post_ops = transpiled.count_ops()
        post_stats = {
            'n_qubits': transpiled.num_qubits,
            'depth': transpiled.depth(),
            'total_gates': sum(post_ops.values()),
            'cx_count': post_ops.get('cx', 0),
            'gate_breakdown': dict(post_ops),
        }

        results.append({
            'n': n,
            'image_size': f'{size}x{size}',
            'pre_transpilation': pre_stats,
            'post_transpilation': post_stats,
            'cx_overhead': post_stats['cx_count'],
            'depth_overhead': post_stats['depth'] / max(pre_stats['depth'], 1),
        })

    return results


def full_pipeline_resource_estimate(image_size: int = 256) -> dict:
    """
    Estimate total quantum resources for the complete encryption pipeline
    on a given image size.

    This provides theoretical estimates since full circuits for large images
    can't be simulated — matching the paper's analysis approach.

    Parameters
    ----------
    image_size : int
        Image dimension (square). Default 256 for paper comparison.

    Returns
    -------
    estimate : dict
        Resource estimates for each pipeline component.
    """
    n = int(np.log2(image_size))
    n_pixels = image_size * image_size

    # CNOT cascade resources (per pixel)
    cnot_per_pixel = 7  # Both Circuit A and B use exactly 7 CNOTs

    # NEQR encoding: each non-zero pixel bit requires a multi-controlled X
    # Worst case: all 8 bits set for all pixels → 8 * n_pixels MCX gates
    # Each MCX on 2n controls decomposes to O(n) Toffolis → O(n) CX each
    mcx_per_pixel = 8  # worst case
    cx_per_mcx = max(1, 2 * n - 3) * 2  # Rough MCX decomposition cost

    # Geometric permutation (SWAP network)
    # Permutation of 2^n elements: O(n * 2^n) SWAP gates
    swap_count_estimate = n * image_size  # per dimension
    cx_per_swap = 3  # Each SWAP = 3 CX

    estimate = {
        'image_size': f'{image_size}x{image_size}',
        'n_param': n,
        'total_pixels': n_pixels,

        'neqr': {
            'qubits': 2 * n + 8,
            'max_mcx_gates': mcx_per_pixel * n_pixels,
            'estimated_cx_after_decomposition': mcx_per_pixel * n_pixels * cx_per_mcx,
            'description': f'NEQR encoding for {image_size}x{image_size}: {2*n+8} qubits',
        },

        'pm1_1_cnot_cascade': {
            'cnot_per_pixel': cnot_per_pixel,
            'total_cnot': cnot_per_pixel * n_pixels,
            'depth_per_pixel': 7,
            'description': 'Circuit A/B CNOT cascade (7 CX gates each)',
        },

        'pm2_1_geometric_permutation': {
            'estimated_swap_gates': swap_count_estimate * 2,  # row + col
            'estimated_cx': swap_count_estimate * 2 * cx_per_swap,
            'description': f'SWAP network for {image_size}-element permutation',
        },

        'total_estimated': {
            'qubits': 2 * n + 8,
            'estimated_cx_total': (
                mcx_per_pixel * n_pixels * cx_per_mcx +
                cnot_per_pixel * n_pixels +
                swap_count_estimate * 2 * cx_per_swap
            ),
            'note': 'Theoretical estimate — actual circuit may be optimized',
        },

        'paper_comparison': {
            'paper_circuit_a_cnot': 7,
            'paper_circuit_b_cnot': 7,
            'paper_qubits_256x256': 24,
            'paper_keyspace': '>10^105',
            'note': 'Values from paper Section V-B, Table II',
        },
    }

    return estimate


def generate_resource_report() -> dict:
    """
    Generate a comprehensive resource analysis report.

    Returns
    -------
    report : dict
        Complete resource analysis across all components.
    """
    report = {}

    print("  Analyzing CNOT cascade circuits...")
    report['cnot_circuits'] = analyze_cnot_circuits()

    print("  Analyzing NEQR scaling (2x2 to 8x8)...")
    report['neqr_scaling'] = analyze_neqr_scaling(max_n=3)

    print("  Analyzing transpiled resources (2x2 to 4x4)...")
    report['transpiled'] = analyze_transpiled_resources(max_n=2)

    print("  Computing full pipeline estimates...")
    for size in [4, 8, 16, 64, 256]:
        report[f'pipeline_{size}x{size}'] = full_pipeline_resource_estimate(size)

    return report
