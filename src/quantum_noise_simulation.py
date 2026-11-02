"""
AerSimulator with Noise Models — Task T4
==========================================
Runs the quantum circuits (NEQR encoding, CNOT cascades, SWAP permutation)
on Qiskit's AerSimulator with realistic noise models.

Noise models applied (standard Qiskit Aer):
  1. Depolarizing noise on single-qubit and two-qubit gates
  2. Readout (measurement) errors
  3. Thermal relaxation (optional, for deeper analysis)

The goal is to measure how quantum noise affects the encryption pipeline's
fidelity — matching the paper's discussion in Section V about quantum
resource viability.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from .neqr import build_neqr_circuit, decode_neqr_statevector, get_circuit_stats
from .quantum_cnot_circuits import build_circuit_a, build_circuit_b


# ─────────────────────────────────────────────────────────────────────────────
# Noise model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_depolarizing_noise_model(
    p1q: float = 0.001,
    p2q: float = 0.01,
    p_readout: float = 0.02
) -> NoiseModel:
    """
    Build a depolarizing noise model with readout errors.

    Parameters
    ----------
    p1q : float
        Single-qubit gate depolarizing error rate (default 0.1%).
    p2q : float
        Two-qubit gate (CX) depolarizing error rate (default 1%).
    p_readout : float
        Measurement bit-flip probability (default 2%).

    Returns
    -------
    noise_model : NoiseModel
        Configured noise model for AerSimulator.
    """
    noise_model = NoiseModel()

    # Depolarizing errors on single-qubit gates
    error_1q = depolarizing_error(p1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['x', 'h', 'u1', 'u2', 'u3', 'id'])

    # Depolarizing errors on two-qubit gates (CX / CNOT)
    error_2q = depolarizing_error(p2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    # Readout (measurement) errors
    # Probability matrix: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
    readout_err = ReadoutError(
        [[1 - p_readout, p_readout],
         [p_readout, 1 - p_readout]]
    )
    noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


# ─────────────────────────────────────────────────────────────────────────────
# Noisy simulation runners
# ─────────────────────────────────────────────────────────────────────────────

def simulate_neqr_noisy(
    image: np.ndarray,
    noise_model: NoiseModel = None,
    shots: int = 8192
) -> dict:
    """
    Simulate the NEQR circuit with a noise model and measure pixel fidelity.

    Parameters
    ----------
    image : ndarray, shape (2^n, 2^n), dtype uint8
        Grayscale image to encode.
    noise_model : NoiseModel, optional
        Noise model to apply. If None, builds default depolarizing model.
    shots : int
        Number of measurement shots.

    Returns
    -------
    results : dict
        Contains decoded image, fidelity metrics, and circuit stats.
    """
    if noise_model is None:
        noise_model = build_depolarizing_noise_model()

    H, W = image.shape
    n = int(np.log2(H))
    n_total = 2 * n + 8

    # Build NEQR circuit
    qc = build_neqr_circuit(image)

    # Add measurements
    qc_measured = qc.copy()
    qc_measured.measure_all()

    # Run on ideal simulator first (statevector)
    ideal_sv = Statevector(qc)
    ideal_decoded = decode_neqr_statevector(np.array(ideal_sv), n)

    # Run on noisy simulator with shots
    noisy_sim = AerSimulator(noise_model=noise_model)
    result = noisy_sim.run(qc_measured, shots=shots).result()
    counts = result.get_counts()

    # Decode noisy measurement results
    noisy_decoded = _decode_counts_to_image(counts, n, n_total)

    # Compute fidelity metrics
    pixel_match = np.sum(ideal_decoded == noisy_decoded)
    total_pixels = H * W
    pixel_fidelity = pixel_match / total_pixels

    mse = np.mean((image.astype(float) - noisy_decoded.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255.0 ** 2 / mse)
    else:
        psnr = float('inf')

    circuit_stats = get_circuit_stats(qc)

    return {
        'original': image,
        'ideal_decoded': ideal_decoded,
        'noisy_decoded': noisy_decoded,
        'pixel_fidelity': pixel_fidelity,
        'mse': mse,
        'psnr': psnr,
        'shots': shots,
        'n_counts': len(counts),
        'circuit_stats': circuit_stats,
        'ideal_match': bool(np.array_equal(image, ideal_decoded)),
    }


def _decode_counts_to_image(counts: dict, n: int, n_total: int) -> np.ndarray:
    """
    Decode shot-based measurement counts back into an image.

    For each position (Y, X), we take the most frequently measured
    pixel value across all shots that measured that position.
    """
    size = 2 ** n
    # Accumulate pixel value votes for each position
    pixel_votes = {}  # (y, x) -> {pixel_val: count}

    for bitstring, count in counts.items():
        # Qiskit returns bitstrings in big-endian format: MSB first
        # But our qubit layout has q[0] = LSB
        # bitstring[0] = MSB qubit, bitstring[-1] = q[0] (LSB)
        bits_be = bitstring.replace(' ', '')

        if len(bits_be) != n_total:
            continue

        # Convert to little-endian (q[0] is rightmost in big-endian string)
        bits_le = bits_be[::-1]

        # Extract fields (in little-endian order)
        pixel_val = int(bits_le[:8][::-1], 2)      # pixel q[0..7]
        x_val = int(bits_le[8:8+n][::-1], 2)        # x-coord
        y_val = int(bits_le[8+n:2*n+8][::-1], 2)    # y-coord

        if 0 <= y_val < size and 0 <= x_val < size:
            key = (y_val, x_val)
            if key not in pixel_votes:
                pixel_votes[key] = {}
            pixel_votes[key][pixel_val] = pixel_votes[key].get(pixel_val, 0) + count

    # Reconstruct image using majority vote
    image = np.zeros((size, size), dtype=np.uint8)
    for (y, x), votes in pixel_votes.items():
        best_val = max(votes, key=votes.get)
        image[y, x] = best_val

    return image


def simulate_cnot_cascade_noisy(
    pixel_values: list[int] = None,
    noise_model: NoiseModel = None,
    shots: int = 8192
) -> dict:
    """
    Simulate the CNOT cascade circuits (A and B) with noise and measure
    the bit-error rate.

    Parameters
    ----------
    pixel_values : list of int, optional
        Pixel values to test (0-255). Defaults to [0, 42, 127, 200, 255].
    noise_model : NoiseModel
        Noise model to use.
    shots : int
        Number of shots per simulation.

    Returns
    -------
    results : dict
        Per-pixel results including ideal vs noisy outputs and error rates.
    """
    if pixel_values is None:
        pixel_values = [0, 42, 85, 127, 170, 200, 255]
    if noise_model is None:
        noise_model = build_depolarizing_noise_model()

    noisy_sim = AerSimulator(noise_model=noise_model)
    results = {'circuit_a': [], 'circuit_b': []}

    for circuit_fn, label in [(build_circuit_a, 'circuit_a'),
                               (build_circuit_b, 'circuit_b')]:
        for pval in pixel_values:
            # Build input state + CNOT cascade
            qc = QuantumCircuit(8, 8)
            for bit_idx in range(8):
                if (pval >> bit_idx) & 1:
                    qc.x(bit_idx)
            qc = qc.compose(circuit_fn())
            qc.measure(list(range(8)), list(range(8)))

            # Ideal result
            sv = Statevector(QuantumCircuit(8).compose(
                QuantumCircuit(8, name='init').compose(qc.remove_final_measurements(inplace=False))
            ))
            ideal_sv_qc = QuantumCircuit(8)
            for bit_idx in range(8):
                if (pval >> bit_idx) & 1:
                    ideal_sv_qc.x(bit_idx)
            ideal_sv_qc = ideal_sv_qc.compose(circuit_fn())
            ideal_sv = Statevector(ideal_sv_qc)
            ideal_output = int(np.argmax(np.abs(np.array(ideal_sv))))

            # Noisy result
            result = noisy_sim.run(qc, shots=shots).result()
            counts = result.get_counts()

            # Most frequent output
            most_common = max(counts, key=counts.get)
            noisy_output = int(most_common, 2)

            # Error rate: fraction of shots that don't match ideal
            ideal_bitstring = format(ideal_output, '08b')
            correct_shots = counts.get(ideal_bitstring, 0)
            error_rate = 1.0 - (correct_shots / shots)

            results[label].append({
                'input': pval,
                'ideal_output': ideal_output,
                'noisy_most_common': noisy_output,
                'error_rate': error_rate,
                'n_unique_outputs': len(counts),
            })

    return results


def noise_sweep_analysis(image_size: int = 4, shots: int = 4096) -> dict:
    """
    Sweep noise parameters and measure NEQR fidelity degradation.

    Tests different depolarizing error rates to characterize the
    noise tolerance of the encryption circuits.

    Parameters
    ----------
    image_size : int
        Image dimension (must be power of 2, typically 4 or 8).
    shots : int
        Shots per simulation point.

    Returns
    -------
    sweep_results : dict
        Results for each noise level tested.
    """
    n = int(np.log2(image_size))

    # Create a small test image
    rng = np.random.RandomState(42)
    test_image = rng.randint(0, 256, (image_size, image_size), dtype=np.uint8)

    noise_levels = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    sweep_results = []

    for p2q in noise_levels:
        p1q = p2q / 10  # Single-qubit errors typically 10x lower
        p_ro = p2q * 2   # Readout errors comparable to 2-qubit gate errors

        noise_model = build_depolarizing_noise_model(p1q, p2q, p_ro)

        try:
            result = simulate_neqr_noisy(test_image, noise_model, shots)
            sweep_results.append({
                'p2q': p2q,
                'p1q': p1q,
                'p_readout': p_ro,
                'pixel_fidelity': result['pixel_fidelity'],
                'mse': result['mse'],
                'psnr': result['psnr'],
            })
        except Exception as e:
            sweep_results.append({
                'p2q': p2q,
                'p1q': p1q,
                'p_readout': p_ro,
                'error': str(e),
            })

    return {
        'image_size': image_size,
        'shots': shots,
        'results': sweep_results,
    }
