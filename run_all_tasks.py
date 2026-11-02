"""
Comprehensive Task Runner — T3 through T8
============================================
Runs all the new task implementations and generates results.

Tasks:
  T3: Geometric Permutation as SWAP Network
  T4: AerSimulator + Noise Model
  T5: Quantum Resource Analysis
  T6: MCQI RGB Color Image Extension
  T7: Noise Attack Resilience Tests
  T8: Comparative Benchmarks
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
import time

from src.encryption import encrypt, decrypt, EncryptionKey, load_image, save_image


def setup_directories():
    """Create output directories."""
    dirs = [
        'results', 'results/images', 'results/plots', 'results/tables',
        'results/quantum', 'results/rgb', 'results/noise_attacks',
        'results/benchmarks'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_test_images(image_dir: str = 'test_images',
                     target_size: tuple = (256, 256)) -> dict:
    """Load all test images from the directory."""
    images = {}
    for fname in os.listdir(image_dir):
        if fname.endswith(('.png', '.jpg', '.bmp', '.tif')):
            path = os.path.join(image_dir, fname)
            name = os.path.splitext(fname)[0]
            try:
                img = load_image(path, target_size)
                images[name] = img
                print(f"  Loaded: {name} ({img.shape})")
            except Exception as e:
                print(f"  Warning: Could not load {fname}: {e}")
    return images


# ═══════════════════════════════════════════════════════════════════════════════
# T3: Geometric Permutation as SWAP Network
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_t3():
    """T3: Build and verify quantum SWAP permutation circuits."""
    print("\n" + "=" * 70)
    print("TASK T3: Geometric Permutation as Quantum SWAP Network")
    print("=" * 70)

    from src.quantum_swap_permutation import (
        build_geometric_permutation_circuit,
        verify_permutation_circuit
    )

    results = {}

    # Build circuits for small image sizes
    for n in [2, 3]:
        size = 2 ** n
        print(f"\n  Building SWAP circuit for {size}×{size} image (n={n})...")
        try:
            qc, stats = build_geometric_permutation_circuit(n)
            print(f"    Qubits: {stats['n_qubits']}")
            print(f"    Depth:  {stats['depth']}")
            print(f"    Total gates: {stats['total_gates']}")
            print(f"    Gate breakdown: {stats['gate_breakdown']}")
            results[f'{size}x{size}'] = stats
        except Exception as e:
            print(f"    ERROR: {e}")
            results[f'{size}x{size}'] = {'error': str(e)}

    # Verify correctness on 4×4
    print("\n  Verifying 4×4 permutation circuit correctness...")
    try:
        verify_result = verify_permutation_circuit(2)
        if verify_result['passed']:
            print("    [OK] All basis states permuted correctly!")
        else:
            print(f"    [FAIL] {len(verify_result['errors'])} errors found:")
            for err in verify_result['errors'][:5]:
                print(f"      {err}")
        results['verification_4x4'] = verify_result
    except Exception as e:
        print(f"    Verification error: {e}")
        results['verification_4x4'] = {'error': str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# T4: AerSimulator + Noise Model
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_t4():
    """T4: Run quantum circuits with noise models."""
    print("\n" + "=" * 70)
    print("TASK T4: AerSimulator + Noise Model")
    print("=" * 70)

    from src.quantum_noise_simulation import (
        build_depolarizing_noise_model,
        simulate_neqr_noisy,
        simulate_cnot_cascade_noisy,
        noise_sweep_analysis
    )

    results = {}

    # Test NEQR with noise on a 4×4 image
    print("\n  Simulating NEQR encoding with noise (4×4 image)...")
    rng = np.random.RandomState(42)
    test_img_4x4 = rng.randint(0, 256, (4, 4), dtype=np.uint8)

    try:
        noise_model = build_depolarizing_noise_model(p1q=0.001, p2q=0.01, p_readout=0.02)
        neqr_result = simulate_neqr_noisy(test_img_4x4, noise_model, shots=8192)
        print(f"    Ideal encoding match: {neqr_result['ideal_match']}")
        print(f"    Noisy pixel fidelity: {neqr_result['pixel_fidelity']:.4f}")
        print(f"    PSNR (noisy): {neqr_result['psnr']:.2f} dB")
        print(f"    Circuit stats: {neqr_result['circuit_stats']['n_qubits']} qubits, "
              f"depth={neqr_result['circuit_stats']['depth']}")
        results['neqr_4x4'] = {k: v for k, v in neqr_result.items()
                                if k not in ('original', 'ideal_decoded', 'noisy_decoded')}
    except Exception as e:
        print(f"    ERROR: {e}")
        results['neqr_4x4'] = {'error': str(e)}

    # Test CNOT cascades with noise
    print("\n  Testing CNOT cascades under noise...")
    try:
        cnot_result = simulate_cnot_cascade_noisy(
            pixel_values=[0, 42, 127, 200, 255],
            shots=4096
        )
        for label in ['circuit_a', 'circuit_b']:
            avg_error = np.mean([r['error_rate'] for r in cnot_result[label]])
            print(f"    {label}: avg error rate = {avg_error:.4f}")
        results['cnot_noisy'] = cnot_result
    except Exception as e:
        print(f"    ERROR: {e}")
        results['cnot_noisy'] = {'error': str(e)}

    # Noise sweep analysis
    print("\n  Running noise sweep analysis (4×4)...")
    try:
        sweep = noise_sweep_analysis(image_size=4, shots=4096)
        for r in sweep['results']:
            if 'error' in r:
                print(f"    p2q={r['p2q']:.4f}: ERROR - {r['error']}")
            else:
                print(f"    p2q={r['p2q']:.4f}: fidelity={r['pixel_fidelity']:.4f}, "
                      f"PSNR={r['psnr']:.1f} dB")
        results['noise_sweep'] = sweep

        # Plot noise sweep
        valid_sweep = [r for r in sweep['results'] if 'error' not in r]
        if valid_sweep:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            p2qs = [r['p2q'] for r in valid_sweep]
            fidelities = [r['pixel_fidelity'] for r in valid_sweep]
            psnrs = [r['psnr'] if r['psnr'] != float('inf') else 50 for r in valid_sweep]

            ax1.semilogx(p2qs, fidelities, 'o-', color='#2196F3', linewidth=2)
            ax1.set_xlabel('2-Qubit Gate Error Rate')
            ax1.set_ylabel('Pixel Fidelity')
            ax1.set_title('NEQR Fidelity vs Noise')
            ax1.grid(True, alpha=0.3)

            ax2.semilogx(p2qs, psnrs, 's-', color='#F44336', linewidth=2)
            ax2.set_xlabel('2-Qubit Gate Error Rate')
            ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('NEQR PSNR vs Noise')
            ax2.grid(True, alpha=0.3)

            plt.suptitle('T4: Quantum Noise Impact on NEQR Encoding', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('results/quantum/t4_noise_sweep.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  Saved: results/quantum/t4_noise_sweep.png")

    except Exception as e:
        print(f"    ERROR: {e}")
        results['noise_sweep'] = {'error': str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# T5: Quantum Resource Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_t5():
    """T5: Analyze quantum circuit resources."""
    print("\n" + "=" * 70)
    print("TASK T5: Quantum Resource Analysis")
    print("=" * 70)

    from src.quantum_resource_analysis import generate_resource_report

    try:
        report = generate_resource_report()

        # Print CNOT circuit analysis
        cnot = report['cnot_circuits']
        print(f"\n  CNOT Circuit A: {cnot['circuit_a']['cnot_count']} CX gates, "
              f"depth={cnot['circuit_a']['depth']}")
        print(f"  CNOT Circuit B: {cnot['circuit_b']['cnot_count']} CX gates, "
              f"depth={cnot['circuit_b']['depth']}")
        print(f"  Paper reference: {cnot['paper_reference']['cnot_per_circuit']} CX, "
              f"depth={cnot['paper_reference']['depth']}")

        # Print NEQR scaling
        print(f"\n  NEQR Scaling:")
        for s in report['neqr_scaling']:
            print(f"    {s['image_size']}: {s['n_qubits']} qubits, "
                  f"depth={s['depth']}, gates={s['total_gates']}")

        # Print pipeline estimates
        print(f"\n  Full Pipeline Estimates:")
        for size in [4, 8, 16, 64, 256]:
            key = f'pipeline_{size}x{size}'
            if key in report:
                est = report[key]['total_estimated']
                print(f"    {size}×{size}: {est['qubits']} qubits, "
                      f"~{est['estimated_cx_total']:,} CX gates")

        # Save as JSON (filter non-serializable items)
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        with open('results/quantum/t5_resource_analysis.json', 'w') as f:
            json.dump(report, f, indent=2, default=make_serializable)
        print("\n  Saved: results/quantum/t5_resource_analysis.json")

        return report

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# T6: MCQI RGB Extension
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_t6():
    """T6: Test RGB color image encryption."""
    print("\n" + "=" * 70)
    print("TASK T6: MCQI RGB Color Image Extension")
    print("=" * 70)

    from src.rgb_encryption import (
        encrypt_rgb, decrypt_rgb, evaluate_rgb_encryption,
        rgb_differential_analysis, load_color_image, save_color_image
    )

    key = EncryptionKey()
    results = {}

    # Create a test RGB image
    rng = np.random.RandomState(42)
    test_rgb = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Also try loading a real image if available
    for test_path in ['test_images/baboon_synthetic.png', 'test_images/natural_blobs.png']:
        if os.path.exists(test_path):
            try:
                test_rgb = load_color_image(test_path, target_size=(256, 256))
                print(f"  Using real image: {test_path}")
                break
            except Exception:
                pass

    print(f"  Test image shape: {test_rgb.shape}")

    # Encrypt and decrypt
    print("\n  Encrypting RGB image...")
    try:
        cipher_rgb, key, enc_timing = encrypt_rgb(test_rgb, key, verbose=True)
        print(f"  Encryption time: {enc_timing['total']:.3f}s")

        print("\n  Decrypting RGB image...")
        decrypted_rgb, dec_timing = decrypt_rgb(cipher_rgb, key, verbose=True)
        print(f"  Decryption time: {dec_timing['total']:.3f}s")

        # Save images
        save_color_image(test_rgb, 'results/rgb/original_rgb.png')
        save_color_image(cipher_rgb, 'results/rgb/encrypted_rgb.png')
        save_color_image(decrypted_rgb, 'results/rgb/decrypted_rgb.png')

        # Evaluate
        print("\n  Evaluating RGB encryption quality...")
        eval_results = evaluate_rgb_encryption(test_rgb, cipher_rgb, decrypted_rgb)

        print(f"    Perfect recovery: {eval_results.get('rgb_perfect_recovery', 'N/A')}")
        print(f"    Avg encrypted entropy: {eval_results['avg_entropy_encrypted']:.4f}")
        for direction in ['horizontal', 'vertical', 'diagonal']:
            print(f"    Avg corr ({direction}): "
                  f"{eval_results[f'avg_corr_enc_{direction}']:.4f}")

        results['evaluation'] = {k: v for k, v in eval_results.items()
                                  if not isinstance(v, np.ndarray)}

        # Plot RGB comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(test_rgb)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cipher_rgb)
        axes[0, 1].set_title('Encrypted RGB')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(decrypted_rgb)
        axes[0, 2].set_title('Decrypted RGB')
        axes[0, 2].axis('off')

        channel_names = ['Red', 'Green', 'Blue']
        channel_colors = ['red', 'green', 'blue']
        for ch in range(3):
            axes[1, ch].hist(test_rgb[:, :, ch].flatten(), bins=256,
                           range=(0, 255), alpha=0.5, color=channel_colors[ch],
                           label='Original', density=True)
            axes[1, ch].hist(cipher_rgb[:, :, ch].flatten(), bins=256,
                           range=(0, 255), alpha=0.5, color='gray',
                           label='Encrypted', density=True)
            axes[1, ch].set_title(f'{channel_names[ch]} Histogram')
            axes[1, ch].legend(fontsize=8)

        plt.suptitle('T6: MCQI RGB Color Image Encryption', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/rgb/t6_rgb_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: results/rgb/t6_rgb_analysis.png")

        # Differential analysis
        print("\n  Running RGB differential analysis (NPCR/UACI)...")
        diff_results = rgb_differential_analysis(test_rgb, key)
        for ch in ['red', 'green', 'blue']:
            print(f"    {ch.capitalize()}: NPCR={diff_results[f'{ch}_npcr']:.2f}%, "
                  f"UACI={diff_results[f'{ch}_uaci']:.2f}%")
        results['differential'] = diff_results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# T7: Noise Attack Resilience Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_t7(images: dict):
    """T7: Test noise attack resilience."""
    print("\n" + "=" * 70)
    print("TASK T7: Noise Attack Resilience Tests")
    print("=" * 70)

    from src.noise_attacks import (
        run_noise_attack_suite,
        run_noise_attack_comparison
    )

    key = EncryptionKey()
    name, img = next(iter(images.items()))
    print(f"  Using image: {name}")

    results = {}

    # Run full noise attack suite
    try:
        print("\n  Running noise attack suite...")
        suite_results = run_noise_attack_suite(img, key, verbose=True)

        # Save sample attacked images
        for attack_key in ['gaussian_sigma25', 'salt_pepper_5pct', 'jpeg_q20', 'crop_25pct']:
            if attack_key in suite_results and 'recovered' in suite_results[attack_key]:
                save_image(suite_results[attack_key]['recovered'],
                          f'results/noise_attacks/recovered_{attack_key}.png')

        results['suite'] = {k: {kk: vv for kk, vv in v.items()
                                if kk not in ('attacked_cipher', 'recovered')}
                           for k, v in suite_results.items()}

        # Plot noise attack comparison
        print("\n  Running noise sweep analysis for plots...")
        comparison = run_noise_attack_comparison(img, key)
        results['comparison'] = comparison

        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Gaussian noise sweep
        gaus = comparison['gaussian_sweep']
        axes[0, 0].plot([r['sigma'] for r in gaus], [r['psnr'] for r in gaus],
                       'o-', color='#2196F3', linewidth=2)
        axes[0, 0].set_xlabel('Noise sigma')
        axes[0, 0].set_ylabel('Recovery PSNR (dB)')
        axes[0, 0].set_title('Gaussian Noise Attack')
        axes[0, 0].grid(True, alpha=0.3)

        # Salt & pepper sweep
        sp = comparison['salt_pepper_sweep']
        axes[0, 1].plot([r['amount_pct'] for r in sp], [r['psnr'] for r in sp],
                       's-', color='#F44336', linewidth=2)
        axes[0, 1].set_xlabel('Noise Amount (%)')
        axes[0, 1].set_ylabel('Recovery PSNR (dB)')
        axes[0, 1].set_title('Salt & Pepper Noise Attack')
        axes[0, 1].grid(True, alpha=0.3)

        # JPEG compression sweep
        jpeg = comparison['jpeg_sweep']
        axes[1, 0].plot([r['quality'] for r in jpeg], [r['psnr'] for r in jpeg],
                       'D-', color='#4CAF50', linewidth=2)
        axes[1, 0].set_xlabel('JPEG Quality Factor')
        axes[1, 0].set_ylabel('Recovery PSNR (dB)')
        axes[1, 0].set_title('JPEG Compression Attack')
        axes[1, 0].grid(True, alpha=0.3)

        # Cropping sweep
        crop = comparison['cropping_sweep']
        axes[1, 1].plot([r['crop_pct'] for r in crop], [r['psnr'] for r in crop],
                       '^-', color='#FF9800', linewidth=2)
        axes[1, 1].set_xlabel('Crop Percentage (%)')
        axes[1, 1].set_ylabel('Recovery PSNR (dB)')
        axes[1, 1].set_title('Cropping Attack')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('T7: Noise Attack Resilience Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/noise_attacks/t7_noise_attacks.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: results/noise_attacks/t7_noise_attacks.png")

        # Visual comparison for key attacks
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        # Row labels
        attack_samples = [
            ('gaussian_sigma25', 'Gaussian sigma=25'),
            ('salt_pepper_5pct', 'S&P 5%'),
            ('jpeg_q20', 'JPEG Q=20'),
            ('crop_25pct', 'Crop 25%'),
        ]

        axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        cipher, _, _ = encrypt(img, key)
        axes[1, 0].imshow(cipher, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title('Encrypted')
        axes[1, 0].axis('off')

        clean_dec, _ = decrypt(cipher, key)
        axes[2, 0].imshow(clean_dec, cmap='gray', vmin=0, vmax=255)
        axes[2, 0].set_title('Clean Recovery')
        axes[2, 0].axis('off')

        for j, (akey, alabel) in enumerate(attack_samples):
            if akey in suite_results and 'attacked_cipher' in suite_results[akey]:
                r = suite_results[akey]
                axes[0, j+1].set_visible(False)
                axes[1, j+1].imshow(r['attacked_cipher'], cmap='gray', vmin=0, vmax=255)
                axes[1, j+1].set_title(f'Attacked\n({alabel})')
                axes[1, j+1].axis('off')

                axes[2, j+1].imshow(r['recovered'], cmap='gray', vmin=0, vmax=255)
                psnr_val = r.get('psnr', 0)
                ssim_val = r.get('ssim', 0)
                axes[2, j+1].set_title(f'Recovered\nPSNR={psnr_val:.1f}dB')
                axes[2, j+1].axis('off')

        plt.suptitle('T7: Visual Noise Attack Recovery', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/noise_attacks/t7_visual_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: results/noise_attacks/t7_visual_comparison.png")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# T8: Comparative Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_t8(images: dict):
    """T8: Run comparative benchmarks."""
    print("\n" + "=" * 70)
    print("TASK T8: Comparative Benchmarks")
    print("=" * 70)

    from src.benchmarks import (
        run_comparative_benchmarks,
        generate_benchmark_plots
    )

    name, img = next(iter(images.items()))
    print(f"  Using image: {name}")

    results = {}

    try:
        benchmark_results = run_comparative_benchmarks(img, verbose=True)

        # Filter for JSON serialization
        results['benchmarks'] = [{k: v for k, v in r.items()
                                   if not isinstance(v, np.ndarray)}
                                  for r in benchmark_results]

        # Generate comparison plots
        generate_benchmark_plots(benchmark_results, 'results/benchmarks')

        # Save results as table
        with open('results/benchmarks/t8_comparison.json', 'w') as f:
            json.dump(results['benchmarks'], f, indent=2, default=str)
        print("  Saved: results/benchmarks/t8_comparison.json")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run all tasks T3-T8."""
    print("=" * 70)
    print("CHAOTIC QUANTUM IMAGE ENCRYPTION — TASKS T3-T8")
    print("=" * 70)

    overall_start = time.perf_counter()
    setup_directories()

    # Load test images
    print("\nLoading test images...")
    images = load_test_images('test_images', target_size=(256, 256))
    if not images:
        print("ERROR: No test images found. Run: python generate_test_images.py")
        return

    all_results = {}

    # T3: Quantum SWAP Permutation
    try:
        all_results['T3'] = run_task_t3()
    except Exception as e:
        print(f"\nT3 FAILED: {e}")
        all_results['T3'] = {'error': str(e)}

    # T4: AerSimulator + Noise
    try:
        all_results['T4'] = run_task_t4()
    except Exception as e:
        print(f"\nT4 FAILED: {e}")
        all_results['T4'] = {'error': str(e)}

    # T5: Resource Analysis
    try:
        all_results['T5'] = run_task_t5()
    except Exception as e:
        print(f"\nT5 FAILED: {e}")
        all_results['T5'] = {'error': str(e)}

    # T6: RGB Extension
    try:
        all_results['T6'] = run_task_t6()
    except Exception as e:
        print(f"\nT6 FAILED: {e}")
        all_results['T6'] = {'error': str(e)}

    # T7: Noise Attacks
    try:
        all_results['T7'] = run_task_t7(images)
    except Exception as e:
        print(f"\nT7 FAILED: {e}")
        all_results['T7'] = {'error': str(e)}

    # T8: Comparative Benchmarks
    try:
        all_results['T8'] = run_task_t8(images)
    except Exception as e:
        print(f"\nT8 FAILED: {e}")
        all_results['T8'] = {'error': str(e)}

    # Summary
    total_time = time.perf_counter() - overall_start
    print("\n" + "=" * 70)
    print(f"ALL TASKS COMPLETE — Total time: {total_time:.1f}s")
    print("=" * 70)
    print("\nOutput directories:")
    print("  results/quantum/       — T3/T4/T5 quantum analysis results")
    print("  results/rgb/           — T6 RGB encryption results")
    print("  results/noise_attacks/ — T7 noise attack results")
    print("  results/benchmarks/    — T8 comparison benchmarks")


if __name__ == '__main__':
    main()
