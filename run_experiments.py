"""
Comprehensive Experiment Runner
================================
Runs all security evaluation experiments from the paper (Section V)
and generates plots and result tables.

Experiments:
1. Visual Encryption/Decryption Analysis (Fig. 17)
2. Histogram Analysis (Fig. 18)
3. Correlation Analysis (Fig. 19, Table I)
4. Entropy Analysis (Table II)
5. Differential Attack Analysis - NPCR & UACI (Table III, Fig. 20)
6. Key Sensitivity Analysis
7. Occlusion Attack Resilience (Fig. 21)
8. Execution Time Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import json

from src.encryption import encrypt, decrypt, EncryptionKey, load_image, save_image
from src.evaluation import (
    compute_entropy, compute_correlation, compute_npcr, compute_uaci,
    compute_mse, compute_psnr, histogram_uniformity_chi_squared,
    full_evaluation, differential_attack_analysis,
    key_sensitivity_analysis, occlusion_attack_analysis,
)


def setup_directories():
    """Create output directories."""
    dirs = ['results', 'results/images', 'results/plots', 'results/tables']
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


def experiment_1_visual_analysis(images: dict, key: EncryptionKey):
    """
    Experiment 1: Visual Encryption/Decryption Analysis (reproduces Fig. 17).
    Shows original, encrypted, and decrypted images side by side.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Visual Encryption/Decryption Analysis")
    print("="*70)

    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for i, (name, img) in enumerate(images.items()):
        cipher, _, _ = encrypt(img, key)
        decrypted, _ = decrypt(cipher, key)

        # Save individual images
        save_image(img, f'results/images/{name}_original.png')
        save_image(cipher, f'results/images/{name}_encrypted.png')
        save_image(decrypted, f'results/images/{name}_decrypted.png')

        # Plot
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'{name}\n(Original)', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(cipher, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'{name}\n(Encrypted)', fontsize=10)
        axes[1, i].axis('off')

        axes[2, i].imshow(decrypted, cmap='gray', vmin=0, vmax=255)
        axes[2, i].set_title(f'{name}\n(Decrypted)', fontsize=10)
        axes[2, i].axis('off')

        # Verify recovery
        recovery_ok = np.array_equal(img, decrypted)
        print(f"  {name}: Recovery = {'PERFECT' if recovery_ok else 'FAILED'}")

    plt.suptitle('Visual Encryption/Decryption Analysis\n(Reproduced from Paper Fig. 17)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp1_visual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp1_visual_analysis.png")


def experiment_2_histogram_analysis(images: dict, key: EncryptionKey):
    """
    Experiment 2: Histogram Analysis (reproduces Fig. 18).
    Compare histograms of plaintext and ciphertext images.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Histogram Analysis")
    print("="*70)

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, img) in enumerate(images.items()):
        cipher, _, _ = encrypt(img, key)

        chi2_orig = histogram_uniformity_chi_squared(img)
        chi2_enc = histogram_uniformity_chi_squared(cipher)

        # Plaintext histogram
        axes[0, i].hist(img.flatten(), bins=256, range=(0, 255),
                        color='steelblue', alpha=0.7, density=True)
        axes[0, i].set_title(f'{name}\n(Plaintext)', fontsize=9)
        axes[0, i].set_xlabel('Pixel Intensity')
        axes[0, i].set_ylabel('Frequency')

        # Ciphertext histogram
        axes[1, i].hist(cipher.flatten(), bins=256, range=(0, 255),
                        color='crimson', alpha=0.7, density=True)
        axes[1, i].set_title(f'{name}\n(Ciphertext)', fontsize=9)
        axes[1, i].set_xlabel('Pixel Intensity')
        axes[1, i].set_ylabel('Frequency')

        print(f"  {name}: Chi2(original)={chi2_orig:.1f}, Chi2(encrypted)={chi2_enc:.1f}")

    plt.suptitle('Histogram Analysis\n(Reproduced from Paper Fig. 18)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp2_histogram_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp2_histogram_analysis.png")


def experiment_3_correlation_analysis(images: dict, key: EncryptionKey):
    """
    Experiment 3: Correlation Analysis (reproduces Fig. 19, Table I).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Correlation Analysis")
    print("="*70)

    table_data = []
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, img) in enumerate(images.items()):
        cipher, _, _ = encrypt(img, key)

        # Compute correlations
        corr_orig = {
            'H': compute_correlation(img, 'horizontal'),
            'V': compute_correlation(img, 'vertical'),
            'D': compute_correlation(img, 'diagonal'),
        }
        corr_enc = {
            'H': compute_correlation(cipher, 'horizontal'),
            'V': compute_correlation(cipher, 'vertical'),
            'D': compute_correlation(cipher, 'diagonal'),
        }

        table_data.append({
            'Image': name,
            'Original H': f"{corr_orig['H']:.4f}",
            'Original V': f"{corr_orig['V']:.4f}",
            'Original D': f"{corr_orig['D']:.4f}",
            'Encrypted H': f"{corr_enc['H']:.4f}",
            'Encrypted V': f"{corr_enc['V']:.4f}",
            'Encrypted D': f"{corr_enc['D']:.4f}",
        })

        # Scatter plots for horizontal correlation
        rng = np.random.RandomState(42)
        n_samples = 3000
        xs = rng.randint(0, img.shape[1] - 1, n_samples)
        ys = rng.randint(0, img.shape[0], n_samples)

        # Original
        p1_orig = img[ys, xs].astype(float)
        p2_orig = img[ys, xs + 1].astype(float)
        axes[0, i].scatter(p1_orig, p2_orig, s=1, alpha=0.3, color='steelblue')
        axes[0, i].set_title(f'{name} (Original)\nH={corr_orig["H"]:.4f}', fontsize=9)
        axes[0, i].set_xlabel('Pixel (x,y)')
        axes[0, i].set_ylabel('Pixel (x+1,y)')
        axes[0, i].set_xlim(0, 255)
        axes[0, i].set_ylim(0, 255)

        # Encrypted
        p1_enc = cipher[ys, xs].astype(float)
        p2_enc = cipher[ys, xs + 1].astype(float)
        axes[1, i].scatter(p1_enc, p2_enc, s=1, alpha=0.3, color='crimson')
        axes[1, i].set_title(f'{name} (Encrypted)\nH={corr_enc["H"]:.4f}', fontsize=9)
        axes[1, i].set_xlabel('Pixel (x,y)')
        axes[1, i].set_ylabel('Pixel (x+1,y)')
        axes[1, i].set_xlim(0, 255)
        axes[1, i].set_ylim(0, 255)

        print(f"  {name}:")
        print(f"    Original:  H={corr_orig['H']:.4f}, V={corr_orig['V']:.4f}, D={corr_orig['D']:.4f}")
        print(f"    Encrypted: H={corr_enc['H']:.4f}, V={corr_enc['V']:.4f}, D={corr_enc['D']:.4f}")

    plt.suptitle('Correlation Analysis\n(Reproduced from Paper Fig. 19)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp3_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp3_correlation_analysis.png")

    return table_data


def experiment_4_entropy_analysis(images: dict, key: EncryptionKey):
    """
    Experiment 4: Entropy Analysis (reproduces Table II).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Entropy Analysis")
    print("="*70)

    table_data = []
    for name, img in images.items():
        cipher, _, _ = encrypt(img, key)

        ent_orig = compute_entropy(img)
        ent_enc = compute_entropy(cipher)

        table_data.append({
            'Image': name,
            'Plaintext Entropy': f"{ent_orig:.4f}",
            'Ciphertext Entropy': f"{ent_enc:.4f}",
            'Ideal': '8.0000',
        })

        print(f"  {name}: Original={ent_orig:.4f}, Encrypted={ent_enc:.4f} (ideal=8.0)")

    return table_data


def experiment_5_differential_attack(images: dict, key: EncryptionKey):
    """
    Experiment 5: Differential Attack Analysis - NPCR & UACI (reproduces Fig. 20, Table III).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Differential Attack Analysis (NPCR & UACI)")
    print("="*70)

    table_data = []
    n = len(images)
    fig, axes = plt.subplots(4, n, figsize=(4*n, 16))
    if n == 1:
        axes = axes.reshape(4, 1)

    for i, (name, img) in enumerate(images.items()):
        result = differential_attack_analysis(img, encrypt, key)

        table_data.append({
            'Image': name,
            'NPCR (%)': f"{result['npcr']:.2f}",
            'UACI (%)': f"{result['uaci']:.2f}",
            'NPCR Ideal': '99.6',
            'UACI Ideal': '33.46',
        })

        cipher1, _, _ = encrypt(img, key)
        altered = img.copy()
        altered[0, 0] ^= 1
        cipher2, _, _ = encrypt(altered, key)

        # Row 1: Original plaintext
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'{name}\n(Plaintext)', fontsize=9)
        axes[0, i].axis('off')

        # Row 2: Cipher of original
        axes[1, i].imshow(cipher1, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'Cipher\n(Original)', fontsize=9)
        axes[1, i].axis('off')

        # Row 3: Cipher of 1-bit altered
        axes[2, i].imshow(cipher2, cmap='gray', vmin=0, vmax=255)
        axes[2, i].set_title(f'Cipher\n(1-bit altered)', fontsize=9)
        axes[2, i].axis('off')

        # Row 4: Difference
        axes[3, i].imshow(result['cipher_diff'], cmap='hot', vmin=0, vmax=255)
        axes[3, i].set_title(f'Difference\nNPCR={result["npcr"]:.1f}%', fontsize=9)
        axes[3, i].axis('off')

        print(f"  {name}: NPCR={result['npcr']:.2f}%, UACI={result['uaci']:.2f}%")

    plt.suptitle('Differential Attack Analysis\n(Reproduced from Paper Fig. 20)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp5_differential_attack.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp5_differential_attack.png")

    return table_data


def experiment_6_key_sensitivity(images: dict, key: EncryptionKey):
    """
    Experiment 6: Key Sensitivity Analysis.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Key Sensitivity Analysis")
    print("="*70)

    # Use first image for detailed analysis
    name, img = next(iter(images.items()))
    print(f"  Using image: {name}")

    deltas = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    results_list = []

    for delta in deltas:
        result = key_sensitivity_analysis(img, encrypt, decrypt, key, delta)
        results_list.append({
            'delta': delta,
            'npcr': result['npcr_wrong_key'],
            'mse': result['mse_wrong_key'],
            'psnr': result['psnr_wrong_key'],
        })
        print(f"  delta={delta:.0e}: NPCR={result['npcr_wrong_key']:.2f}%, "
              f"MSE={result['mse_wrong_key']:.1f}, PSNR={result['psnr_wrong_key']:.2f}dB")

    # Plot key sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_labels = [f"1e{int(np.log10(d))}" for d in deltas]
    npcrs = [r['npcr'] for r in results_list]
    psnrs = [r['psnr'] for r in results_list]

    ax1.bar(x_labels, npcrs, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Key Perturbation (delta)')
    ax1.set_ylabel('NPCR (%)')
    ax1.set_title('Key Sensitivity: NPCR vs Key Change')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=99.6, color='red', linestyle='--', alpha=0.7, label='Ideal (99.6%)')
    ax1.legend()

    ax2.bar(x_labels, psnrs, color='crimson', alpha=0.8)
    ax2.set_xlabel('Key Perturbation (delta)')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Key Sensitivity: PSNR of Wrong Decryption')

    plt.suptitle('Key Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp6_key_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp6_key_sensitivity.png")

    # Show visual comparison for one delta
    result = key_sensitivity_analysis(img, encrypt, decrypt, key, 1e-10)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cipher, _, _ = encrypt(img, key)

    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cipher, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Encrypted')
    axes[1].axis('off')

    correct_dec, _ = decrypt(cipher, key)
    axes[2].imshow(correct_dec, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Correct Key\n(Perfect Recovery)')
    axes[2].axis('off')

    axes[3].imshow(result['wrong_decrypted'], cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f'Wrong Key (delta=1e-10)\n(Noise)')
    axes[3].axis('off')

    plt.suptitle('Key Sensitivity: Correct vs Wrong Key Decryption',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp6_key_sensitivity_visual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp6_key_sensitivity_visual.png")


def experiment_7_occlusion_attack(images: dict, key: EncryptionKey):
    """
    Experiment 7: Occlusion Attack Resilience (reproduces Fig. 21).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: Occlusion Attack Resilience")
    print("="*70)

    name, img = next(iter(images.items()))
    print(f"  Using image: {name}")

    cipher, _, _ = encrypt(img, key)
    results = occlusion_attack_analysis(img, cipher, decrypt, key,
                                         [10, 25, 50, 60])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Row 1: Original and occluded ciphers
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Row 2: Recovered images
    correct_dec, _ = decrypt(cipher, key)
    axes[1, 0].imshow(correct_dec, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Full Recovery')
    axes[1, 0].axis('off')

    for j, pct in enumerate([10, 25, 50, 60]):
        key_name = f'occlusion_{pct}pct'
        r = results[key_name]

        axes[0, j+1].imshow(r['occluded_cipher'], cmap='gray', vmin=0, vmax=255)
        axes[0, j+1].set_title(f'Occluded ({pct}%)')
        axes[0, j+1].axis('off')

        axes[1, j+1].imshow(r['recovered'], cmap='gray', vmin=0, vmax=255)
        axes[1, j+1].set_title(f'Recovered\nPSNR={r["psnr"]:.1f}dB')
        axes[1, j+1].axis('off')

        print(f"  {pct}% occlusion: MSE={r['mse']:.1f}, PSNR={r['psnr']:.2f}dB")

    plt.suptitle('Occlusion Attack Resilience\n(Reproduced from Paper Fig. 21)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/exp7_occlusion_attack.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/plots/exp7_occlusion_attack.png")


def experiment_8_timing_analysis(images: dict, key: EncryptionKey):
    """
    Experiment 8: Execution Time Analysis.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 8: Execution Time Analysis")
    print("="*70)

    table_data = []
    for name, img in images.items():
        # Run 3 iterations for averaging
        enc_times = []
        dec_times = []
        for _ in range(3):
            cipher, _, enc_t = encrypt(img, key)
            _, dec_t = decrypt(cipher, key)
            enc_times.append(enc_t['total'])
            dec_times.append(dec_t['total'])

        avg_enc = np.mean(enc_times)
        avg_dec = np.mean(dec_times)

        table_data.append({
            'Image': name,
            'Size': f"{img.shape[0]}x{img.shape[1]}",
            'Encryption (s)': f"{avg_enc:.3f}",
            'Decryption (s)': f"{avg_dec:.3f}",
            'Total (s)': f"{avg_enc + avg_dec:.3f}",
        })

        print(f"  {name} ({img.shape}): Enc={avg_enc:.3f}s, Dec={avg_dec:.3f}s")

    return table_data


def save_results_table(table_data: list, filename: str, title: str):
    """Save results as a formatted text table."""
    if not table_data:
        return

    # Compute column widths
    headers = list(table_data[0].keys())
    widths = {h: max(len(h), max(len(str(row[h])) for row in table_data))
              for h in headers}

    # Format table
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(title)
    lines.append(f"{'='*70}")

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in table_data:
        row_line = " | ".join(str(row[h]).ljust(widths[h]) for h in headers)
        lines.append(row_line)

    lines.append("")

    table_text = "\n".join(lines)
    print(table_text)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(table_text)


def main():
    """Run all experiments."""
    print("=" * 70)
    print("CHAOTIC QUANTUM IMAGE ENCRYPTION - EXPERIMENT SUITE")
    print("Implementing: Khan et al., IEEE TCE, 2024")
    print("=" * 70)

    setup_directories()

    # Load test images
    print("\nLoading test images...")
    images = load_test_images('test_images', target_size=(256, 256))

    if not images:
        print("ERROR: No test images found in test_images/")
        print("Run: python generate_test_images.py first")
        return

    # Use default key from the paper
    key = EncryptionKey()
    print(f"\nUsing encryption key with De Jong parameters:")
    print(f"  Phase 1: a={key.dj1_a}, b={key.dj1_b}, c={key.dj1_c}, d={key.dj1_d}")
    print(f"  Phase 2: a={key.dj2_a}, b={key.dj2_b}, c={key.dj2_c}, d={key.dj2_d}")
    print(f"  VdP: h={key.vdp_h}, mu={key.vdp_mu}")

    # Run experiments
    experiment_1_visual_analysis(images, key)
    experiment_2_histogram_analysis(images, key)
    corr_table = experiment_3_correlation_analysis(images, key)
    entropy_table = experiment_4_entropy_analysis(images, key)
    npcr_uaci_table = experiment_5_differential_attack(images, key)
    experiment_6_key_sensitivity(images, key)
    experiment_7_occlusion_attack(images, key)
    timing_table = experiment_8_timing_analysis(images, key)

    # Save tables
    save_results_table(corr_table, 'results/tables/table1_correlation.txt',
                       'Table I: Correlation Analysis')
    save_results_table(entropy_table, 'results/tables/table2_entropy.txt',
                       'Table II: Entropy Analysis')
    save_results_table(npcr_uaci_table, 'results/tables/table3_npcr_uaci.txt',
                       'Table III: NPCR and UACI Analysis')
    save_results_table(timing_table, 'results/tables/table4_timing.txt',
                       'Table IV: Execution Time Analysis')

    # Save all results as JSON
    all_results = {
        'correlation': corr_table,
        'entropy': entropy_table,
        'npcr_uaci': npcr_uaci_table,
        'timing': timing_table,
    }
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("\nOutput directories:")
    print("  results/plots/   - Experiment plots (PNG)")
    print("  results/images/  - Encrypted/decrypted images")
    print("  results/tables/  - Result tables")
    print("  results/all_results.json - All numerical results")


if __name__ == '__main__':
    main()
