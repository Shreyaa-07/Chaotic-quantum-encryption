"""
Comprehensive correctness tests for the full encryption/decryption pipeline.
Tests: chaotic generator validity, individual module roundtrips, full pipeline,
and key sensitivity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.encryption import encrypt, decrypt, EncryptionKey
from src.chaotic_maps import (
    generate_qubit_key_lasm,
    generate_selection_key_slmm,
    generate_diffusion_keystream_lasm,
    lasm_2d,
    slmm_2d,
)
from src.qubit_transform import qubit_transform_encrypt, qubit_transform_decrypt
from src.geometric_transform import geometric_transform_encrypt, geometric_transform_decrypt
from src.selective_shuffle import selective_shuffle_encrypt, selective_shuffle_decrypt
from src.seed_diffusion import seed_diffusion_encrypt, seed_diffusion_decrypt

SEP = "=" * 60

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

def psnr_label(a, b):
    m = mse(a, b)
    return "inf" if m == 0.0 else f"{10 * np.log10(255**2 / m):.2f} dB"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0: Validate the chaotic maps themselves
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STAGE 0: VALIDATING PAPER-CORRECT CHAOTIC MAPS")
print(SEP)

# 2D-LASM: check output range and non-convergence
x_l, y_l = lasm_2d(x0=0.12, y0=0.34, mu=0.75, n_iterations=10000)
assert x_l.min() >= -1.0 and x_l.max() <= 1.0, "LASM X out of sine range"
assert y_l.min() >= -1.0 and y_l.max() <= 1.0, "LASM Y out of sine range"
assert x_l.std() > 0.05, "LASM X sequence appears constant — not chaotic"
assert y_l.std() > 0.05, "LASM Y sequence appears constant — not chaotic"
print(f"  [OK] 2D-LASM — X range [{x_l.min():.4f}, {x_l.max():.4f}],"
      f"  std={x_l.std():.4f}")

# 2D-SLMM: check output range and non-convergence
x_s, y_s = slmm_2d(x0=0.23, y0=0.67, alpha=3.20, beta=0.20, n_iterations=10000)
finite_xs = x_s[np.isfinite(x_s)]
assert len(finite_xs) > 100, "SLMM produced almost no finite values"
assert finite_xs.std() > 0.001, "SLMM X sequence appears constant -- not chaotic"
print(f"  [OK] 2D-SLMM -- finite samples={len(finite_xs)}/10000,"
      f"  std of finite values={finite_xs.std():.4f}")

# Key sensitivity: changing mu by 1e-14 should produce completely different seq
x_ref, _ = lasm_2d(x0=0.12, y0=0.34, mu=0.75,           n_iterations=1000)
x_alt, _ = lasm_2d(x0=0.12, y0=0.34, mu=0.75 + 1e-14,   n_iterations=1000)
diverge_idx = np.where(np.abs(x_ref - x_alt) > 0.01)[0]
assert len(diverge_idx) > 0, "LASM shows no sensitivity to IC perturbation"
print(f"  [OK] 2D-LASM key sensitivity — sequences diverge at iteration {diverge_idx[0]}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Individual module roundtrips
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STAGE 1: TESTING INDIVIDUAL MODULE CORRECTNESS (ROUNDTRIP)")
print(SEP)

rng = np.random.RandomState(42)
test_img = rng.randint(0, 256, (16, 16), dtype=np.uint8)

key = EncryptionKey()
qkey = generate_qubit_key_lasm(key.lasm1_x0, key.lasm1_y0, key.lasm1_mu, 16, 16)
skey = generate_selection_key_slmm(key.slmm_x0, key.slmm_y0,
                                    key.slmm_alpha, key.slmm_beta, 16, 16)
dmat, dmat_bwd = generate_diffusion_keystream_lasm(key.lasm2_x0, key.lasm2_y0, key.lasm2_mu, 16, 16)

# [TEST 1] Qubit transform
enc1 = qubit_transform_encrypt(test_img, qkey)
dec1 = qubit_transform_decrypt(enc1, qkey)
ok1 = np.array_equal(test_img, dec1)
print(f"[TEST 1] Qubit-Level Transform (PM1_1):        {'PASSED' if ok1 else 'FAILED'}"
      f"  (PSNR = {psnr_label(test_img, dec1)}, MSE = {mse(test_img, dec1):.4f})")
assert ok1

# [TEST 2] Geometric transform
enc2 = geometric_transform_encrypt(test_img)
dec2 = geometric_transform_decrypt(enc2)
ok2 = np.array_equal(test_img, dec2)
print(f"[TEST 2] Geometric Transform (PM2_1):          {'PASSED' if ok2 else 'FAILED'}"
      f"  (PSNR = {psnr_label(test_img, dec2)}, MSE = {mse(test_img, dec2):.4f})")
assert ok2

# [TEST 3] Selective shuffle
enc3 = selective_shuffle_encrypt(test_img, skey)
dec3 = selective_shuffle_decrypt(enc3, skey)
ok3 = np.array_equal(test_img, dec3)
print(f"[TEST 3] Selective Perfect Shuffle (PM1_2):    {'PASSED' if ok3 else 'FAILED'}"
      f"  (PSNR = {psnr_label(test_img, dec3)}, MSE = {mse(test_img, dec3):.4f})")
assert ok3

# [TEST 4] Seed diffusion
enc4 = seed_diffusion_encrypt(test_img, dmat, dmat_bwd)
dec4 = seed_diffusion_decrypt(enc4, dmat, dmat_bwd)
ok4 = np.array_equal(test_img, dec4)
print(f"[TEST 4] Seed Diffusion (PM2_2):               {'PASSED' if ok4 else 'FAILED'}"
      f"  (PSNR = {psnr_label(test_img, dec4)}, MSE = {mse(test_img, dec4):.4f})")
assert ok4

print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Full pipeline roundtrip
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STAGE 2: TESTING COMPLETE PIPELINE ENCRYPT / DECRYPT")
print(SEP)

for label, img in [
    ("8×8 random",      rng.randint(0, 256, (8, 8),   dtype=np.uint8)),
    ("32×32 random",    rng.randint(0, 256, (32, 32),  dtype=np.uint8)),
    ("16×16 all-zeros", np.zeros((16, 16), dtype=np.uint8)),
    ("16×16 all-255",   np.full((16, 16), 255, dtype=np.uint8)),
]:
    cipher, k, enc_t = encrypt(img, key)
    recovered, dec_t = decrypt(cipher, k)

    match = np.array_equal(img, recovered)
    differs = not np.array_equal(img, cipher)
    status = "PASSED" if (match and differs) else "FAILED"
    print(f"[TEST 5] Full Pipeline ({label}):  {status}"
          f"  (PSNR={psnr_label(img, recovered)}, MSE={mse(img, recovered):.4f},"
          f"  enc={enc_t['total']:.4f}s)")
    assert match, f"Roundtrip FAILED for {label}!"
    assert differs, f"Cipher == plaintext for {label}!"

print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Key sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STAGE 3: KEY SENSITIVITY")
print(SEP)

ref_img = rng.randint(0, 256, (64, 64), dtype=np.uint8)
cipher_ref, key_ref, _ = encrypt(ref_img, key)

# Perturb each key parameter and check that decryption fails
wrong_keys = [
    ("lasm1_x0 + 1e-14",   EncryptionKey(lasm1_x0=key.lasm1_x0 + 1e-14)),
    ("lasm1_mu + 1e-14",   EncryptionKey(lasm1_mu=key.lasm1_mu + 1e-14)),
    ("slmm_alpha + 1e-12", EncryptionKey(slmm_alpha=key.slmm_alpha + 1e-12)),
    ("lasm2_y0 + 1e-14",   EncryptionKey(lasm2_y0=key.lasm2_y0 + 1e-14)),
]

for desc, wkey in wrong_keys:
    wrong_dec, _ = decrypt(cipher_ref, wkey)
    recovered = np.array_equal(ref_img, wrong_dec)
    print(f"  [OK] Wrong key ({desc}) -> {'accidentally recovered (BAD)' if recovered else 'did NOT recover original'}")
    assert not recovered, f"Wrong key {desc} accidentally recovered the original!"

print()
print(SEP)
print("ALL TESTS PASSED")
print(SEP)
