"""Quick correctness test for the encryption/decryption pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.encryption import encrypt, decrypt, EncryptionKey

print("=" * 60)
print("CORRECTNESS TEST: Encrypt → Decrypt Roundtrip")
print("=" * 60)

# Test 1: Small synthetic image
print("\n[Test 1] 8×8 synthetic image...")
rng = np.random.RandomState(42)
test_img = rng.randint(0, 256, (8, 8), dtype=np.uint8)
key = EncryptionKey()

cipher, key_used, enc_timing = encrypt(test_img, key, verbose=False)
decrypted, dec_timing = decrypt(cipher, key_used, verbose=False)

assert np.array_equal(test_img, decrypted), "FAIL: Decrypted image != original!"
assert not np.array_equal(test_img, cipher), "FAIL: Cipher image == original!"
print(f"  ✓ Perfect roundtrip recovery (8×8)")
print(f"  ✓ Cipher differs from original")
print(f"  Encryption time: {enc_timing['total']:.4f}s")
print(f"  Decryption time: {dec_timing['total']:.4f}s")

# Test 2: 32×32 image
print("\n[Test 2] 32×32 synthetic image...")
test_img_32 = rng.randint(0, 256, (32, 32), dtype=np.uint8)
cipher_32, key_32, _ = encrypt(test_img_32, key)
decrypted_32, _ = decrypt(cipher_32, key_32)

assert np.array_equal(test_img_32, decrypted_32), "FAIL: 32×32 roundtrip failed!"
print(f"  ✓ Perfect roundtrip recovery (32×32)")

# Test 3: All-zeros image
print("\n[Test 3] 16×16 all-zeros image...")
test_zeros = np.zeros((16, 16), dtype=np.uint8)
cipher_z, key_z, _ = encrypt(test_zeros, key)
decrypted_z, _ = decrypt(cipher_z, key_z)

assert np.array_equal(test_zeros, decrypted_z), "FAIL: Zero image roundtrip failed!"
assert not np.array_equal(test_zeros, cipher_z), "FAIL: Cipher of zeros == zeros!"
print(f"  ✓ Perfect roundtrip recovery (all-zeros)")
print(f"  ✓ Cipher of all-zeros is non-trivial")

# Test 4: All-255 image
print("\n[Test 4] 16×16 all-255 image...")
test_ones = np.full((16, 16), 255, dtype=np.uint8)
cipher_o, key_o, _ = encrypt(test_ones, key)
decrypted_o, _ = decrypt(cipher_o, key_o)

assert np.array_equal(test_ones, decrypted_o), "FAIL: All-255 roundtrip failed!"
print(f"  ✓ Perfect roundtrip recovery (all-255)")

# Test 5: Key sensitivity - wrong key should NOT recover original
print("\n[Test 5] Key sensitivity...")
wrong_key = EncryptionKey(dj1_a=1.646)  # Very slightly different
wrong_decrypted, _ = decrypt(cipher, wrong_key)
assert not np.array_equal(test_img, wrong_decrypted), "FAIL: Wrong key recovered original!"
print(f"  ✓ Wrong key does NOT recover original")

# Test 6: Individual module roundtrips
print("\n[Test 6] Individual module roundtrips...")
from src.qubit_transform import qubit_transform_encrypt, qubit_transform_decrypt
from src.geometric_transform import geometric_transform_encrypt, geometric_transform_decrypt
from src.selective_shuffle import selective_shuffle_encrypt, selective_shuffle_decrypt
from src.seed_diffusion import seed_diffusion_encrypt, seed_diffusion_decrypt
from src.chaotic_maps import generate_chaotic_key_dejong, generate_selection_key_dejong, generate_diffusion_matrix_vanderpol

test_pixel_img = rng.randint(0, 256, (16, 16), dtype=np.uint8)
qkey = generate_chaotic_key_dejong(1.645, 1.905, 0.32, 1.53, 0.1, 0.1, 16, 16)
skey = generate_selection_key_dejong(1.42, -2.28, 2.38, -2.08, 0.1, 0.1, 16, 16)
dmat = generate_diffusion_matrix_vanderpol(0.29, 0.06, 0.5, 0.5, 16, 16)

# Qubit transform
enc1 = qubit_transform_encrypt(test_pixel_img, qkey)
dec1 = qubit_transform_decrypt(enc1, qkey)
assert np.array_equal(test_pixel_img, dec1), "FAIL: Qubit transform roundtrip!"
print(f"  ✓ Qubit transform roundtrip OK")

# Geometric transform
enc2 = geometric_transform_encrypt(test_pixel_img)
dec2 = geometric_transform_decrypt(enc2)
assert np.array_equal(test_pixel_img, dec2), "FAIL: Geometric transform roundtrip!"
print(f"  ✓ Geometric transform roundtrip OK")

# Selective shuffle
enc3 = selective_shuffle_encrypt(test_pixel_img, skey)
dec3 = selective_shuffle_decrypt(enc3, skey)
assert np.array_equal(test_pixel_img, dec3), "FAIL: Selective shuffle roundtrip!"
print(f"  ✓ Selective shuffle roundtrip OK")

# Seed diffusion
enc4 = seed_diffusion_encrypt(test_pixel_img, dmat)
dec4 = seed_diffusion_decrypt(enc4, dmat)
assert np.array_equal(test_pixel_img, dec4), "FAIL: Seed diffusion roundtrip!"
print(f"  ✓ Seed diffusion roundtrip OK")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
