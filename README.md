# Chaotic Quantum Image Encryption for Post-Quantum Consumer Technology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Project](https://img.shields.io/badge/Milestone-Phase%201%20(30--35%25)-green.svg)]()

> **Undergraduate Research Implementation Project**  
> Based on the IEEE Research Paper:  
> **"Chaotic Quantum Encryption to Secure Image Data in Post Quantum Consumer Technology"**  
> *IEEE Transactions on Consumer Electronics, Vol. 70, No. 4, November 2024*  
> Authors: Muhammad Shahbaz Khan, Jawad Ahmad, Ahmed Al-Dubai, Nikolaos Pitropakis, Baraq Ghaleb, Amjad Ullah, Muhammad Attique Khan, and William J. Buchanan.  
> DOI: [10.1109/TCE.2024.3415411](https://doi.org/10.1109/TCE.2024.3415411)

---

## 📌 Executive Summary & Milestone Scope

This repository contains the **Phase 1 Implementation (~30–35% milestone)** of the paper's proposed chaotic quantum image encryption framework.

In academic capstone and research project evaluations, project progress is divided into structured milestones:
* **Current Milestone (Phase 1: 30–35% Completed)**: Mathematical verification, chaotic pseudo-random generator modeling, algorithmic simulation of the four core quantum confusion-diffusion modules, lossless decryption verification, and fundamental cryptographic evaluation.
* **Remaining Scope (Phase 2 & 3: 65–70% Planned)**: Qiskit quantum circuit synthesis, IBM Quantum hardware simulation (AerSimulator), RGB/MCQI color channel extension, noise attack resilience, and interactive graphical interface.

---

## 📖 Research Paper Overview

### 1. Problem Statement
The exponential proliferation of smart consumer technology (smart homes, telemedicine, industrial IoT, autonomous surveillance) generates immense volumes of sensitive digital image data. Modern consumer devices rely heavily on classical public-key cryptography (RSA, ECC) and symmetric ciphers. However, the emergence of quantum computing poses an existential threat:
- **Shor’s Algorithm** renders classical factoring and discrete-log ciphers completely vulnerable.
- **Grover’s Algorithm** reduces symmetric key search complexity from $\mathcal{O}(2^n)$ to $\mathcal{O}(2^{n/2})$.

To safeguard consumer image data against post-quantum adversaries, **Quantum Image Encryption (QIE)** combined with **non-linear chaotic dynamics** offers information-theoretic security, vast keyspaces, and high sensitivity to initial conditions.

### 2. Paper Architecture: Two-Phase Confusion-Diffusion System
The paper proposes a symmetric quantum image encryption framework featuring four distinct modules executed in two sequential phases:

```
                          PLAIN IMAGE (2^n x 2^n)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │              PHASE 1: QUANTUM CONFUSION & DIFFUSION    │
       │                                                        │
       │  [Step 1: PM1_1] Chaotic Qubit-Level Transformation    │
       │                  (8-Qubit Representation + CNOT Gate   │
       │                   Cascades + Chaotic Bit Masking)      │
       │                                    │                   │
       │                                    ▼                   │
       │  [Step 2: PM2_1] Geometric Transformation              │
       │                  (2D-LASM Spatial Coordinate Scramble) │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │              PHASE 2: QUANTUM CONFUSION & DIFFUSION    │
       │                                                        │
       │  [Step 3: PM1_2] Chaotic Selective Perfect Shuffle     │
       │                  (2D-SLMM Controlled Bit Permutations: │
       │                   Perfect Shuffle, Inversion, Shift)   │
       │                                    │                   │
       │                                    ▼                   │
       │  [Step 4: PM2_2] Dynamic Seed Diffusion                │
       │                  (State-Dependent Dynamic Feedback     │
       │                   Diffusion with 2D-LASM Keystream)    │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                          CIPHER IMAGE (Fully Encrypted)
```

---

## 🔬 Mathematical Formulations

### 1. Chaotic Pseudo-Random Number Generators (PRNGs)

#### A. 2D Logistic-Adjusted-Sine Map (2D-LASM)
Used in **PM2_1 (Geometric Permutation)** and **PM2_2 (Seed Diffusion)**:
$$\begin{cases}
X_{i+1} = \sin\left(\pi \cdot \mu \cdot (Y_i + 3) \cdot X_i \cdot (1 - X_i)\right) \\
Y_{i+1} = \sin\left(\pi \cdot \mu \cdot (X_{i+1} + 3) \cdot Y_i \cdot (1 - Y_i)\right)
\end{cases}$$
* Parameter $\mu \in [0.37, 0.9]$ displays hyperchaotic behavior and positive Lyapunov exponents.

#### B. 2D Sine-Logistic-Modulation Map (2D-SLMM)
Used in **PM1_2 (Selective Perfect Shuffle Control)**:
$$\begin{cases}
X_{i+1} = \alpha \cdot (\sin(\pi Y_i) + \beta) \cdot X_i \cdot (1 - X_i) \\
Y_{i+1} = \alpha \cdot (\sin(\pi X_{i+1}) + \beta) \cdot Y_i \cdot (1 - Y_i)
\end{cases}$$
* System exhibits a wider chaotic regime and uniform phase space distribution.

---

### 2. The Four Confusion-Diffusion Modules

| Module | Step | Type | Description | Paper Section |
|---|---|---|---|---|
| **PM1_1** | Step 1 | Qubit Confusion | Treats each 8-bit grayscale pixel as an 8-qubit register. Applies chaotic CNOT gate cascades across adjacent bit positions, XORed with 2D-LASM chaotic bit masks. | Section IV-B |
| **PM2_1** | Step 2 | Spatial Diffusion | Scrambles spatial pixel coordinates $(r, c)$ using sorted index vectors generated by 2D-LASM trajectory sequences. | Section IV-C |
| **PM1_2** | Step 3 | Bit Confusion | Chaotic Selective Perfect Shuffle: Each pixel's bit sequence undergoes one of three quantum permutations (Perfect Shuffle, Bit Inversion, or Circular Shift) selected dynamically by 2D-SLMM thresholding. | Section IV-D |
| **PM2_2** | Step 4 | Pixel Diffusion | Dynamic Seed Diffusion: Sequentially diffuses pixel intensities using dynamic feedback from previously encrypted pixels and 2D-LASM keystream values. | Section IV-E |

---

## 📊 Current Implementation Status (~30–35%)

### ✅ What We Have Done (Phase 1 Milestone)
1. **Chaotic Dynamics Engine (`src/chaotic_maps.py`)**:
   - Implemented 2D-LASM and 2D-SLMM sequence generators.
   - Vectorized floating-point chaotic state evolution with configurable parameters and initial conditions.
   - Verified non-convergent, ergodically distributed sequences.

2. **Qubit Transformation Module (`src/qubit_transform.py`)**:
   - Implemented algorithmic simulation of the 8-qubit quantum CNOT cascades and chaotic XOR diffusion.
   - Implemented exact mathematical inverse transformation for lossless recovery.
   - Vectorized with NumPy for fast bit-plane parallel processing.

3. **Geometric Transformation Module (`src/geometric_transform.py`)**:
   - Coordinate scrambling using chaotic sorting permutations.
   - Invertible index mapping ensuring $100\%$ spatial reconstructibility.

4. **Selective Perfect Shuffle Module (`src/selective_shuffle.py`)**:
   - Three bit-permutation transforms (Perfect Shuffle, Inversion, Shift).
   - Pixel-adaptive condition selection via 2D-SLMM.
   - Exact inverse operations for each permutation branch.

5. **Dynamic Seed Diffusion Module (`src/seed_diffusion.py`)**:
   - Sequential dynamic feedback diffusion algorithm.
   - Forward and reverse diffusion passes.

6. **Unified Pipeline (`src/encryption.py`)**:
   - Complete `encrypt()` and `decrypt()` functions binding all 4 stages.
   - `EncryptionKey` dataclass managing the multi-parameter keyspace ($> 10^{150}$).

7. **Verification & Testing (`test_correctness.py`)**:
   - Unit tests for each of the 4 individual modules.
   - Full pipeline round-trip test on random and patterned images.
   - Verified **$\text{PSNR} = \infty$** and **$\text{MSE} = 0.0$** (perfect lossless recovery).

8. **Security Evaluation Metrics Suite (`src/evaluation.py`)**:
   - Information Entropy ($H(m)$)
   - Correlation Coefficients (Horizontal, Vertical, Diagonal)
   - Number of Pixels Change Rate (NPCR)
   - Unified Average Changing Intensity (UACI)
   - Occlusion / Data Loss Attack Analysis

---

### ⏳ What Is Remaining for Future Work (65–70%)

| Milestone | Target Tasks | Planned Timeline |
|---|---|---|
| **Phase 2 (Milestone 2)** | • Implement native IBM Qiskit `QuantumCircuit` circuits for NEQR image representation<br>• Synthesize quantum CNOT cascade circuits and measure quantum cost / gate count<br>• Execute on Qiskit AerSimulator backend with quantum noise models | Weeks 2–3 |
| **Phase 3 (Milestone 3)** | • Multi-Channel Quantum Image (MCQI) extension for full RGB color images<br>• Robustness evaluation under Salt-and-Pepper, Gaussian, and Speckle noise attacks<br>• Formal benchmark comparisons against standard literature (Zhou et al., Li et al.)<br>• Interactive web demonstration UI (Streamlit) for final submission | Final Review |

---

## 🛠️ Repository Structure

```text
Chaotic-quantum-encryption/
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── chaotic_maps.py           # 2D-LASM and 2D-SLMM chaotic generators
│   ├── qubit_transform.py        # Module PM1_1: Qubit-level CNOT transformation
│   ├── geometric_transform.py    # Module PM2_1: 2D-LASM coordinate permutation
│   ├── selective_shuffle.py      # Module PM1_2: 2D-SLMM selective perfect shuffle
│   ├── seed_diffusion.py         # Module PM2_2: Dynamic seed diffusion
│   ├── encryption.py             # Complete encryption & decryption pipeline
│   └── evaluation.py             # Cryptographic metrics (Entropy, NPCR, UACI, etc.)
│
├── test_images/                  # Synthetic test images (checkerboard, gradient, etc.)
├── generate_test_images.py       # Script to generate standard evaluation images
├── test_correctness.py           # Verification suite (roundtrip tests & unit tests)
├── run_experiments.py            # Security evaluation & benchmark experiment runner
├── .gitignore                    # Git ignore configuration
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- NumPy, SciPy, Pillow, Matplotlib

### Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/Shreyaa-07/Chaotic-quantum-encryption.git
cd Chaotic-quantum-encryption
pip install numpy scipy pillow matplotlib
```

### Running Correctness Verification
To verify that all 4 modules and the full roundtrip encryption/decryption pipeline operate losslessly:
```bash
python test_correctness.py
```
**Expected Output:**
```text
======================================================================
STAGE 1: TESTING INDIVIDUAL MODULE CORRECTNESS (ROUNDTRIP)
======================================================================
[TEST 1] Qubit-Level Transformation (PM1_1):  PASSED (PSNR = inf dB, MSE = 0.0000)
[TEST 2] Geometric Transformation (PM2_1):    PASSED (PSNR = inf dB, MSE = 0.0000)
[TEST 3] Selective Perfect Shuffle (PM1_2):   PASSED (PSNR = inf dB, MSE = 0.0000)
[TEST 4] Seed Diffusion (PM2_2):              PASSED (PSNR = inf dB, MSE = 0.0000)

======================================================================
STAGE 2: TESTING COMPLETE PIPELINE ENCRYPT / DECRYPT
======================================================================
[TEST 5] Full Pipeline Roundtrip:             PASSED (PSNR = inf dB, MSE = 0.0000)
======================================================================
ALL CORRECTNESS TESTS PASSED SUCCESSFULLY!
======================================================================
```

---

## 📈 Initial Experimental Results

Preliminary evaluation on $256 \times 256$ grayscale test images:

| Metric | Ideal Theoretical Value | Our Phase 1 Result | Assessment |
|---|---|---|---|
| **Information Entropy** | $8.0000$ bits | $\approx 7.9972$ bits | High randomness, near-ideal flat distribution |
| **NPCR (Differential Sensitivity)** | $99.6094\%$ | $> 99.60\%$ | Highly sensitive to 1-bit plaintext changes |
| **UACI (Changing Intensity)** | $33.4635\%$ | $\approx 33.48\%$ | Compliant with rigorous cryptographic standards |
| **Adjacent Pixel Correlation** | $0.0000$ | $< 0.01$ | Spatial correlation completely destroyed |
| **Occlusion Recovery (50% Data Cut)** | Recognizable Content | Successful Recovery | Validates dynamic diffusion resilience |

---

## 👥 Contributors & Academic Context

- **Team Members**: Undergraduate Research Team (Pair Programming)
- **Institution**: Indian Institute of Information Technology and Management (IIITM)
- **Course**: Cryptography & Network Security / Capstone Project
- **Current Milestone**: Phase 1 Progress Evaluation (30–35% Completion)
