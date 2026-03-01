# ⚙ SugarSense — AI-Driven Equipment Failure Forecasting & Revenue Protection for Sugar Mills

> *From a worn bearing to a farmer's delayed payment — we predict it before it happens.*

**Team Muffin** · Sneha Chakraborty & Divyansh Pathak · **ISMA Sugar-NXT Hackathon 2026**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

---

## Table of Contents

- [The Problem](#the-problem)
- [Our Solution](#our-solution)
- [How It Works — 3-Layer Architecture](#how-it-works--3-layer-architecture)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Step-by-Step Guide](#step-by-step-guide)
- [Technical Details](#technical-details)
- [Key Results](#key-results)
- [Data Sources & Citations](#data-sources--citations)
- [Team](#team)

---

## The Problem

Indian sugar mills run 478 crushing operations (ISMA, Dec 2025) during a narrow 4–5 month season. Inside every mill, roller bearings spin under enormous compressive load — and when they fail, the consequences cascade silently and fast:

| Stage | What Happens |
|-------|-------------|
| ⚙ Bearing degrades | Friction increases, roller gap widens, crushing pressure drops |
| 📉 Recovery rate falls | Less juice per tonne of cane; Brix drops below the critical 10.5% threshold |
| 💰 Revenue bleeds | ABB India (2023) puts unplanned downtime at ₹70L+/hour in Indian industry |
| 🌾 Farmers pay the price | Delayed crushing → delayed FRP payments; aggregate disruption pressures consumer prices |

A **4-hour unplanned stoppage** costs an estimated **₹2.8 Crore+**. Yet today, most mills have no early warning system — bearings are replaced reactively, after failure.

---

## Our Solution

SugarSense is a **3-layer AI system** that connects a worn bearing to a rupee figure on the mill owner's screen — in real time, before failure occurs.

```
Vibration Signal → [Layer 1: 1D CNN] → Health Score & RUL
                          ↓
              [Layer 2: Evaporator ODE] → Recovery Rate Drop
                          ↓
              [Layer 3: Economic Engine] → ₹ Loss Forecast & Action
```

**Example output:** *"Roller Mill #2 — 31 hours remaining. Schedule replacement now. Projected saving: ₹18.7 Lakh."*

---

## How It Works — 3-Layer Architecture

### Layer 1 — Equipment Health (1D CNN)

- Reads vibration sensor data (horizontal + vertical acceleration)
- Extracts 24 time-domain and frequency-domain features per window:
  - **Time domain:** RMS, Peak, Crest Factor, Kurtosis, Skewness, Variance
  - **Frequency domain:** FFT band energies (0–1kHz, 1–5kHz, 5–12.8kHz), Spectral Entropy, Spectral Centroid, High-Frequency Ratio
- A 1D Convolutional Neural Network trained on the **PRONOSTIA/FEMTO** benchmark dataset predicts:
  - **Health Score** (0.0 = new → 1.0 = failed)
  - **Remaining Useful Life (RUL)** in hours

### Layer 2 — Process Impact (Physics Simulation)

Sugar-specific, not a generic proxy. Maps bearing health through the actual process chain:

```
Bearing health → Roller pressure drop → Juice extraction efficiency
              → Mixed-juice Brix deviation → 5-effect evaporator ODE
              → Sugar recovery rate (%) → Crystallisation risk flag
```

Uses `scipy.integrate.odeint` to simulate a falling-film 5-effect evaporator — the standard configuration in Indian mills.

### Layer 3 — Economic Impact Engine

Converts process degradation to rupee figures mill owners can act on:

- **Daily revenue loss** = drop in sugar tonnes × MSP (₹3,600/quintal, Govt 2024-25)
- **48-hour risk window** = revenue loss + probability-weighted emergency downtime cost
- **FRP payment risk** = shortfall in cane payment capacity (days)
- **Maintenance ROI** = savings vs. bearing replacement cost (~₹1.5L)
- **Ethanol diversion flag** = whether the next ethanol batch is at risk

---

## Repository Structure

```
SugarSense/
│
├── SugarSense_Notebook.ipynb      ← Full end-to-end notebook (run this on Colab)
│
├── step1_load_dataset.py           ← PRONOSTIA download, feature extraction, .npz output
├── step2_train_model.py            ← 1D CNN training, evaluation, permutation importance
├── step3_physics_simulation.py     ← Sugar process physics + economic impact engine
├── step4_dashboard.py              ← Gradio interactive dashboard
│
├── docs/
│   ├── SugarSense_Submission.pdf   ← Hackathon presentation deck
│   └── architecture_diagram.png    ← System architecture (optional)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### Option A — Google Colab (Recommended, zero setup)

Open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kweenbee187/SugarSense/blob/main/SugarSense_Notebook.ipynb)

Select **Runtime → Change runtime type → T4 GPU** before running.

### Option B — Run scripts locally

```bash
git clone https://github.com/Kweenbee187/SugarSense.git
cd SugarSense
pip install -r requirements.txt

python step1_load_dataset.py     # ~5-10 min (downloads 280MB dataset)
python step2_train_model.py      # ~10-20 min on GPU
python step3_physics_simulation.py
python step4_dashboard.py        # Opens Gradio dashboard with public URL
```

---

## Step-by-Step Guide

### Step 1 — Load Dataset & Extract Features (`step1_load_dataset.py`)

**What it does:**
- Downloads the PRONOSTIA/FEMTO bearing dataset (~280 MB) from GitHub
- Parses all 17 run-to-failure bearing test runs
- Extracts 24 features from non-overlapping 2560-sample windows (0.1s @ 25.6kHz)
- Assigns normalised RUL labels (0.0 = new, 1.0 = about to fail)
- Saves everything to `/content/sugarsense_features.npz`
- Produces `feature_distribution.png` showing feature evolution across the bearing lifecycle

**Output:** `sugarsense_features.npz` — feature matrix X (N × 24) and labels y (N,)

---

### Step 2 — Train 1D CNN (`step2_train_model.py`)

**What it does:**
- Loads features from Step 1
- Scales features with `StandardScaler`
- Trains a 1D CNN with the following architecture:

```
Input (24, 1)
→ Conv1D(64) × 2 + BN + MaxPool + Dropout(0.2)
→ Conv1D(128) × 2 + BN + GlobalAvgPool + Dropout(0.3)
→ Dense(64) → Dense(32) → Dense(1, sigmoid)
```

- Trains for up to 60 epochs with early stopping (patience=10) and LR reduction
- Evaluates on a held-out 20% test set (MAE, R²)
- Computes **permutation feature importance** (no SHAP/C-extension dependencies)
- Saves model in `.keras` and `.h5` formats, plus scaler and importance arrays

**Output:** `sugarsense_model/` directory containing:
- `best_model.keras` / `best_model.h5`
- `scaler.pkl`
- `shap_mean_values.npy` (permutation importance scores)
- `shap_feature_names.npy`

**Typical performance:** MAE ≈ 0.04–0.08, R² ≈ 0.90–0.95 on the test split.

---

### Step 3 — Physics Simulation (`step3_physics_simulation.py`)

**What it does:**
- Implements the full bearing health → process → economics pipeline
- Runs a CLI demo on 4 representative bearings
- Generates `health_sweep.png` — a 4-panel visualisation showing how degradation propagates

**Key functions:**

| Function | Description |
|----------|-------------|
| `bearing_to_extraction(health)` | Nonlinear map: bearing health → juice extraction efficiency |
| `juice_brix(extraction_eff)` | Extraction efficiency → mixed-juice Brix (°Bx) |
| `evaporator_odes(state, t, ...)` | 5-effect falling-film evaporator ODE system |
| `run_evaporator(health)` | Full evaporator simulation, returns recovery rate + crystallisation risk |
| `compute_economic_impact(health, rul_hours)` | Complete pipeline → ₹ loss dict + action recommendation |

**Output:** `sugarsense_simulation/simulation_results.json` and `health_sweep.png`

---

### Step 4 — Gradio Dashboard (`step4_dashboard.py`)

**What it does:**
- Launches a real-time interactive dashboard with a public Gradio URL (no token needed)
- Monitors 4 bearings simultaneously: Roller Mill #1 & #2, Centrifugal #1, Conveyor #3
- Updates all charts live as sliders are adjusted

**Dashboard panels:**

| Panel | Description |
|-------|-------------|
| Bearing Health Gauges | 4 colour-coded radial gauges (green/amber/red) |
| Sugar Recovery Rate | Bar chart vs. 10.5% critical threshold |
| Revenue at Risk | 48-hour delay cost per bearing in ₹ Lakh |
| OEE Degradation Timeline | 72-hour projection for the selected bearing |
| Full Analysis | Markdown table with all metrics + action recommendation |

**Configurable inputs:** Mill capacity (TCD), MSP (₹/quintal), health score and RUL for each bearing.

---

## Technical Details

### Model Architecture

```
SugarSense_CNN
─────────────────────────────────────
Layer               Output Shape    Params
─────────────────────────────────────
Input               (None, 24, 1)
Conv1D(64, k=3)     (None, 24, 64)   256
BatchNorm           (None, 24, 64)   256
Conv1D(64, k=3)     (None, 24, 64)  12,352
BatchNorm           (None, 24, 64)   256
MaxPool(2)          (None, 12, 64)
Dropout(0.2)
Conv1D(128, k=3)    (None, 12, 128) 24,704
BatchNorm           (None, 12, 128)  512
Conv1D(128, k=3)    (None, 12, 128) 49,280
BatchNorm           (None, 12, 128)  512
GlobalAvgPool       (None, 128)
Dropout(0.3)
Dense(64)           (None, 64)       8,256
Dropout(0.2)
Dense(32)           (None, 32)       2,080
Dense(1, sigmoid)   (None, 1)          33
─────────────────────────────────────
Total params: ~98,497
```

### Feature Engineering

24 features extracted from each 0.1-second vibration window (2560 samples @ 25.6kHz), computed independently for horizontal and vertical channels:

| # | Feature | Domain |
|---|---------|--------|
| 1 | RMS | Time |
| 2 | Peak | Time |
| 3 | Crest Factor | Time |
| 4 | Kurtosis | Time |
| 5 | Skewness | Time |
| 6 | Variance | Time |
| 7 | FFT Band Energy 0–1kHz | Frequency |
| 8 | FFT Band Energy 1–5kHz | Frequency |
| 9 | FFT Band Energy 5–12.8kHz | Frequency |
| 10 | Spectral Entropy | Frequency |
| 11 | Spectral Centroid | Frequency |
| 12 | High-Freq Ratio | Frequency |

### Evaporator Physics

The 5-effect falling-film evaporator is modelled as a coupled ODE system:

```
dBrix_i/dt = (target_i × feed_factor − Brix_i) / τ
```

Where τ = 0.5 hr (industry standard residence time), target Brix per effect = [25, 35, 45, 55, 65°Bx], and feed_factor scales with juice extraction efficiency.

### Economic Model

```
sugar_output   = TCD × recovery_rate / 100                 (tonnes)
daily_revenue  = sugar_output × MSP_per_quintal × 10       (₹)
daily_loss     = nominal_revenue − actual_revenue           (₹)

P(failure_48h) = sigmoid(−5 × (health − 0.35))
loss_48h       = daily_loss × 2 + P(failure) × 12hr × ₹70L/hr

FRP_risk_days  = (TCD × recovery_drop × ₹340/tonne) / ₹10L threshold
```

---

## Key Results

| Bearing | Health | RUL | Recovery Rate | Revenue at Risk (48h) | Status |
|---------|--------|-----|---------------|----------------------|--------|
| Roller Mill #1 | 82% | 97 hrs | 10.39% | ₹1.8L | ✅ OK |
| Roller Mill #2 | 34% | 31 hrs | 9.10% | ₹18.7L | 🚨 CRITICAL |
| Centrifugal #1 | 61% | 58 hrs | 10.21% | ₹4.2L | ⚠️ WARNING |
| Conveyor #3 | 91% | 142 hrs | 10.47% | ₹0.6L | ✅ OK |

**Maintenance ROI for Roller Mill #2:** Bearing replacement costs ~₹1.5L. Acting on the alert saves ₹18.7L → **ROI of 12×**.

---

## Data Sources & Citations

| Data | Source |
|------|--------|
| PRONOSTIA/FEMTO bearing dataset | Nectoux et al., IEEE PHM 2012 Data Challenge |
| Downtime cost (₹70L+/hr) | ABB Value of Reliability Survey, India 2023 |
| Sugar production (261 LMT) | ISMA, December 2025 |
| Operating mills (478) | ISMA, December 2025 |
| MSP ₹3,600/quintal | Government of India, 2024-25 |
| FRP ₹340/tonne cane | Government of India, 2024-25 |
| Evaporator physics | Sugar Engineering literature (Hugot, Rein) |

---

## Tech Stack

All open-source. Runs entirely on free Google Colab T4 GPU.

- **TensorFlow / Keras** — 1D CNN training and inference
- **SciPy** — Signal processing + ODE evaporator simulation
- **NumPy / Pandas** — Feature engineering and data handling
- **Gradio** — Interactive real-time dashboard
- **Plotly** — Dashboard visualisations
- **scikit-learn** — Scaling, metrics, train/test split
- **Matplotlib** — Training plots and analysis charts

---

## Team

**Team Muffin**

| Name | GitHub |
|------|--------|
| Sneha Chakraborty | [@Kweenbee187](https://github.com/Kweenbee187) |
| Divyansh Pathak | [@tituatgithub](https://github.com/tituatgithub) |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*SugarSense aligns with ISMA's mandate: AI automation · Energy efficiency · Sustainable manufacturing · Bio-energy protection · Farmer welfare*
