# YAQS Noise Twin Demo

This directory contains a first prototype of the **quantum noise twin pipeline** using YAQS.

The goal is to follow the project pipeline from the sketch:

```text
Analog simulation

initial state
    vector / matrix / MPS
        ↓
Hamiltonian of model
    Ising / Heisenberg / transverse-field model
        ↓
artificial or hardware-like noise
    Pauli noise, crosstalk, σx, σy, σz
    noise strength: 10^-3 to 10^-1
    Gaussian / Bell-curve / hardware-calibrated values
        ↓
run YAQS simulations
        ↓
train ML model
    linear regression, random forest, small neural network
        ↓
test whether ML infers noise parameters
        ↓
use learned noise model to predict another experiment
```

In this first version, the pipeline uses YAQS to simulate an analog Ising model with qubit-specific noise values. Then it generates a small dataset and trains simple machine-learning models on the simulation outputs.

---

## Files in this directory

Expected structure:

```text
examples/noise_twin/
├── yaqs_analog_noise_twin_demo.py
├── generate_noise_twin_dataset.py
├── train_noise_twin_model.py
├── README.md
└── output/
    ├── noise_twin_dataset.csv
    └── noise_twin_timeseries.npz
```

The `output/` directory is generated automatically and should usually not be committed to Git.

---

## 1. Goal of the demo

The goal is to build the first working version of a **hardware-like quantum noise twin**.

The idea is:

1. Choose an initial quantum state.
2. Choose an analog Hamiltonian model, currently Ising.
3. Add artificial or hardware-like noise.
4. Run YAQS simulations.
5. Save observables as a dataset.
6. Train ML models to learn how noise changes the output.
7. Later use the learned model to predict another experiment.

This is not yet a full production noise twin. It is a small working prototype that proves the full pipeline is possible.

---

## 2. How to run

Run all commands from the YAQS repository root:

```bash
cd ~/QEL_ws/yaqs
```

### Install dependencies with uv

If you use the YAQS uv environment:

```bash
uv sync
```

If the ML script complains about missing packages, add them:

```bash
uv add pandas scikit-learn
```

If you are using your active `qel_env` instead of the repo `.venv`, use:

```bash
uv pip install pandas scikit-learn --active
uv run --active python examples/noise_twin/train_noise_twin_model.py
```

Normally, the recommended way is to use the project environment:

```bash
uv run python examples/noise_twin/generate_noise_twin_dataset.py
uv run python examples/noise_twin/train_noise_twin_model.py
```

---

## 3. Run the single YAQS demo

The single demo runs one analog simulation.

```bash
uv run python examples/noise_twin/yaqs_analog_noise_twin_demo.py
```

This script does:

```text
initial state
    State(length, initial="zeros")
        ↓
Hamiltonian
    Hamiltonian.ising(length, J, g)
        ↓
hardware-like noise
    qubit-specific lowering + pauli_z noise
        ↓
YAQS simulation
    Simulator.run(...)
        ↓
save result
    examples/noise_twin/output/yaqs_noise_twin_demo.npz
```

Use this script only to confirm that YAQS simulation works.

---

## 4. Generate the dataset

After the single demo works, generate a small dataset:

```bash
uv run python examples/noise_twin/generate_noise_twin_dataset.py
```

Expected output:

```text
Running experiment 1/20
Running experiment 2/20
...
Saved dataset:
CSV: examples/noise_twin/output/noise_twin_dataset.csv
NPZ: examples/noise_twin/output/noise_twin_timeseries.npz
```

The generated CSV contains one row per experiment. Each experiment uses a different sampled noise profile.

Example columns:

```text
experiment_id
length
J
g
elapsed_time
dt
num_traj
max_bond_dim
mean_gamma_lowering
std_gamma_lowering
mean_gamma_dephasing
std_gamma_dephasing
gamma_lowering_q0
gamma_dephasing_q0
...
final_mean_x
```

The important idea is that the input is the Hamiltonian and noise profile, and the output is the final observable.

---

## 5. Train the ML model

After generating the dataset, train the first ML models:

```bash
uv run python examples/noise_twin/train_noise_twin_model.py
```

This script trains two types of models.

### Forward model

The forward model learns:

```text
noise parameters + Hamiltonian parameters
        ↓
final noisy observable
```

In code, this means:

```text
input X:
    J, g, dt, elapsed_time, noise strengths, qubit-level gamma values

target y:
    final_mean_x
```

This answers:

> Can we predict the noisy output from the hardware-like noise profile?

### Inverse model

The inverse model learns:

```text
observed output
        ↓
noise parameter estimate
```

In code, this first version tries to infer:

```text
mean_gamma_dephasing
```

from:

```text
final_mean_x, J, g, dt, elapsed_time
```

This answers:

> Can ML infer the noise strength from the observed simulation output?

The inverse problem is harder. With only one final observable, the result may not be very accurate yet. This is expected.

---

## 6. How the code works

### `yaqs_analog_noise_twin_demo.py`

This is the simplest script. It runs one YAQS simulation.

Main blocks:

```python
state = State(length, initial="zeros")
hamiltonian = Hamiltonian.ising(length, J, g)
noise_model = NoiseModel(noise_processes)
sim_params = AnalogSimParams(...)
result = simulator.run(state, hamiltonian, sim_params, noise_model)
```

This directly matches the project pipeline:

```text
initial state → Hamiltonian → noise → YAQS simulation → output
```

---

### `generate_noise_twin_dataset.py`

This script runs many simulations.

The most important loop is:

```python
for experiment_id in range(n_experiments):
    row, expectation_values = run_single_experiment(...)
    rows.append(row)
    all_timeseries.append(expectation_values)
```

Each experiment does:

1. Create initial state.
2. Create Ising Hamiltonian.
3. Sample a new hardware-like noise profile.
4. Run YAQS.
5. Extract the final observable.
6. Save one dataset row.

The noise values are sampled in the range:

```text
10^-3 to 10^-1
```

and each qubit receives a slightly different value using a Gaussian-like distribution.

This matches the screenshot block:

```text
artificial noise
    Pauli, crosstalk, 10^-3 ~ 10^-1
    Bell curve / Gaussian
```

Currently implemented noise channels:

```text
lowering noise
pauli_z noise
```

Later we can add:

```text
pauli_x
pauli_y
crosstalk
readout error
hardware calibration files
```

---

### `train_noise_twin_model.py`

This script loads the CSV dataset and trains ML models.

It uses:

```python
pandas
scikit-learn
LinearRegression
Ridge
RandomForestRegressor
```

It prints metrics such as:

```text
MAE
R2
```

Meaning:

- `MAE`: average prediction error.
- `R2`: how much of the output variation the model explains.

For a very small dataset, the metrics can be unstable. That is okay. The purpose is to prove that the pipeline works end-to-end.

---

## 7. Important parameters to change

In `generate_noise_twin_dataset.py`, start with small values:

```python
n_experiments = 20
length = 5
elapsed_time = 2.0
dt = 0.2
num_traj = 30
max_bond_dim = 4
```

After the code works, increase gradually:

```python
n_experiments = 100
num_traj = 100
elapsed_time = 5.0
max_bond_dim = 8
```

Do not increase everything at once. YAQS trajectory simulations can become expensive.

---

## 8. Current limitations

This is a first demo, so it has limitations:

1. The noise is hardware-like but not yet from real hardware calibration data.
2. Only a small Ising model is used.
3. Only simple observables are used.
4. The inverse ML model uses only a weak feature set.
5. The dataset is small.
6. Crosstalk is not implemented yet.

---

## 9. Next steps

### Step 1: Add better observables

Currently the ML model mostly uses `final_mean_x`.

Next add:

```text
final value per site
mean over time
standard deviation over time
maximum value
minimum value
slope / decay rate
correlation functions
```

This will make inverse noise inference easier.

---

### Step 2: Add more noise types

Current noise:

```text
lowering
pauli_z
```

Next noise channels:

```text
pauli_x
pauli_y
crosstalk
coupling disorder
readout error
```

---

### Step 3: Add hardware-profile input files

Instead of hardcoding noise sampling inside Python, create a file like:

```text
examples/noise_twin/hardware_profiles/example_profile.yaml
```

Example:

```yaml
device: mock_analog_device
qubits:
  0:
    gamma_lowering: 0.001
    gamma_dephasing: 0.005
  1:
    gamma_lowering: 0.0015
    gamma_dephasing: 0.004
```

Then the pipeline becomes more hardware-like:

```text
load hardware profile → build YAQS NoiseModel → run simulation
```

---

### Step 4: Compare ML models

Start with:

```text
linear regression
ridge regression
random forest
```

Later add:

```text
small neural network
Gaussian process regression
uncertainty estimates
```

---

### Step 5: Predict another experiment

After the ML model works, add a script:

```text
predict_new_experiment.py
```

It should:

1. Load a trained model.
2. Define a new Hamiltonian/noise profile.
3. Predict the observable.
4. Compare with a fresh YAQS simulation.

This is the final screenshot step:

```text
use learned noise model to predict another experiment
```

---

## 10. One-sentence summary

This demo builds the first YAQS-based quantum noise twin pipeline: it runs analog Ising simulations with qubit-specific hardware-like noise, generates a dataset of noisy observables, and trains simple ML models to predict or infer noise effects.
