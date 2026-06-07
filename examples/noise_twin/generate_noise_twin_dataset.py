"""Generate a small hardware-like noise twin dataset with YAQS.

Pipeline:
initial state
-> Hamiltonian model
-> hardware-like / artificial noise
-> run YAQS simulations
-> save dataset for ML
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import X


OUTPUT_DIR = Path("examples/noise_twin/output")
DATASET_CSV = OUTPUT_DIR / "noise_twin_dataset.csv"
DATASET_NPZ = OUTPUT_DIR / "noise_twin_timeseries.npz"


def sample_positive_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    size: int,
    min_value: float = 0.0,
) -> np.ndarray:
    """Sample a Gaussian hardware-like noise vector and clip it to positive values."""
    values = rng.normal(loc=mean, scale=std, size=size)
    return np.clip(values, min_value, None)


def build_hardware_like_noise_profile(
    length: int,
    gamma_lowering: np.ndarray,
    gamma_dephasing: np.ndarray,
) -> NoiseModel:
    """Create a YAQS NoiseModel with qubit-specific noise strengths.

    Each qubit receives its own relaxation and dephasing value.
    This is the screenshot's artificial/hardware-like noise block.
    """
    noise_processes: list[dict[str, Any]] = []

    for site in range(length):
        noise_processes.append(
            {
                "name": "lowering",
                "sites": [site],
                "strength": float(gamma_lowering[site]),
            }
        )

        noise_processes.append(
            {
                "name": "pauli_z",
                "sites": [site],
                "strength": float(gamma_dephasing[site]),
            }
        )

    return NoiseModel(noise_processes)


def extract_final_mean_x(result: Any) -> float:
    """Extract one scalar target from YAQS expectation values.

    YAQS analog examples use result.expectation_values as a heatmap-like array/list.
    Here we average the final-time <X> over all sites.
    """
    values = np.asarray(result.expectation_values, dtype=float)

    # Expected common shape: time x site, but this function is defensive.
    if values.ndim == 1:
        return float(values[-1])

    if values.ndim == 2:
        return float(np.mean(values[-1, :]))

    # Fallback for higher-dimensional outputs.
    return float(np.mean(values.reshape(values.shape[0], -1)[-1, :]))


def run_single_experiment(
    experiment_id: int,
    rng: np.random.Generator,
    simulator: Simulator,
    length: int,
    elapsed_time: float,
    dt: float,
    num_traj: int,
    max_bond_dim: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Run one YAQS analog simulation with one sampled noise profile."""

    # -----------------------------
    # 1. Initial state
    # -----------------------------
    state = State(length, initial="zeros")

    # -----------------------------
    # 2. Hamiltonian of model
    # -----------------------------
    J = 1.0

    # Sweep the transverse field slightly.
    # This gives the ML model more than only noise features.
    g = float(rng.uniform(0.3, 1.2))

    hamiltonian = Hamiltonian.ising(length, J, g)

    # -----------------------------
    # 3. Artificial / hardware-like noise
    # -----------------------------
    # Choose experiment-level mean values.
    # These are in the screenshot range: around 10^-3 to 10^-1.
    mean_gamma_lowering = float(10 ** rng.uniform(-3, -1))
    mean_gamma_dephasing = float(10 ** rng.uniform(-3, -1))

    # Hardware is not uniform, so every qubit gets a slightly different value.
    std_gamma_lowering = 0.20 * mean_gamma_lowering
    std_gamma_dephasing = 0.20 * mean_gamma_dephasing

    gamma_lowering = sample_positive_normal(
        rng,
        mean=mean_gamma_lowering,
        std=std_gamma_lowering,
        size=length,
    )

    gamma_dephasing = sample_positive_normal(
        rng,
        mean=mean_gamma_dephasing,
        std=std_gamma_dephasing,
        size=length,
    )

    noise_model = build_hardware_like_noise_profile(
        length=length,
        gamma_lowering=gamma_lowering,
        gamma_dephasing=gamma_dephasing,
    )

    # -----------------------------
    # 4. Run YAQS simulations
    # -----------------------------
    sim_params = AnalogSimParams(
        observables=[Observable(X(), site) for site in range(length)],
        elapsed_time=elapsed_time,
        dt=dt,
        num_traj=num_traj,
        max_bond_dim=max_bond_dim,
        svd_threshold=1e-6,
        order=2,
        sample_timesteps=True,
        random_seed=10_000 + experiment_id,
    )

    result = simulator.run(
        copy.deepcopy(state),
        hamiltonian,
        sim_params,
        copy.deepcopy(noise_model),
    )

    expectation_values = np.asarray(result.expectation_values, dtype=float)
    final_mean_x = extract_final_mean_x(result)

    # This row is the ML-ready summary.
    row: dict[str, float | int] = {
        "experiment_id": experiment_id,
        "length": length,
        "J": J,
        "g": g,
        "elapsed_time": elapsed_time,
        "dt": dt,
        "num_traj": num_traj,
        "max_bond_dim": max_bond_dim,
        "mean_gamma_lowering": float(np.mean(gamma_lowering)),
        "std_gamma_lowering": float(np.std(gamma_lowering)),
        "mean_gamma_dephasing": float(np.mean(gamma_dephasing)),
        "std_gamma_dephasing": float(np.std(gamma_dephasing)),
        "final_mean_x": final_mean_x,
    }

    # Add qubit-level values as columns.
    for site in range(length):
        row[f"gamma_lowering_q{site}"] = float(gamma_lowering[site])
        row[f"gamma_dephasing_q{site}"] = float(gamma_dephasing[site])

    return row, expectation_values


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep this small first.
    # Increase later after confirming the pipeline works.
    n_experiments = 20
    length = 5
    elapsed_time = 2.0
    dt = 0.2
    num_traj = 30
    max_bond_dim = 4

    rng = np.random.default_rng(1234)

    # Use serial first for easier debugging.
    # Later change parallel=True.
    simulator = Simulator(parallel=False, show_progress=False)

    rows: list[dict[str, float | int]] = []
    all_timeseries: list[np.ndarray] = []

    for experiment_id in range(n_experiments):
        print(f"Running experiment {experiment_id + 1}/{n_experiments}")

        row, expectation_values = run_single_experiment(
            experiment_id=experiment_id,
            rng=rng,
            simulator=simulator,
            length=length,
            elapsed_time=elapsed_time,
            dt=dt,
            num_traj=num_traj,
            max_bond_dim=max_bond_dim,
        )

        rows.append(row)
        all_timeseries.append(expectation_values)

    # Save summary CSV for ML.
    fieldnames = list(rows[0].keys())

    with DATASET_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save full time-series too.
    np.savez(
        DATASET_NPZ,
        expectation_values=np.asarray(all_timeseries, dtype=float),
    )

    print("\nSaved dataset:")
    print(f"CSV: {DATASET_CSV}")
    print(f"NPZ: {DATASET_NPZ}")


if __name__ == "__main__":
    main()
