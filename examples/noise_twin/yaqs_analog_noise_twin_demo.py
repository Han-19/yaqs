"""YAQS analog noise twin demo.

This follows the project pipeline:

initial state
-> Hamiltonian model
-> hardware-like noise profile
-> run YAQS simulation
-> save observables for later ML
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import X


def build_hardware_like_noise_profile(length: int) -> list[dict[str, object]]:
    """Create manually chosen qubit-specific noise values.

    For now, these values are hand-written.
    Later, this function can load real hardware calibration data.
    """

    # Example hardware-like values.
    # Each qubit has slightly different relaxation and dephasing strength.
    gamma_lowering = [0.08, 0.10, 0.12, 0.09, 0.11]
    gamma_dephasing = [0.04, 0.05, 0.06, 0.045, 0.055]

    noise_processes: list[dict[str, object]] = []

    for site in range(length):
        noise_processes.append(
            {
                "name": "lowering",
                "sites": [site],
                "strength": gamma_lowering[site],
            }
        )

        noise_processes.append(
            {
                "name": "pauli_z",
                "sites": [site],
                "strength": gamma_dephasing[site],
            }
        )

    return noise_processes


def run_one_yaqs_experiment() -> None:
    # -----------------------------
    # 1. Initial state
    # -----------------------------
    length = 5
    state = State(length, initial="zeros")

    # -----------------------------
    # 2. Hamiltonian of model
    # -----------------------------
    J = 1.0
    g = 0.5
    hamiltonian = Hamiltonian.ising(length, J, g)

    # -----------------------------
    # 3. Hardware-like noise
    # -----------------------------
    noise_processes = build_hardware_like_noise_profile(length)
    noise_model = NoiseModel(noise_processes)

    # -----------------------------
    # 4. YAQS simulation parameters
    # -----------------------------
    sim_params = AnalogSimParams(
        observables=[Observable(X(), site) for site in range(length)],
        elapsed_time=5.0,
        dt=0.1,
        num_traj=100,
        max_bond_dim=8,
        svd_threshold=1e-6,
        order=2,
        sample_timesteps=True,
        random_seed=42,
    )

    # -----------------------------
    # 5. Run YAQS simulation
    # -----------------------------
    simulator = Simulator(parallel=True, show_progress=True)

    result = simulator.run(
        copy.deepcopy(state),
        hamiltonian,
        sim_params,
        copy.deepcopy(noise_model),
    )

    # -----------------------------
    # 6. Save dataset
    # -----------------------------
    output_dir = Path("examples/noise_twin/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    expectation_values = np.asarray(result.expectation_values)

    np.savez(
        output_dir / "yaqs_noise_twin_demo.npz",
        expectation_values=expectation_values,
        J=J,
        g=g,
        length=length,
        elapsed_time=sim_params.elapsed_time,
        dt=sim_params.dt,
        num_traj=sim_params.num_traj,
    )

    print("Finished YAQS analog noise twin demo.")
    print(f"Expectation values shape: {expectation_values.shape}")
    print(f"Saved to: {output_dir / 'yaqs_noise_twin_demo.npz'}")


if __name__ == "__main__":
    run_one_yaqs_experiment()
