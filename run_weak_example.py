import numpy as np
from qiskit.circuit.library.n_local import TwoLocal

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import WeakSimParams
from mqt.yaqs.core.data_structures.state import State

num_qubits = 5

circuit = TwoLocal(
    num_qubits,
    ["rx"],
    ["rzz"],
    entanglement="linear",
    reps=num_qubits,
).decompose()

num_pars = len(circuit.parameters)
rng = np.random.default_rng()
values = rng.uniform(-np.pi, np.pi, size=num_pars)
circuit.assign_parameters(values, inplace=True)
circuit.measure_all()

state = State(num_qubits, initial="zeros")

gamma = 0.1
noise_model = NoiseModel([
    {"name": name, "sites": [i], "strength": gamma}
    for i in range(num_qubits)
    for name in ["lowering", "pauli_z"]
])

sim_params = WeakSimParams(
    shots=1024,
    max_bond_dim=4,
    svd_threshold=1e-6,
)

sim = Simulator()
result = sim.run(state, circuit, sim_params, noise_model)

print(result.counts)
