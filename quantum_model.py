import torch
import torch.nn as nn

from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumNeuralNetwork(nn.Module):
    def __init__(self, feature_map, num_qubits):
        super().__init__()

        # Variational ansatz
        ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            reps=1,
            entanglement="full"
        )

        circuit = feature_map.compose(ansatz)

        # 🔑 MULTI-OBSERVABLE MEASUREMENT (KEY FIX)
        observables = []

        for i in range(num_qubits):
            pauli_str = ["I"] * num_qubits
            pauli_str[i] = "Z"
            observables.append(
                SparsePauliOp("".join(pauli_str))
            )

        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        self.qnn = TorchConnector(qnn)

    def forward(self, x):
        return self.qnn(x)
