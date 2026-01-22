# quantum_model.py
import torch.nn as nn

from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumNeuralNetwork(nn.Module):
    """
    Trainable Quantum Neural Network using
    ZZFeatureMap + RealAmplitudes Ansatz
    """

    def __init__(self, feature_map, n_qubits):
        super().__init__()

        ansatz = RealAmplitudes(
            num_qubits=n_qubits,
            reps=1,
            entanglement='full'
        )

        qnn = EstimatorQNN(
            circuit=feature_map.compose(ansatz),
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        self.qnn_layer = TorchConnector(qnn)

    def forward(self, x):
        return self.qnn_layer(x)
