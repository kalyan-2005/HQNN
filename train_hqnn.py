import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from qiskit.circuit.library import zz_feature_map
from quantum_model import QuantumNeuralNetwork
from hybrid_classifier import HybridClassifier

# Load data
features = torch.load("features_train.pt").float()
labels = torch.load("labels_train.pt").long()

# Dataset & DataLoader (VERY IMPORTANT)
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Quantum feature map
feature_map = zz_feature_map(
    feature_dimension=8,
    reps=1,
    entanglement='full'
)

# Models
qnn = QuantumNeuralNetwork(feature_map, 8)
classifier = HybridClassifier(num_classes=4)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(qnn.parameters()) + list(classifier.parameters()),
    lr=0.001
)

# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    total_loss = 0.0
    print(f"Epoch {epoch+1} started")

    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        optimizer.zero_grad()

        q_out = qnn(x_batch)        # NOW only 8 circuits
        outputs = classifier(q_out)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"  Batch {batch_idx+1}/{len(loader)} completed")

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

print("Training completed successfully.")

# Save trained models
torch.save(qnn.state_dict(), "qnn_model.pth")
torch.save(classifier.state_dict(), "classifier_model.pth")

print("Models saved successfully.")
