import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qiskit.circuit.library import zz_feature_map
from quantum_model import QuantumNeuralNetwork
from hybrid_classifier import HybridClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- CONFIG ----------------
TEST_DIR = "data/Testing"   # change if your test folder name is different
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
NUM_CLASSES = 4
FEATURE_DIM = 8
N_QUBITS = 8

QNN_MODEL_PATH = "qnn_model.pth"
CLASSIFIER_MODEL_PATH = "classifier_model.pth"

DEVICE = "cpu"   # Qiskit QNN runs on CPU generally
# ----------------------------------------


# Same CNN Feature Extractor used in training
class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45], std=[0.21])
    ])

    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return dataset, loader


def extract_features(model, loader):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            feats = model(imgs)
            all_features.append(feats)
            all_labels.append(labels)

    all_features = torch.cat(all_features).float()
    all_labels = torch.cat(all_labels).long()
    return all_features, all_labels


def main():
    print("✅ Loading test dataset...")
    test_dataset, test_loader = get_test_loader()
    print(f"Total test images: {len(test_dataset)}")
    print("Class mapping:", test_dataset.class_to_idx)

    # 1) Load CNN feature extractor (same architecture)
    print("\n✅ Extracting classical features...")
    cnn_extractor = ClassicalFeatureExtractor(FEATURE_DIM).to(DEVICE)

    # NOTE: You did not save CNN weights during training.
    # So this CNN is random unless you trained it.
    # If you want correct evaluation, you MUST train and save CNN too.
    # For now, it matches your pipeline behavior.

    X_test, y_test = extract_features(cnn_extractor, test_loader)
    print("Extracted feature shape:", X_test.shape)

    # 2) Load Quantum Model + Classifier
    print("\n✅ Loading trained QNN + classifier...")

    feature_map = zz_feature_map(
        feature_dimension=N_QUBITS,
        reps=1,
        entanglement="full"
    )

    qnn = QuantumNeuralNetwork(feature_map, N_QUBITS).to(DEVICE)
    classifier = HybridClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    qnn.load_state_dict(torch.load(QNN_MODEL_PATH, map_location=DEVICE))
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))

    qnn.eval()
    classifier.eval()

    # 3) Predict
    print("\n✅ Evaluating model...")
    y_pred = []

    with torch.no_grad():
        # batch evaluation for speed
        for i in range(0, len(X_test), BATCH_SIZE):
            x_batch = X_test[i:i+BATCH_SIZE]
            q_out = qnn(x_batch)
            logits = classifier(q_out)
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().numpy())

    y_pred = torch.tensor(y_pred).numpy()
    y_true = y_test.cpu().numpy()

    # 4) Metrics
    acc = accuracy_score(y_true, y_pred)
    print("\n================ RESULTS ================")
    print(f"✅ Accuracy: {acc * 100:.2f}%")

    print("\n📌 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("=========================================")


if __name__ == "__main__":
    main()
