# hybrid_classifier.py
import torch.nn as nn

class HybridClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        return self.fc(x)
