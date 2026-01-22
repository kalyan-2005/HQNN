import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import io

DATA_DIR = 'data/Training'
FEATURE_DIM = 8
BATCH_SIZE = 32
OUTPUT_FILE_TENSOR = 'features_train.pt'
OUTPUT_FILE_LOG = 'feature_extraction_output_TRAIN.txt'

class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def extract_all_features(log_file):
    with io.StringIO() as log_capture:
        def log(*args):
            print(*args, file=log_capture)

        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.45], [0.21])
        ])

        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ClassicalFeatureExtractor(FEATURE_DIM)
        model.eval()

        features = []
        with torch.no_grad():
            for imgs, _ in loader:
                features.append(model(imgs))

        features = torch.cat(features)
        torch.save(features, OUTPUT_FILE_TENSOR)

        log(f"Feature shape: {features.shape}")
        with open(log_file, 'w') as f:
            f.write(log_capture.getvalue())

if __name__ == "__main__":
    extract_all_features(OUTPUT_FILE_LOG)
