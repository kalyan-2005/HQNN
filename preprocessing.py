import torch
from torchvision import datasets, transforms
import os
import io

DATA_DIR = 'data/Training'
IMAGE_SIZE = (128, 128)
OUTPUT_FILE_LABELS = 'labels_train.pt'
OUTPUT_FILE_LOG = 'preprocessing_output_TRAIN.txt'

def preprocess_and_load_data(log_file):
    with io.StringIO() as log_capture:
        def log(*args):
            print(*args, file=log_capture)

        log("Running preprocessing...")

        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45], std=[0.21])
        ])

        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        labels = torch.tensor(dataset.targets)
        torch.save(labels, OUTPUT_FILE_LABELS)

        log(f"Total images: {len(dataset)}")
        log("Labels saved successfully.")

        with open(log_file, 'w') as f:
            f.write(log_capture.getvalue())

if __name__ == "__main__":
    preprocess_and_load_data(OUTPUT_FILE_LOG)
