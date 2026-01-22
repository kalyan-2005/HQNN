import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import io
from qiskit.circuit.library import zz_feature_map

INPUT_FILE_TENSOR = 'features_train.pt'
OUTPUT_FILE_LOG = 'quantum_encoding_output_TRAIN.txt'
N_QUBITS = 8

def run_quantum_encoding(log_file):
    with io.StringIO() as log_capture:
        def log(*args):
            print(*args, file=log_capture)

        features = torch.load(INPUT_FILE_TENSOR)
        total = features.shape[0]
        log(f"Total feature vectors: {total}")

        feature_map = zz_feature_map(
            feature_dimension=N_QUBITS,
            reps=1,
            entanglement='full'
        )

        for i in range(total):
            encoded = feature_map.assign_parameters(features[i].numpy())
            if (i+1) % 500 == 0:
                log(f"Encoded {i+1}/{total}")

        with open(log_file, 'w') as f:
            f.write(log_capture.getvalue())

if __name__ == "__main__":
    run_quantum_encoding(OUTPUT_FILE_LOG)
