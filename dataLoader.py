import torch
import random
import numpy as np
from tqdm import tqdm

def loadData(polytopes, batch_size):
    train_data_raw, train_pBatches, train_batches, train_labels = [], [], [], []
    test_data_raw, test_pBatches, test_batches, test_labels = [], [], [], []

    print("Generating data:")
    for poly in tqdm(polytopes):
        poly.generateData()
        random.shuffle(poly.translations)

    print("Compiling data:")
    for poly in tqdm(polytopes):
        # Splitting random translations into training and testing (Can switch for 80-20 instead)
        for idx, T in enumerate(poly.translations):
            if idx <= len(poly.translations) - 3:
                train_data_raw += T
            else:
                test_data_raw += T

    print("Batching packaged data:")
    for poly in tqdm(polytopes):
        # Batching packaged data
        for _ in range(3):
            random.shuffle(train_data_raw)
            random.shuffle(test_data_raw)
        train_pBatches += [train_data_raw[k:k + batch_size] for k in range(0, len(train_data_raw), batch_size)]
        test_pBatches += [test_data_raw[k:k + batch_size] for k in range(0, len(test_data_raw), batch_size)]

    print("Splitting train data:")
    for b in tqdm(train_pBatches):
        batch, labels = [], []
        for packagedV in b:
            batch.append(packagedV[0])
            labels.append(packagedV[1])
        train_batches.append(batch)
        train_labels.append(labels)

    print("Splitting test data:")
    for b in tqdm(test_pBatches):
        batch, labels = [], []
        for packagedV in b:
            batch.append(packagedV[0])
            labels.append(packagedV[1])
        test_batches.append(batch)
        test_labels.append(labels)

    train_data = (torch.tensor(train_batches), torch.tensor(train_labels))
    test_data = (torch.tensor(test_batches), torch.tensor(test_labels))

    return train_data, test_data
