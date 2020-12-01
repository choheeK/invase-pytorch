from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class SyntheticDataset(Dataset):
    def __init__(self, X, y, g):
        self.x = torch.tensor(X).float()
        self.y = torch.tensor(y)
        self.g = torch.tensor(g)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        X_batch = self.x[idx]
        y_batch = self.y[idx]
        g_batch = self.g[idx]
        return X_batch, y_batch, g_batch


def make_synthetic_loaders(train_dataset, test_dataset, args):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, test_loader
