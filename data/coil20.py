from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import cv2
COIL_SEED = 0


class COIL20(Dataset):
    def __init__(self, train=True, base_path="./data-dir/coil20"):
        self.x = None
        self.y = None
        self.g = None
        self.means = None
        self.stds = None
        self.load_dataset(base_path, train)

    def load_dataset(self, base_path, train):
        all_images = os.listdir(base_path)
        y_list = []
        img_list = []
        for img_path in all_images:
            class_idx = int(img_path[3:6].split("_")[0])-1
            y_list.append(class_idx)
            img_path = os.path.join(base_path, img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)/255
            img = cv2.resize(img, (20, 20))
            img = np.reshape(img, (1, 20*20))
            img_list.append(img)

        img_arr = np.concatenate(img_list, axis=0)
        y_arr = np.array(y_list)
        np.random.seed(0)
        N = y_arr.shape[0]
        train_pick = np.random.choice(N, int(0.8 * N), replace=False)
        train_idx = np.zeros((N), dtype=bool)
        train_idx[train_pick] = True
        if train:
            idx = train_idx
        else:
            idx = ~train_idx
        self.x = torch.tensor(img_arr[idx]).float()
        self.y = torch.tensor(y_arr[idx])
        self.means = torch.tensor(np.mean(img_arr[train_idx]))
        self.stds = torch.tensor(np.std(img_arr[train_idx]))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        X_batch = (self.x[idx]-self.means)/self.stds
        y_batch = self.y[idx]
        return X_batch, y_batch, -1


def one_hot(arr):
    temp = torch.zeros((arr.shape[0], arr.max() + 1))
    temp[torch.arange(arr.shape[0]), arr] = 1
    return temp


def get_coil20(args):
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.batch_size if args.batch_size else 512
    num_workers = args.workers if args.workers else 4

    train_data = COIL20(train=True)

    train_data.bounds = [0, 1]
    train_data.input_size = train_data.x.shape[1]
    train_data.y = one_hot(train_data.y)
    train_data.output_size = train_data.y.shape[1]

    test_data = COIL20(train=False)
    test_data.y = one_hot(test_data.y)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
