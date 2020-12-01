from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


class MNISTInvase(MNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTInvase, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.view(-1)
        # Below -1 is due to G being undefined
        return img, target, -1


def one_hot(arr):
    temp = torch.zeros((arr.shape[0], arr.max() + 1))
    temp[torch.arange(arr.shape[0]), arr] = 1
    return temp


def get_mnist(args):
    base_path = "./data-dir"
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.batch_size if args.batch_size else 512
    num_workers = args.workers if args.workers else 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_data = MNISTInvase(base_path, train=True, download=True,
                             transform=transform)
    train_data.means = (0.1307,)
    train_data.stds = (0.3081,)
    train_data.bounds = [0, 1]
    train_data.input_size = 784
    train_data.output_size = 10

    train_data.targets = one_hot(train_data.targets)

    test_data = MNISTInvase(base_path, train=False,
                            transform=transform)
    test_data.targets = one_hot(test_data.targets)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                                              shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

