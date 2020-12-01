from .synthetic import SyntheticDataset, make_synthetic_loaders
from .mnist import get_mnist
from .coil20 import get_coil20
from .data_generation import generate_dataset


__datagetters__ = {"mnist": get_mnist,
                   "coil20": get_coil20}

def get_data(args):
    if args.data_type in __datagetters__:
        train_loader, test_loader = __datagetters__[args.data_type](args)
        dim = train_loader.dataset.input_size
        label_dim = train_loader.dataset.output_size
        return dim, label_dim, train_loader, test_loader
    else:
        x_train, y_train, g_train = generate_dataset(n=args.train_no,
                                                     dim=args.dim,
                                                     data_type=args.data_type,
                                                     seed=0)
        x_test, y_test, g_test = generate_dataset(n=args.test_no,
                                                  dim=args.dim,
                                                  data_type=args.data_type,
                                                  seed=1)
        ds_train = SyntheticDataset(x_train, y_train, g_train)
        ds_test = SyntheticDataset(x_test, y_test, g_test)

        train_loader, test_loader = make_synthetic_loaders(ds_train, ds_test, args)
        dim = x_train.shape[1]
        label_dim = y_train.shape[1]
        return dim, label_dim, train_loader, test_loader

