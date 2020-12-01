from data import get_data
from experiments.invase_experiment import INVASETrainer
from config import get_args


def main(args):
    dim, label_dim, train_loader, test_loader = get_data(args)
    trainer = INVASETrainer(dim, label_dim, args)
    performance_dict = trainer.train_model(train_loader, test_loader)

    return performance_dict


if __name__ == '__main__':
    args = get_args()
    # Call main function
    performance = main(args)
