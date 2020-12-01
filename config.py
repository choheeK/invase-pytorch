import argparse
# Adapted from original INVASE repository


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        choices=['syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6', 'mnist', 'coil20'],
        default='syn1',
        type=str)
    parser.add_argument(
        '--train_no',
        help='the number of training data',
        default=10000,
        type=int)
    parser.add_argument(
        '--test_no',
        help='the number of testing data',
        default=10000,
        type=int)
    parser.add_argument(
        '--dim',
        help='the number of features',
        choices=[11, 100],
        default=11,
        type=int)
    parser.add_argument(
        '--lamda',
        help='invase hyper-parameter lambda',
        default=0.1,
        type=float)
    parser.add_argument(
        '--actor_h_dim',
        help='hidden state dimensions for actor',
        default=100,
        type=int)
    parser.add_argument(
        '--critic_h_dim',
        help='hidden state dimensions for critic',
        default=200,
        type=int)
    parser.add_argument(
        '--n_layer',
        help='the number of layers',
        default=3,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini batch',
        default=1000,
        type=int)
    parser.add_argument(
        '--activation',
        help='activation function of the networks',
        choices=['selu', 'relu'],
        default='relu',
        type=str)
    parser.add_argument(
        '--learning_rate',
        help='learning rate of model training',
        default=0.0001,
        type=float)
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--max-epochs',
        default=1000,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--eval-freq',
        default=10,
        type=int,
        help="frequency of evaluations"
    )
    parser.add_argument(
        '--device',
        default="cpu",
        type=str,
        help="device to keep the sensors in",
        choices=["cpu", "cuda"]
    )

    return parser.parse_args()


def get_decode_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained-path',
        required=True,
        type=str,
        help="path to the pretrained model",
    )
    parser.add_argument(
        '--decoder-epochs',
        default=10,
        type=int,
        help="number of epochs to tune the decoder",
    )
    parser.add_argument(
        '--device',
        default="cpu",
        type=str,
        help="device to keep the sensors in",
        choices=["cpu", "cuda"]
    )
    parser.add_argument(
        '--eval-freq',
        default=5,
        type=int,
        help="frequency of evaluations"
    )
    return parser.parse_args()

