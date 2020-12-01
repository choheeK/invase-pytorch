from data import get_data
from experiments import INVASETrainer
from config import get_decode_args
import torch
from models.decoder import LinearDecoder
import os
from tqdm import tqdm
from robustness.tools.helpers import AverageMeter
from argparse import Namespace
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import numpy as np


N_IMAGES = 2


def load_ckpt(trainer, ckpt_path):
    model = trainer.model
    optimizer = trainer.optimizer
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    epoch = state_dict["epoch"]
    trainer.epoch = epoch
    return trainer


def main(args):
    path = args.trained_path
    ckpt_path = os.path.join(path, "checkpoint")
    config_path = os.path.join(path, "config.json")
    decode_result_path = os.path.join(path, "decode_results.json")

    # Reload the experiment configurations
    with open(config_path, "r") as fp:
        trainer_args_dict = json.load(fp)
    trainer_args = Namespace(**trainer_args_dict)

    # Get the data
    dim, label_dim, train_loader, test_loader = get_data(trainer_args)
    dim = train_loader.dataset.input_size
    label_dim = train_loader.dataset.output_size

    # Load from the checkpoint
    trainer = INVASETrainer(dim, label_dim, trainer_args, path)
    trainer = load_ckpt(trainer, ckpt_path)

    # Construct the decoder
    decoder = LinearDecoder(dim)
    optimizer = optim.Adam(decoder.parameters(), 0.1, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Obtain these parameters to undo normalization
    mean = torch.tensor(train_loader.dataset.means)
    std = torch.tensor(train_loader.dataset.stds)

    # Tuning the decoder
    for i in range(args.decoder_epochs):
        MSE = AverageMeter()
        b_loader = tqdm(train_loader)
        trainer.model.eval()
        for x_batch, y_batch, _ in b_loader:
            b_loader.set_description(
                f"EpochProvision: DecodingMSE: {MSE.avg}")

            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            # Generate a batch of selections
            selection_probability = trainer.model(x_batch, fw_module="selector")
            # Predictor objective
            used, reconstruction = decoder(selection_probability, x_batch)

            # Convert to pixels space
            reconstruction = reconstruction * std + mean
            x_batch = x_batch * std + mean
            loss = loss_fn(reconstruction, x_batch)
            MSE.update(loss.detach().item(), y_batch.shape[0])
            loss.backward()
            optimizer.step()

        if (i+1) % args.eval_freq == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/2

    fig, axs = plt.subplots(N_IMAGES, 3, figsize=(10, 5))
    flat_shape = x_batch.shape[1]
    img_dim = int(np.sqrt(flat_shape))
    for i in range(N_IMAGES):
        im = x_batch[i].detach().numpy().reshape((img_dim, img_dim))
        im_rec = reconstruction[i].detach().numpy().reshape((img_dim, img_dim))
        im_chosen = used[i].detach().numpy().reshape((img_dim, img_dim))
        axs[i][0].imshow(im)
        axs[i][1].imshow(im_rec)
        axs[i][2].imshow(im_chosen)
        axs[i][0].set_axis_off()
        axs[i][1].set_axis_off()
        axs[i][2].set_axis_off()

    axs[0][0].set_title("Original Image", fontsize=18)
    axs[0][1].set_title("Reconstructed Image", fontsize=18)
    axs[0][2].set_title("Chosen Pixels", fontsize=18)

    fig.savefig(os.path.join(path, "reconstruction_viz.pdf"))
    fig.savefig(os.path.join(path, "reconstruction_viz.png"))
    plt.close(fig)

    MSE = AverageMeter()
    modes = [("Train", train_loader), ("Test", test_loader)]
    decoder.eval()
    trainer.model.eval()
    result = dict()
    for mode, loader in modes:
        b_loader = tqdm(loader)
        for x_batch, y_batch, _ in b_loader:
            b_loader.set_description(
                f"EpochProvision: DecodingMSE: {MSE.avg}")
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            selection_probability = trainer.model(x_batch, fw_module="selector")
            used, reconstruction = decoder(selection_probability, x_batch)
            reconstruction = reconstruction * std + mean
            x_batch = x_batch * std + mean
            loss = loss_fn(reconstruction, x_batch)
            MSE.update(loss.detach().item(), y_batch.shape[0])

        print(f"{mode} Final: ", MSE.avg)
        result[mode] = MSE.avg

    with open(decode_result_path, "w") as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    args = get_decode_args()
    main(args)
