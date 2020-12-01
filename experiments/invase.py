import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import InvaseModel
from robustness.tools.helpers import accuracy, AverageMeter
from tqdm import tqdm
import numpy as np
import time
import os
import json
from .invase_legacy.utils import prediction_performance_metric, feature_performance_metric


class INVASETrainer:

    def __init__(self, dim, label_dim, args, load_path=None):
        self.dim = dim
        self.label_dim = label_dim
        self.model = InvaseModel(self.dim, self.label_dim, args)
        self.baseline_loss = F.cross_entropy
        self.critic_loss = nn.CrossEntropyLoss()
        self.lamda = args.lamda
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3, eps=1e-8)
        self.epoch = 0
        self.train_history = {"CriticAcc": [], "BaselineAcc": [], "ActorLoss": []}
        self.eval_history = {}
        self.args = args
        self.result_dir = load_path if load_path else f"./results/{args.data_type}/{time.strftime('%Y%m%d-%H%M%S')}"
        self.softmax = nn.Softmax(dim=1)

    def actor_loss(self, batch_data, actor_pred):
        actor_out = batch_data[:, :self.dim]
        critic_out = batch_data[:, self.dim:(self.dim + self.label_dim)]
        baseline_out = batch_data[:, (self.dim + self.label_dim):(self.dim + 2 * self.label_dim)]
        label = batch_data[:, (self.dim + 2 * self.label_dim):]

        critic_loss = -torch.sum(label * torch.log(critic_out + 1e-8), dim=1)
        baseline_loss = -torch.sum(label * torch.log(baseline_out + 1e-8), dim=1)
        Reward = -(critic_loss - baseline_loss)
        # Policy gradient loss computation.
        custom_actor_loss = \
            Reward * torch.sum(actor_out * torch.log(actor_pred + 1e-8) + (1 - actor_out) * torch.log(1 - actor_pred + 1e-8), dim=1)

        custom_actor_loss -= self.lamda * torch.mean(actor_pred, dim=1)
        # custom actor loss
        custom_actor_loss = torch.mean(-custom_actor_loss)

        return custom_actor_loss

    def train_step(self, train_loader):
        device = self.args.device
        self.model.train()

        CriticAcc = AverageMeter()
        BaselineAcc = AverageMeter()
        ActorLoss = AverageMeter()
        
        b_loader = tqdm(train_loader)
        for x_batch, y_batch, _ in b_loader:
            b_loader.set_description(f"EpochProvision: Critic: {CriticAcc.avg}, Baseline: {BaselineAcc.avg}, Actor: {ActorLoss.avg}")
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Select a random batch of samples
            self.optimizer.zero_grad()
            labels = torch.argmax(y_batch, dim=1).long()
            # Generate a batch of selections
            selection_probability = self.model(x_batch, fw_module="selector")
            selection = torch.bernoulli(selection_probability).detach()

            # Predictor objective
            critic_input = x_batch * selection
            critic_out = self.model(critic_input, fw_module="predictor")
            critic_loss = self.critic_loss(critic_out, labels)
            # Baseline objective
            baseline_out = self.model(x_batch, fw_module="baseline")
            baseline_loss = self.baseline_loss(baseline_out, labels)

            batch_data = torch.cat([selection.clone().detach(),
                                    self.softmax(critic_out).clone().detach(),
                                    self.softmax(baseline_out).clone().detach(),
                                    y_batch.float()], dim=1)

            # Actor objective
            actor_output = self.model(x_batch, fw_module="selector")
            actor_loss = self.actor_loss(batch_data, actor_output)

            total_loss = actor_loss + critic_loss + baseline_loss
            total_loss.backward()
            self.optimizer.step()

            N = labels.shape[0]
            critic_acc = accuracy(critic_out, labels)[0]
            baseline_acc = accuracy(baseline_out, labels)[0]
            CriticAcc.update(critic_acc.detach().item(), N)
            BaselineAcc.update(baseline_acc.detach().item(), N)
            ActorLoss.update(actor_loss.detach().item(), N)

        summary = {"CriticAcc": CriticAcc.avg,
                   "BaselineAcc": BaselineAcc.avg,
                   "ActorLoss": ActorLoss.avg}

        return summary

    def plot_results(self):
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        import matplotlib.pyplot as plt
        fig_path = os.path.join(self.result_dir, "acc") + ".png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        critic_hist = np.array(self.train_history["CriticAcc"])
        base_hist = np.array(self.train_history["BaselineAcc"])
        ax.plot(np.arange(critic_hist.shape[0]), critic_hist, label="Predictor Accuracy")
        ax.plot(np.arange(base_hist.shape[0]), base_hist, label="Baseline Accuracy")
        ax.legend()
        ax.set_title("Accuracies")
        ax.set_xlabel("Epoch")
        fig.savefig(fig_path)
        plt.close(fig)

        fig_path = os.path.join(self.result_dir, "actor-loss") + ".png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        actor_hist = np.array(self.train_history["ActorLoss"])
        ax.plot(np.arange(actor_hist.shape[0]), actor_hist, label="Actor Loss")
        ax.legend()
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        fig.savefig(fig_path)
        plt.close(fig)

        for k, v in self.eval_history.items():
            fig_path = os.path.join(self.result_dir, "eval-"+k) + ".png"
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            v = np.array(v)
            ax.plot(np.arange(v.shape[0]), v)
            ax.set_title(k)
            ax.set_xlabel("Epoch")
            fig.savefig(fig_path)
            plt.close(fig)

        config = os.path.join(self.result_dir, "config.json")
        with open(config, 'w') as fp:
            json.dump(vars(self.args), fp)

    def eval_metrics(self, loader, mode="Train", feature_metrics=True, pred_metrics=True):
        result_dict = {}
        if feature_metrics:
            names = [mode + "-TPR-Mean", mode + "-TPR-STD", mode + "-FDR-Mean", mode + "-FDR-STD"]
            for name in names:
                result_dict[name] = AverageMeter()

        if pred_metrics:
            names = [mode + "-AUC", mode + "-APR", mode + "-ACC"]
            for name in names:
                result_dict[name] = AverageMeter()
        g_hats, y_hats = [], []
        g_trues, y_trues = [], []
        with torch.no_grad():
            for x, y, g in loader:
                x = x.to(self.args.device)
                y_hat = self.model.predict(x).detach().numpy()
                g_hat = self.model.importance_score(x).detach().numpy()
                if pred_metrics:

                    auc, apr, acc = prediction_performance_metric(y, y_hat)
                    result_dict[mode + "-AUC"].update(auc, y.shape[0])
                    result_dict[mode + "-APR"].update(apr, y.shape[0])
                    result_dict[mode + "-ACC"].update(acc, y.shape[0])

                if feature_metrics:
                    importance_score = 1. * (g_hat > 0.5)
                    # Evaluate the performance of feature importance
                    mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(g.detach().numpy(), importance_score)
                    result_dict[mode + "-TPR-Mean"].update(mean_tpr, y.shape[0])
                    result_dict[mode + "-TPR-STD"].update(std_tpr, y.shape[0])
                    result_dict[mode + "-FDR-Mean"].update(mean_fdr, y.shape[0])
                    result_dict[mode + "-FDR-STD"].update(std_fdr, y.shape[0])
                g_hats.append(g_hat)
                y_hats.append(y_hat)
                g_trues.append(g.detach().numpy())
                y_trues.append(y.detach().numpy())

        for metric, val in result_dict.items():
            result_dict[metric] = val.avg

        g_hat = np.concatenate(g_hats, axis=0)
        y_hat = np.concatenate(y_hats, axis=0)
        g_true = np.concatenate(g_trues, axis=0)
        y_true = np.concatenate(y_trues, axis=0)
        return result_dict, g_hat, y_hat, g_true, y_true

    def eval_model(self, train_loader, test_loader, feature_metrics=False, save_arr=True):
        pred_dir = os.path.join(self.result_dir, "pred-dir")
        if not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
        self.model.eval()

        modes = [("Train", train_loader), ("Test", test_loader)]
        arrays = {}
        perf = {}
        for mode_type, loader in modes:
            perf_mode, g_pred, y_pred, g_true, y_true = self.eval_metrics(loader, mode_type, feature_metrics)
            perf.update(perf_mode)
            if save_arr:
                arrays[mode_type] = {"g": g_true, "y": y_true, "g_pred": g_pred, "y_pred": y_pred}

        perf_file = os.path.join(self.result_dir, "performance.json")
        with open(perf_file, 'w') as fp:
            json.dump(perf, fp)

        if save_arr:
            for mode, d in arrays.items():
                for arr_name, arr in d.items():
                    np.save(os.path.join(pred_dir, f"{arr_name}-{mode}"), arr)

        return perf, arrays

    def save_checkpoint(self):
        ckpt_file = os.path.join(self.result_dir, "checkpoint")
        state_dict = dict()
        state_dict["model"] = self.model.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["epoch"] = self.epoch
        torch.save(state_dict, ckpt_file)

    def train_model(self, train_loader, test_loader):
        device = self.args.device
        self.model.actor.to(device)
        self.model.baseline.to(device)
        self.model.critic.to(device)
        is_synthetic = "syn" in self.args.data_type
        t_loader = tqdm(range(self.args.max_epochs))
        for i in t_loader:
            summary = self.train_step(train_loader)
            for key, val in summary.items():
                self.train_history[key].append(val)
            self.epoch += 1

            desc = list([f"Epoch: {self.epoch}"])
            for k, v in summary.items():
                desc.append(f"{k}: {v:.3f}")
            desc = " ".join(desc)
            t_loader.set_description(desc)

            if (self.epoch % self.args.eval_freq) == 0:
                performance, _ = self.eval_model(train_loader, test_loader, feature_metrics=is_synthetic, save_arr=False)
                print(json.dumps(performance))

        performance, _ = self.eval_model(train_loader, test_loader)
        print(json.dumps(performance))
        self.plot_results()
        self.save_checkpoint()
        return performance

