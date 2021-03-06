import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy


class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget, self.args.batch_size)

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img
    """
    def OUI(self, v):
        v = nn.functional.softmax(v, dim=1)
        v_max, idx = torch.max(v, 1)
        bc = v.size()[0]
        c = v.size()[1]
        v_ = (1 - v_max) / (c - 1)  # [128]
        v__ = [[] for i in range(bc)]
        v__ = [[j.item()] * (c - 1) + [v_max[i]] for i, j in enumerate(v_)]
        v_ = torch.tensor(v__)  # [128,10]
        var_v = torch.var(v, 1)
        min_var = torch.var(v_, 1)
        if self.args.cuda:
            var_v = var_v.cuda()
            min_var = min_var.cuda()
        indicator = 1 - min_var / var_v * v_max  # [128]
        return indicator.detach()
    """
    def OUI(self, v):
        v = nn.functional.softmax(v, dim=1)
        v_max, idx = torch.max(v, 1)
        c = v.size()[1]
        min_var = ((v_max - 1 / c) ** 2 + (c - 1) * (1 / c - (1 - v_max) / (c - 1)) ** 2) / c
        var_v = torch.var(v, 1)
        indicator = 1 - min_var / var_v * v_max
        return indicator

    def train(
        self,
        querry_dataloader,
        val_dataloader,
        task_model,
        vae,
        discriminator,
        unlabeled_dataloader,
    ):
        self.args.train_iterations = (
            self.args.num_images * self.args.train_epochs
        ) // self.args.batch_size
        lr_change = self.args.train_iterations // 4
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.SGD(
            task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9
        )
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()

        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param["lr"] = param["lr"] / 10
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % 100 == 0:
                print("Current training iteration: {}".format(iter_count))
                print("Current task model loss: {:.4f}".format(task_loss.item()))

            if iter_count % 1000 == 0:
                acc = self.validate(task_model, val_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)

                print("current step: {} acc: {}".format(iter_count, acc))
                print("best acc: ", best_acc)

        for iter_count in range(self.args.train_iterations):
            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param["lr"] = param["lr"] / 10
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # VAE step
            for count in range(self.args.num_vae_steps):
                recon, y, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(
                    labeled_imgs, recon, mu, logvar, self.args.beta
                )
                unlab_recon, unlab_y, unlab_z, unlab_mu, unlab_logvar = vae(
                    unlabeled_imgs
                )
                transductive_loss = self.vae_loss(
                    unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, self.args.beta
                )
                stl_loss = self.ce_loss(y, labels)

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                # discriminatorにラベル付きだと思わせる(理想のdiscriminatorの出力は0)
                # lab_real_preds = torch.ones(labeled_imgs.size(0))
                # unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                lab_real_preds = torch.zeros(labeled_imgs.size(0))
                unlab_real_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                # dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(
                dsc_loss = self.mse_loss(labeled_preds, lab_real_preds) + self.mse_loss(
                    unlabeled_preds, unlab_real_preds
                )
                total_vae_loss = (
                    unsup_loss
                    + transductive_loss
                    + stl_loss
                    + self.args.adversary_param * dsc_loss
                )
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, _, mu, _ = vae(labeled_imgs)
                    _, _, _, unlab_mu, _ = vae(unlabeled_imgs)

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                # v_l = task_model(labeled_imgs)
                v_u = task_model(unlabeled_imgs)
                # lab_real_preds = self.OUI(v_l)
                unlab_fake_preds = self.OUI(v_u)

                lab_real_preds = torch.zeros(labeled_imgs.size(0))
                # unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()

                # dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(
                dsc_loss = self.mse_loss(labeled_preds, lab_real_preds) + self.mse_loss(
                    unlabeled_preds, unlab_fake_preds
                )

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            if iter_count % 100 == 0:
                print("Current training iteration: {}".format(iter_count))
                # print("Current task model loss: {:.4f}".format(task_loss.item()))
                print("Current vae model loss: {:.4f}".format(total_vae_loss.item()))
                print("Current discriminator model loss: {:.4f}".format(dsc_loss.item()))

                # print("dl,du:", labeled_preds[0].item(), unlabeled_preds[0].item())
                # print("ol,ou:", lab_real_preds[0].item(), unlab_fake_preds[0].item())

                # print(unlabeled_preds[:5])
                # print(lab_real_preds[:5])
                # print(unlab_fake_preds[:5])

            if iter_count % 1000 == 0:
                acc = self.validate(task_model, val_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)

                print("current step: {} acc: {}".format(iter_count, acc))
                print("best acc: ", best_acc)

        if self.args.cuda:
            best_model = best_model.cuda()

        final_accuracy = self.test(best_model)
        return final_accuracy, vae, discriminator

    def sample_for_labeling(
        self, vae, discriminator, unlabeled_dataloader, querry_dataloader, t_data
    ):
        querry_indices = self.sampler.sample(
            vae,
            discriminator,
            unlabeled_dataloader,
            querry_dataloader,
            t_data,
            self.args.cuda,
        )

        return querry_indices

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD

