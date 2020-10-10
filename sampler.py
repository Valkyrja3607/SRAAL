import torch
import torch.utils.data as data
import numpy as np


class AdversarySampler:
    def __init__(self, budget, args):
        self.budget = budget
        self.args = args

    def distance(self, mu, l_mu):
        dif = l_mu - mu
        dis = dif ** 2
        res = dis.sum(dim=1).min()
        return res.item()

    def sample(self, vae, discriminator, u_data, l_data, t_data, cuda):
        all_preds = []
        all_indices = []

        for images, _, indices in u_data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)

        # need to multiply by -1 to be able to use torch.topk
        # all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget) * 2)
        querry_indices = np.asarray(all_indices)[querry_indices]
        sampler = data.sampler.SubsetRandomSampler(querry_indices)
        k_data = data.DataLoader(
            t_data, sampler=sampler, batch_size=self.args.batch_size, drop_last=False
        )
        # k-center
        dis_l = []
        u_mu = []
        l_mu = []
        all_indices = []
        for u_images, _, indices in k_data:
            if cuda:
                u_images = u_images.cuda()
            with torch.no_grad():
                _, _, _, u_mu_pred, _ = vae(u_images)
                u_mu.extend(u_mu_pred)
                all_indices.extend(indices)
        for l_images, _, l_indices in l_data:
            if cuda:
                l_images = l_images.cuda()
            _, _, _, l_mu_pred, _ = vae(l_images)
            l_mu.extend(l_mu_pred)
        l_mu = torch.stack(l_mu)
        for mu, ind in zip(u_mu, all_indices):
            mu_d = self.distance(mu, l_mu)
            dis_l.append((mu_d, ind.item()))
        dis_l.sort(reverse=True)
        dis_l = dis_l[: int(self.budget)]
        querry_pool_indices = [i[1] for i in dis_l]
        # querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
