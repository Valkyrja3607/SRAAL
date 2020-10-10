import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch import nn
import vgg
import matplotlib.pyplot as plt

n = float(input("split:"))

batch_size = 128
data_path = "./data"

# 正規化
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5,], std=[0.5, 0.5, 0.5]),
    ]
)

trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

num_images = 50000
num_val = 5000
budget = 2500
initial_budget = 5000
num_classes = 10


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


task_model = vgg.vgg16_bn(num_classes=num_classes)


# GPU or CPU の判別
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# modelの定義
model = task_model.to(device)
optimizer = torch.optim.SGD(
    task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9
)
criterion = nn.CrossEntropyLoss(reduction="mean")
splits = [n]
for split in splits:
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainset, [int(num_images * split), num_images - int(num_images * split)]
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train
    print("train")
    model = model.train()
    loss_train_list = []
    acc_train_list = []
    total, tp = 0, 0
    epoch = 80
    for i in range(epoch):
        acc_count = 0
        for j, (x, y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            # 体格行列でone-hotに変換
            # y=torch.eye(10)[y].to(device)

            # 推定
            predict = model.forward(x)

            # loss(bachの平均)
            loss = criterion(predict, y)
            # eps=1e-7
            # loss=-torch.mean(y*torch.log(predict+eps))
            loss_train_list.append(loss.item())
            # acc
            pre = predict.argmax(1).to(device)
            total += y.shape[0]
            tp += (y == pre).sum().item()
            # 勾配初期化
            optimizer.zero_grad()
            # 勾配計算(backward)
            loss.backward()
            # パラメータ更新
            optimizer.step()
            """
            # 進捗報告
            if j % 100 == 0:
                print(
                    "%.01fsplit,%03depoch, %05d, loss=%.5f, acc=%.5f"
                    % (split, i, j, loss.item(), tp / total)
                )"""
        acc = tp / total
        acc_train_list.append(acc)
    model = model.eval()
    total, tp = 0, 0
    for (x, y) in testloader:
        x = x.to(device)
        # 推定
        predict = model.forward(x)
        pre = predict.argmax(1).to("cpu")  # one-hotからスカラ―値

        # answer count
        total += y.shape[0]
        tp += (y == pre).sum().item()
    acc = tp / total
    print("split", split)
    print("final accuracy=%.3f" % acc)

