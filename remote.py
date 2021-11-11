from nuq.nuq_classifier import NuqClassifier
import random

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from torchvision.datasets import MNIST, CIFAR10, SVHN
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from torch import nn
from torch.nn.utils import spectral_norm


SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

train_transforms = transforms.Compose([
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = MNIST('../checkpoint/data', download=True, train=True, transform=train_transforms)
mnist_test = MNIST('../checkpoint/data', download=True, train=False, transform=test_transforms)


train_loader = DataLoader(mnist_train, batch_size=512, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=512)


def entropy(x):
    x_ = torch.softmax(x, dim=-1)
    return torch.sum(-x_*torch.log(x_), dim=-1)


class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 16, 3, padding=1, bias=False)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 14x14

            spectral_norm(nn.Conv2d(16, 32, 3, padding=1, bias=False)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 7x7

            spectral_norm(nn.Conv2d(32, 32, 3, padding=1, bias=False)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, padding=1), # 4x4

            nn.Flatten(),
            spectral_norm(nn.Linear(512, width, bias=False)),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(),
        )

        self.feature = None
        self.linear = nn.Linear(width, 10)

    def forward(self, x):
        out = self.layers(x)
        self.feature = out.clone().detach()
        return self.linear(out)


def get_model():
    epochs = 5
    model = SimpleConv().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    for e in range(epochs):
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            optimizer.zero_grad()
            preds = model(x_batch)


            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(np.mean(epoch_losses))

    model.eval()
    correct = []
    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            x_batch = x_batch.cuda()
            preds = torch.argmax(model(x_batch).cpu(), dim=-1)
            correct.extend((preds == y_batch).tolist())
    print('Accuracy', np.mean(correct))
    return model


model = get_model()


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        with torch.no_grad():
            x_ = torch.mean(
                torch.stack([m(x.cuda()).cpu() for m in self.models]),
                dim=0
            )
        return x_

    def ue(self, x):
        with torch.no_grad():
            x_ = torch.stack([m(x.cuda()).cpu() for m in self.models])
            x_ = torch.mean(torch.softmax(x_, dim=-1), dim=0)
        return torch.sum(-x_ * torch.log(x_), dim=-1)


ensemble = Ensemble([get_model() for _ in range(5)])
print(ensemble(next(iter(train_loader))[0]))


rotation = (30, 45)
corrupted_transforms = transforms.Compose([
    transforms.RandomRotation(rotation),
    transforms.ToTensor()
])

mnist_corrupted = MNIST('../checkpoint/data', download=True, train=False, transform=corrupted_transforms)
corrupted_loader = DataLoader(mnist_corrupted, batch_size=10_000)
images, labels = next(iter(corrupted_loader))


with torch.no_grad():
    preds = torch.argmax(model(images.cuda()), dim=-1).cpu()

with torch.no_grad():
    preds = model(images.cuda()).cpu()

preds_train = None
for x_batch, y_batch in train_loader:
    with torch.no_grad():
        preds_batch = model(x_batch.cuda()).cpu()
    if preds_train is None:
        preds_train = preds_batch
        y_train = y_batch
    else:
        preds_train = torch.cat((preds_train, preds_batch), dim=0)
        y_train = torch.cat((y_train, y_batch))

preds_train = preds_train.numpy()
y_train = y_train.numpy()


ue_mnist = 1 - torch.max(torch.softmax(preds, dim=-1), dim=-1).values
ue = ue_mnist
ue_entropy = entropy(preds)
ue_ensemble = ensemble.ue(images)




nuq = NuqClassifier(tune_bandwidth='classification', n_neighbors=100)
nuq.fit(X=preds_train, y=y_train)

ue_nuq = nuq.predict_proba(preds.numpy(), return_uncertainty='aleatoric')[1]


train_loader = DataLoader(mnist_train, batch_size=60_000, shuffle=True)
print(train_loader.batch_size)
with torch.no_grad():
    train_images, y_train = next(iter(train_loader))
    y_train = y_train.numpy()
    model(train_images.cuda())
    train_embeddings = model.feature.cpu().numpy()

    model(images.cuda()).cpu()
    embeddings = model.feature.cpu().numpy()
    print(embeddings[:5, :3])

from spectral_normalized_models.ddu import (
    gmm_fit, logsumexp
)
gaussians_model, jitter_eps = gmm_fit(
    embeddings=torch.tensor(train_embeddings), labels=torch.tensor(y_train), num_classes=10
)

ues_test_ddu = gaussians_model.log_prob(torch.tensor(embeddings)[:, None, :].float())
ues_test_ddu = -logsumexp(ues_test_ddu).numpy().flatten()

corrects = (torch.argmax(preds, dim=-1) == labels).numpy()

xs = np.arange(0, 10001, 20)


def show(idx):
    print(torch.argmax(preds, dim=-1)[idx])
    print(labels[idx])
    plt.imshow(images[idx, 0], cmap='gray')
    plt.show()


def splits(ues):
    idxs = np.argsort(ues)
    sorted_corrects = corrects[idxs]
    ys = [1] + [np.mean(sorted_corrects[:num]) for num in xs[1:]]
    return ys

font = { #'family' : 'normal',
        # 'weight' : 'bold',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)
plt.figure(figsize=(6, 5), dpi=150)
linewidth = 3
plt.subplots_adjust(left=0.15, bottom=0.13, right=0.95)
plt.title('Accuracy, MNIST rotated, SN')
plt.ylabel("Accuracy")
plt.xlabel("Samples selected")
plt.plot(xs, splits(ue_mnist), label='MaxProb', linestyle='--', color='tab:green', linewidth=linewidth)
plt.plot(xs, splits(ue_entropy.numpy()), label='Entropy', linestyle=':', color='tab:red', linewidth=linewidth)
plt.plot(xs, splits(ues_test_ddu), label='DDU', linestyle='-.', color='tab:cyan', linewidth=linewidth)
plt.plot(xs, splits(ue_ensemble), label='Ensemble (m=5)', linestyle='-.', color='tab:orange', linewidth=linewidth)
plt.plot(xs, splits(ue_nuq), label='NUQ', linestyle='-.', color='tab:purple', linewidth=linewidth)
plt.legend()
plt.show()

import ipdb; ipdb.set_trace()
