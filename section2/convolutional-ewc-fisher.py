import datetime
import logging
import sys
from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage

SCRIPT_NAME = "convolutional-ewc-fisher"
logger = logging.getLogger(SCRIPT_NAME)


class MyMNIST(Dataset):
    def __init__(self, inputs, targets):
        self.data = inputs
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class Model(nn.Module):
    def __init__(self, shape=None, learning_rate=0.001, device=torch.device("cpu")):
        super(Model, self).__init__()

        self.lr = learning_rate
        self.shape = shape
        self.lmbda = 0.
        self.device = device

        # create network layers
        self.network = nn.ModuleList()
        self.network.append(nn.Conv2d(1, 64, 5).to(device))
        self.network.append(nn.BatchNorm2d(64).to(device))
        self.network.append(nn.Conv2d(64, 128, 3).to(device))
        self.network.append(nn.BatchNorm2d(128).to(device))
        self.network.append(nn.Conv2d(128, 256, 3).to(device))
        self.network.append(nn.BatchNorm2d(256).to(device))
        self.network.append(nn.Linear(8*8*256, 10).to(device))

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.lr)

        # create list for parameter importances
        self.importances = [torch.zeros_like(p, device=self.device)
                            for p in self.network.parameters()]

        # create list to store consolidated parameters
        self.star_params = None

        # temporary accumulators for importances
        self.accumulators = [torch.zeros_like(p, device=self.device)
                             for p in self.network.parameters()]

    def forward(self, inputs):
        h = self.network[0](inputs)
        h = self.network[1](h)
        h = F.leaky_relu(h)
        h = F.max_pool2d(h, kernel_size=(2,2))
        h = self.network[2](h)
        h = self.network[3](h)
        h = F.leaky_relu(h)
        #h = F.max_pool2d(h, kernel_size=(2,2))
        h = self.network[4](h)
        h = self.network[5](h)
        h = F.leaky_relu(h)
        #h = F.max_pool2d(h, kernel_size=(2,2))
        h = h.view(h.shape[0], -1)
        h = self.network[6](h)
        return h

    def step(self, inputs, labels):
        z = self.forward(inputs)
        loss = F.cross_entropy(z, labels)

        if self.lmbda != 0. and self.star_params is not None:
            for p, reg, p_star in zip(list(self.network.parameters()), self.regularizers, self.star_params):
                loss += torch.sum(reg * torch.square(p - p_star))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        # reinitialize network
        for layer in self.network:
            layer.reset_parameters()

        # clear importance
        for imp in self.importances:
            imp.fill_(0.)

    def open_lesson(self, lmbda=0.0):
        """
        :param lmbda:         ewc regularization power
        """
        self.lmbda = lmbda
        self.regularizers = [imp * lmbda for imp in self.importances]

    def close_lesson(self, inputs=None, labels=None):
        """
        Закрытие урока обучения сети на отдельном датасете. Расчет и накопление важностей весов.
        :param closing_set: датасет, на котором будут рассчитаны важности весов после обучения
        :return:
        """
        if inputs is None:
            return

        self.eval()

        s_num = len(inputs)

        for accum in self.accumulators:
            accum.fill_(0.)

        for i in range(s_num):
            _input = torch.tensor(inputs[i: i + 1], device=self.device)
            label = labels[i]

            log_probs = F.log_softmax(self.forward(_input), dim=1)

            self.optimizer.zero_grad()
            prob = log_probs[0, label]
            prob.backward()

            for accum, v in zip(self.accumulators, self.network.parameters()):
                accum.add_(v.grad.square())

        for accum, imp in zip(self.accumulators, self.importances):
            imp.add_(accum / s_num)

        self.star_params = [v.clone().detach() for v in self.network.parameters()]


def train_model(model, train_set, test_sets, batch_size=100, epochs=1):
    """
    Single dataset training
    """
    num_iters = int(np.ceil(train_set[0].data.shape[0] * epochs / batch_size))  #
    train_loader = DataLoader(train_set[0], batch_size=batch_size, shuffle=True)
    model.train()
    idx = 0
    for epoch in range(epochs):
        for inputs, labels in iter(train_loader):

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            model.step(inputs=inputs, labels=labels)

            if idx % 67 == 0:
                print(f'\rTraining {idx+1}/{num_iters} iterations done.', end='')
            idx += 1

    print("\r", end='')
    model.eval()
    accuracy = 0.
    with torch.no_grad():
        for t, test_set in enumerate(test_sets):
            inputs = torch.tensor(test_set[1].data, device=model.device)
            logits = model.forward(inputs)
            results = logits.max(-1).indices
            accuracy += np.mean(results.cpu().numpy() == test_set[1].targets)
    accuracy /= len(test_sets)
    logger.info(f'Training {num_iters}/{num_iters} iterations done. '
                f'Mean accuracy on {len(test_sets)} test sets is {accuracy}')
    return accuracy


def continual_learning(model, mnist_datasets, lmbda, batch_size, epoch_num):
    """
    Continual model training on several datasets
    """
    model.reset()
    logger.info('Model has been cleaned.')
    test_datasets = []
    accuracies = []
    task_num = len(mnist_datasets)
    for idx, dataset in enumerate(mnist_datasets):
        test_datasets.append(dataset)
        model.open_lesson(lmbda)
        accuracy = train_model(model, dataset, test_datasets, batch_size, epoch_num)
        accuracies.append(accuracy)
        if idx != task_num - 1:
            model.close_lesson(dataset[0].data[-5000:], dataset[0].targets[-5000:])
            maxs, means = [], []
            for imp in model.importances:
                maxs.append(imp.max().item())
                means.append(imp.mean().item())
            logger.info(f"Max importance {max(maxs)}, max mean importance {max(means)}.")
    return accuracies


def permute_mnist(mnist):
    idxs = list(range(mnist[0].data.shape[1]))
    np.random.shuffle(idxs)
    mnist2 = []
    for dataset in mnist:
        perm_dataset = deepcopy(dataset)
        for i in range(perm_dataset.data.shape[1]):
            perm_dataset.data[:, i] = dataset.data[:, idxs[i]]
        mnist2.append(perm_dataset)
    return tuple(mnist2)


def negative_mnist(mnist):
    mnist2 = []
    for dataset in mnist:
        new_dataset = deepcopy(dataset)
        new_dataset.data = 1. - new_dataset.data
        mnist2.append(new_dataset)
    return tuple(mnist2)


def rotate_mnist(mnist, degrees=0):
    mnist2 = []
    for dataset in mnist:
        new_dataset = deepcopy(dataset)
        for i in range(len(new_dataset)):
            new_dataset.data[i, 0] = ndimage.rotate(new_dataset.data[i, 0], degrees, reshape=False)
        mnist2.append(new_dataset)
    return tuple(mnist2)


def experiments_run():
    # setup logger to output to console and file
    logFormat = "%(asctime)s [%(levelname)s] %(message)s"
    logFile = "./convolutional-ewc-fisher.log"
    logging.basicConfig(filename=logFile, level=logging.INFO, format=logFormat)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Operating device is {device}")

    dataset_file = 'datasets-conv.dmp'
    try:
        mnist_datasets = joblib.load(dataset_file)
        logger.info('Dataset has been loaded from cache.')
    except FileNotFoundError:
        logger.info('Dataset cache not found. Creating new one.')
        # download and transform mnist dataset
        train_data = datasets.MNIST('../mnist_data', download=True, train=True)
        train_inputs = train_data.data.unsqueeze(1).numpy()
        train_inputs = (train_inputs / 255).astype(np.float32)
        train_labels = train_data.targets.numpy()
        train_dataset = MyMNIST(train_inputs, train_labels)
        test_data = datasets.MNIST('../mnist_data', download=True, train=False)
        test_inputs = test_data.data.unsqueeze(1).numpy()
        test_inputs = (test_inputs / 255).astype(np.float32)
        test_labels = test_data.targets.numpy()
        test_dataset = MyMNIST(test_inputs, test_labels)
        mnist = (train_dataset, test_dataset)

        train_data = datasets.FashionMNIST('../fmnist_data', download=True, train=True)
        train_inputs = train_data.data.unsqueeze(1).numpy()
        train_inputs = (train_inputs / 255).astype(np.float32)
        train_labels = train_data.targets.numpy()
        train_dataset = MyMNIST(train_inputs, train_labels)
        test_data = datasets.FashionMNIST('../fmnist_data', download=True, train=False)
        test_inputs = test_data.data.unsqueeze(1).numpy()
        test_inputs = (test_inputs / 255).astype(np.float32)
        test_labels = test_data.targets.numpy()
        test_dataset = MyMNIST(test_inputs, test_labels)
        fmnist = (train_dataset, test_dataset)

        mnist0 = mnist
        mnist1 = fmnist
        mnist2 = rotate_mnist(mnist, 90)
        mnist3 = rotate_mnist(fmnist, 90)

        mnist_datasets = [mnist0, mnist1, mnist2, mnist3]
#        mnist_datasets = [mnist0, mnist1, mnist2, mnist3, mnist4, mnist5, mnist6, mnist7, mnist8, mnist9]
        joblib.dump(mnist_datasets, dataset_file, compress=3)

    exp_file = "convolutional-ewc-fisher.dmp"
    try:
        experiments = joblib.load(exp_file)
    except FileNotFoundError:
        logger.info('Experiment cache not found. Creating new one.')
        experiments = defaultdict(list)

    # network structure and training parameters
    learning_rate = 0.001
    N = 20
    batch_size = 100
    epoch_num = 6

    model = Model(learning_rate=learning_rate, device=device)

    start_time = datetime.datetime.now()
    time_format = "%Y-%m-%d %H:%M:%S"
    logger.info(f'Continual learning start at {start_time:{time_format}}')

    lmbdas = [0, 100, 200, 400, 500, 550, 600, 650, 675, 700, 725, 750, 800, 900, 1000, 1100, 1200, 1400, 1600, 1900,
              2100, 2400, 2700, 2850, 3000, 3250, 3500, 4000]

    for lmbda in lmbdas:
        exps = experiments[lmbda]
        len_exp = len(exps)
        K = max(0, N - len_exp)
        logger.info(f'Start calc on lambda {lmbda}. {K} experiments are queued.')
        for i in range(K):
            iter_start_time = datetime.datetime.now()
            logger.info(f'{i+1+len_exp}-th experiment on lambda {lmbda} started at {iter_start_time:{time_format}}')
            accuracies = continual_learning(model, mnist_datasets, lmbda=lmbda, batch_size=batch_size, epoch_num=epoch_num)
            exps.append(accuracies)
            joblib.dump(experiments, exp_file, compress=1)
            logger.info(f'{i+1}-th experiment time spent {datetime.datetime.now() - iter_start_time}')
            logger.info(f'For now total time spent {datetime.datetime.now() - start_time}')

    logger.info(f'Done for lambdas {lmbdas}')


if __name__ == "__main__":
    experiments_run()
