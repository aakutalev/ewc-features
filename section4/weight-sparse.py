#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from model_ewc_fis import Model as Model_EWC_FIS
from model_ewc_mas import Model as Model_EWC_MAS
from model_ewc_si import Model as Model_EWC_SI
from model_ewc_sig import Model as Model_EWC_SIG


SCRIPT_NAME = "weight-sparse"
logger = logging.getLogger(SCRIPT_NAME)

ENTIRE = "entire"
BY_LAYER = "by_layer"

student = { 0: 0.,
            1:  12.7062, 2: 4.3027,  3: 3.1824,  4: 2.7764,  5: 2.5706,  6: 2.4469,  7: 2.3646,  8: 2.3060,
            9:  2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199,
            17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860 }


class MyMNIST(Dataset):
    def __init__(self, inputs, targets):
        self.data = inputs
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# setup logger to output to console and file
logFormat = "%(asctime)s [%(levelname)s] %(message)s"
logFile = "./" + SCRIPT_NAME + ".log"
logging.basicConfig(filename=logFile, level=logging.INFO, format=logFormat)

logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

train_data = datasets.MNIST('../mnist_data', download=True, train=True)
train_inputs = train_data.data.numpy()
train_inputs = (train_inputs.reshape(train_inputs.shape[0], -1) / 255).astype(np.float32)
train_labels = train_data.targets.numpy()
train_dataset = MyMNIST(train_inputs, train_labels)

test_data = datasets.MNIST('../mnist_data', download=True, train=False)
test_inputs = test_data.data.numpy()
test_inputs = (test_inputs.reshape(test_inputs.shape[0], -1) / 255).astype(np.float32)
test_labels = test_data.targets.numpy()
test_dataset = MyMNIST(test_inputs, test_labels)

mnist = (train_dataset, test_dataset)

step = 0.001
net_struct = [784, 300, 150, 10]
lr = 0.001
batch_size = 100
epoch_num = 6


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


def calc_mean_sparse_degradation_by_layer(model_class, lr, lmbda, epochs, tries, backup_file=None, sparse_by_weights=False):
    accuracies = []
    proportion = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(tries):
        print(datetime.datetime.now(), f"iter {i} started.")
        model = model_class(net_struct, lr, device)
        model.open_lesson(lmbda)
        train_model(model, mnist, [mnist], epochs=epochs)
        if not sparse_by_weights:
            model.close_lesson(mnist[1].data, mnist[1].targets)

        # подготавливаем веса и их порядок
        views, orders = [], []
        for n, v in enumerate(model.network.parameters()):
            if len(v.shape) < 2:
                continue
            v1 = v.data.reshape(-1)
            views.append(v1)
            v3 = v1.data.cpu().numpy() if sparse_by_weights else model.importances[n].view(-1).data.cpu().numpy()
            orders.append(np.argsort(np.abs(v3)))
        # циклически обнуляем некоторое количество весов и измеряем точность
        pwtc = 0.0  # proportion of weights to clear
        inputs, labels = torch.tensor(mnist[1].data, device=model.device), mnist[1].targets
        accuracy, proportion = [], []
        prev_max_idxs = np.zeros(len(views), dtype=int)
        while pwtc <= 1.0:
            for n in range(len(views)):
                v = views[n]
                o = orders[n]
                max_idx = int(np.round(o.shape[0] * pwtc))
                #for idx in range(prev_max_idxs[n], max_idx):
                #    v1[o1[idx]] = 0.0
                v[o[prev_max_idxs[n]:max_idx]] = 0.0
                prev_max_idxs[n] = max_idx

            logits = model.forward(inputs)
            results = logits.max(-1).indices
            accuracy.append(np.mean(results.cpu().numpy() == mnist[1].targets))
            proportion.append(pwtc)
            pwtc += step
        print(f'degradation calc complete.')
        accuracies.append(accuracy)
    accuracies = np.asarray(accuracies)
    proportion = np.asarray(proportion)
    if backup_file:
        joblib.dump((accuracies, proportion), backup_file, compress=1)
    return accuracies, proportion


def calc_mean_sparse_degradation_entire(model_class, lr, lmbda, epochs, tries, backup_file=None, sparse_by_weights=False):
    accuracies = []
    proportion = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(tries):
        print(datetime.datetime.now(), f"Iteration {i+1} started.")
        model = model_class(net_struct, lr, device)
        model.open_lesson(lmbda)
        train_model(model, mnist, [mnist], epochs=epochs)
        if not sparse_by_weights:
            model.close_lesson(mnist[1].data, mnist[1].targets)

        # подготавливаем веса и их порядок
        views, imps = [], []
        for n, v in enumerate(model.network.parameters()):
            v1 = v.data.reshape(-1)
            views.append(v1)
            v3 = v1.data.cpu().numpy() if sparse_by_weights else model.importances[n].view(-1).data.cpu().numpy()
            imps.append(v3)
        params = torch.cat(views)
        order = np.argsort(np.abs(np.concatenate(imps)))
        # циклически обнуляем некоторое количество весов и измеряем точность
        pwtc = 0.0  # proportion of weights to clear
        inputs, labels = torch.tensor(mnist[1].data, device=model.device), mnist[1].targets
        accuracy, proportion = [], []
        prev_max_idxs = 0
        b1 = len(views[0])
        b2 = b1 + len(views[1])
        b3 = b2 + len(views[2])
        b4 = b3 + len(views[3])
        b5 = b4 + len(views[4])
        b6 = b5 + len(views[5])
        while pwtc <= 1.0:
            max_idx = int(np.round(params.shape[0] * pwtc))
            for idx in order[prev_max_idxs:max_idx]:
                if idx < b1:
                    views[0][idx] = 0.
                elif idx < b2:
                    views[1][idx-b1] = 0.
                elif idx < b3:
                    views[2][idx-b2] = 0.
                elif idx < b4:
                    views[3][idx-b3] = 0.
                elif idx < b5:
                    views[4][idx-b4] = 0.
                elif idx < b6:
                    views[5][idx-b5] = 0.
                else:
                    raise ValueError(f"ERROR! Index {idx} out of range!")
            prev_max_idxs = max_idx

            logits = model.forward(inputs)
            results = logits.max(-1).indices
            accuracy.append(np.mean(results.cpu().numpy() == mnist[1].targets))
            proportion.append(pwtc)
            pwtc += step
        print(f'degradation calc complete.')
        accuracies.append(accuracy)
    accuracies = np.asarray(accuracies)
    proportion = np.asarray(proportion)
    if backup_file:
        joblib.dump((accuracies, proportion), backup_file, compress=1)
    return accuracies, proportion


sparse_type = ENTIRE  # BY_LAYER
recalc = False  # True  #

file_by_w = sparse_type + '_by_w.dmp'
file_by_fis = sparse_type + '_by_fis.dmp'
file_by_mas = sparse_type + '_by_mas.dmp'
file_by_si = sparse_type + '_by_si.dmp'
file_by_sig = sparse_type + '_by_sig.dmp'

if sparse_type == ENTIRE:
    calc_mean_sparse_degradation = calc_mean_sparse_degradation_entire
else:
    calc_mean_sparse_degradation = calc_mean_sparse_degradation_by_layer


if recalc:
    y1, x1 = calc_mean_sparse_degradation(Model_EWC_FIS, lr, 0.,    epoch_num, tries=10, backup_file=file_by_w, sparse_by_weights=True)
    y2, x2 = calc_mean_sparse_degradation(Model_EWC_FIS, lr, 41.,   epoch_num, tries=10, backup_file=file_by_fis)
    y3, x3 = calc_mean_sparse_degradation(Model_EWC_MAS, lr, 4.5,   epoch_num, tries=10, backup_file=file_by_mas)
    y4, x4 = calc_mean_sparse_degradation(Model_EWC_SI,  lr, 0.25,  epoch_num, tries=10, backup_file=file_by_si)
    y5, x5 = calc_mean_sparse_degradation(Model_EWC_SIG, lr, 0.115, epoch_num, tries=10, backup_file=file_by_sig)
else:
    y1, x1 = joblib.load(file_by_w)
    y2, x2 = joblib.load(file_by_fis)
    y3, x3 = joblib.load(file_by_mas)
    y4, x4 = joblib.load(file_by_si)
    y5, x5 = joblib.load(file_by_sig)


y1s = y1.mean(axis=0)
y1d = student[len(y1)-1] * y1.std(axis=0) / np.sqrt(len(y1))

y2s = y2.mean(axis=0)
y2d = student[len(y2)-1] * y2.std(axis=0) / np.sqrt(len(y2))

y3s = y3.mean(axis=0)
y3d = student[len(y3)-1] * y3.std(axis=0) / np.sqrt(len(y3))

y4s = y4.mean(axis=0)
y4d = student[len(y4)-1] * y4.std(axis=0) / np.sqrt(len(y4))

y5s = y5.mean(axis=0)
y5d = student[len(y5)-1] * y5.std(axis=0) / np.sqrt(len(y5))

plt.figure(figsize=(18, 6))
#plt.title(f'Деградация точности при прунинге сети')
#plt.xlabel('Процент обрезанных весов')
#plt.ylabel('Точность')
plt.title(f'Accuracy degradation on weight pruning')
plt.xlabel('Pruned weights percentage')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0)
#plt.plot(x1 * 100, y1s, label='Обрезка по модулю веса')
plt.plot(x1 * 100, y1s, label='Pruning by abs of weights')
plt.fill_between(x1 * 100, y1s - y1d, y1s + y1d, alpha=0.2)
#plt.plot(x2 * 100, y2s, label='Обрезка по важностям на основе матрицы Фишера')
plt.plot(x2 * 100, y2s, label='Pruning by Fisher importance')
plt.fill_between(x2 * 100, y2s - y2d, y2s + y2d, alpha=0.2)
#plt.plot(x3 * 100, y3s, label='Обрезка по важностям на основе метода MAS')
plt.plot(x3 * 100, y3s, label='Pruning by MAS importance')
plt.fill_between(x3 * 100, y3s - y3d, y3s + y3d, alpha=0.2)
#plt.plot(x4 * 100, y4s, label='Обрезка по важностям на основе метода SI')
plt.plot(x4 * 100, y4s, label='Pruning by SI importance')
plt.fill_between(x4 * 100, y4s - y4d, y4s + y4d, alpha=0.2)
#plt.plot(x5 * 100, y5s, label='Обрезка по важностям на основе суммарного прошедшего сигнала')
plt.plot(x5 * 100, y5s, label='Pruning by total abs signal')
plt.fill_between(x5 * 100, y5s - y5d, y5s + y5d, alpha=0.2)
plt.legend()
plt.show()

print("Done!")
