import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, shape=None, learning_rate=0.001, device=torch.device("cpu")):
        super(Model, self).__init__()

        self.lr = learning_rate
        self.shape = shape
        self.lmbda = 0.
        self.device = device

        # create network layers
        self.network = nn.ModuleList()
        for ins, outs in zip(self.shape[:-1], self.shape[1:]):
            ins_ = np.abs(ins)
            outs_ = np.abs(outs)
            self.network.append(nn.Linear(ins_, outs_).to(self.device))

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
        last_layer_idx = len(self.network) - 1
        for i, layer in enumerate(self.network):
            z = layer(inputs)
            if i < last_layer_idx:
                inputs = F.leaky_relu(z)
        return z

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
            res = F.softmax(self.forward(_input), dim=1).square().sum()
            self.optimizer.zero_grad()
            res.backward()
            for accum, v in zip(self.accumulators, self.network.parameters()):
                accum.add_(v.grad.abs())

        for accum, imp in zip(self.accumulators, self.importances):
            imp.add_(accum / s_num)

        self.star_params = [v.clone().detach() for v in self.network.parameters()]
