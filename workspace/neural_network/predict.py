# -*- coding: utf-8 -*-
"""
@author: wfnian
"""

import time

import torch
from torch import nn
from torch.autograd import Variable


class twentyclassification(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(twentyclassification, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def predict_result(datas=None):
    """
    :param datas: list
    :return: int
    """
    if datas is None:
        datas = []
    model = twentyclassification(30, 200, 300, 100, 5)
    model.load_state_dict(torch.load(
        "../model_pth/4classification.pth", map_location='cpu'))
    predict = model(Variable(torch.Tensor([datas]).float())).detach().cpu().numpy().tolist()[0]
    predict = predict.index(max(predict))

    return predict


if __name__ == '__main__':
    data = [0.0, 0.0, 679533.048848439, 1409899.0273498371, 711344.6134389378, 711344.6134389378, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, -0.6641785843316214, 0.8409670355441904, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 1.0]
    print(predict_result(data))
    predict_result(data)
