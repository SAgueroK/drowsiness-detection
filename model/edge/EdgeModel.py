import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.nn import init
from torch.utils.data import DataLoader, Subset

from Mycode.MyDataset import MyCustomDataset
from Mycode.edgeAi.LocalAtt import LocalAttention
from Mycode.models.SkeletonKAN import SkeletonKAN, val
from Mycode.models.kan import KANLinear
from Mycode.show_graphy import load_dataset
from Mycode.utils import loadModel, time_series_decomposition, sample, record_result
import torch.nn.functional as F


class EdgeModel(nn.Module):
    def __init__(self, local_CNNs, regional_CNNs, Kan_input, Kan_num, Kan_hidden, fc):
        super(EdgeModel, self).__init__()
        self.local_CNNs = local_CNNs
        self.regional_CNNs = regional_CNNs
        self.local_att = LocalAttention(channels=1, window_size=3)
        self.fc = fc
        if fc == None:
            self.fc = nn.ModuleList()
            if Kan_num == 1:
                self.fc.append(KANLinear(in_features=Kan_input,
                                         out_features=1))
            else:
                self.fc.append(KANLinear(in_features=Kan_input,
                                         out_features=Kan_hidden))
                if Kan_num > 2:
                    for _ in range(Kan_num - 2):
                        self.fc.append(KANLinear(in_features=Kan_hidden, out_features=Kan_hidden))
                self.fc.append(KANLinear(in_features=Kan_hidden, out_features=1))

    def forward(self, src):
        src = torch.transpose(src, 0, 1)
        local_results = []
        regional_results = []
        for idx in range(src.shape[0]):
            local_results.append(self.local_CNNs[idx](src[idx]).unsqueeze(0))
            regional_results.append(self.regional_CNNs[idx](src[idx]).unsqueeze(0))
        local_res = torch.cat(local_results, 0)
        regional_res = torch.cat(regional_results, 0)
        local_res = torch.transpose(local_res, 0, 1).contiguous()
        regional_res = torch.transpose(regional_res, 0, 1).contiguous()
        shapes = local_res.shape
        self.local_res = local_res.view(shapes[0], 1, shapes[1], shapes[2])
        self.regional_res = regional_res.view(shapes[0], 1, shapes[1], shapes[2])
        self.fusion, att = self.local_att(self.local_res, self.regional_res)
        out = self.fusion + self.local_res
        out = out.view(out.size(0), -1)
        for idx in range(len(self.fc)):
            out = self.fc[idx](F.relu(out))
        return F.sigmoid(out)


if __name__ == '__main__':
    epochs = 10
    window_split = 8
    kernel_size = 13
    kan_num = 2
    kan_hidden = 16
    for root, dirs, files in os.walk('../../datasets'):
        for Dir in dirs:
            print(Dir)
            X, Y = load_dataset(os.path.join(root, Dir), 40, 40, get_Iframe_num=45)
            if X is None:
                continue
            window_sizes = [0]
            for i in range(2, window_split + 1):
                window_sizes.append((85 - 1) // i)
            # X = X[:, 5:6, :, :]
            X = time_series_decomposition(X, window_sizes=window_sizes)

            shapes = X.shape
            X = X.view(-1, 2)
            x_min, _ = X.min(dim=0, keepdim=True)  
            x_max, _ = X.max(dim=0, keepdim=True)  
            X = (X - x_min) / (x_max - x_min)
            X = X.view(shapes)

            X = X.permute(0, 2, 3, 1, 4)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1, 1)
            X = X.permute(0, 3, 1, 2, 4)

            datasets = MyCustomDataset(X.cuda(), Y.cuda())
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            res = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(datasets)):
                print('Fold:', fold)
                train_subset = Subset(datasets, train_idx)
                val_subset = Subset(datasets, val_idx)
                train_loader = sample(train_subset)
                val_dataloader = DataLoader(val_subset, 1, shuffle=True)
                shapes = train_subset[0][0].shape
                local_model = torch.load('your local model files path')
                local_model_CNNs = local_model.CNNs
                local_model_fc = local_model.fc
                regional_model = torch.load('your regional model files path')
                model = EdgeModel(local_model_CNNs, regional_model, Kan_input=shapes[2] * shapes[0], Kan_num=2,
                                  Kan_hidden=16, fc=local_model_fc).to('cuda')

                for param in model.local_CNNs.parameters():
                    param.requires_grad = False

                for param in model.fc.parameters():
                    param.requires_grad = False

                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,
                                              weight_decay=0.1)
                ptimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                LOSS = nn.BCELoss()
                your_train()
