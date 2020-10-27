import config
import torch
import torch.nn as nn
import numpy as np

class STNkd(nn.Module):
    def __init__(self,  k=64):
        super(STNkd, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(k, 256, kernel_size=1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=1), nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=1), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, k*k),nn.ReLU(),
        )
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv(x)
        x = torch.max(x, 2)[0]
        x = self.fc(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,
                                                                            self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = False, feature_transform = False, stn1_dim = 120,
                 stn2_dim = 64):
        super(PointNetfeat, self).__init__()
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.stn1_dim = stn1_dim
        self.stn2_dim = stn2_dim
        
        self.stn = STNkd(k=stn1_dim)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(stn1_dim, 256, kernel_size=1), nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1), nn.ReLU(),
            nn.Conv1d(256, 1024, kernel_size=1), nn.ReLU(),
            nn.Conv1d(1024, 2048, kernel_size=1), nn.ReLU(),
        )
        
        if self.feature_transform:
            self.fstn = STNkd(k=stn2_dim)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        x = self.conv1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        
        x = self.conv2(x)
        x = torch.max(x, 2)[0]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x[:,:,None].repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat