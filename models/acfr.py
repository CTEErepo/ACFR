import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from models.subnet import EncoderNet, PredNet, DiscNet
import torch.nn.functional as F


class acfr(nn.Module):
    def __init__(self, args):
        super(acfr, self).__init__()
        self.args = args
        self.NetE = EncoderNet(self.args['encoder_net']['input_dim'], self.args['encoder_net']['hidden_dims'],
                               self.args['encoder_net']['output_dim'])
        self.NetP = PredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                            self.args['prediction_net']['output_dim'])
        self.NetD = DiscNet(self.args['discrimination_net']['input_dim'], self.args['discrimination_net']['hidden_dims'],
                            self.args['discrimination_net']['output_dim'])


        ## G means combination of Encoder and Predictor
        G_parameters = list(self.NetE.parameters()) + list(self.NetP.parameters())
        self.optimizer_G = torch.optim.Adam(params=G_parameters, lr=float(args['lr1']), weight_decay=float(args['weight_decay']))
        self.optimizer_D = torch.optim.Adam(params=self.NetD.parameters(), lr=float(args['lr2']), weight_decay=float(args['weight_decay']))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.9999)
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.9999)

    def forward_G(self, x, t, err):
        z = self.NetE(x)
        z = F.normalize(z, p=2, dim=1)
        y_hat1 = self.NetP(z, t)
        t_hat = self.NetD(z)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        newz = z.repeat_interleave(self.args['m'], dim=0) + torch.mul(torch.normal(mean=0, std=1, size=(self.args['m']*z.shape[0], z.shape[1])).to(device), self.args['std']+ 100 * err).to(device)
        y_hat2 = self.NetP(newz, torch.unsqueeze(t, 1).repeat_interleave(self.args['m'], 0))
        return y_hat1, t_hat, y_hat2

    def forward_D(self, x, verbose=False):
        z = self.NetE(x)
        z = F.normalize(z, p=2, dim=1)
        t_hat = self.NetD(z)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eps = torch.normal(0, std=0.1, size=t_hat.shape).to(device)
        t_hat2 = eps + t_hat
        return t_hat2


    def forward(self, x, t):
        z = self.NetE(x)
        z = F.normalize(z, p=2, dim=1)
        y_hat = self.NetP(z, t)
        return y_hat

    def backward_G(self, t, y, t_hat, y_hat1, y_hat2, err):
        for name, param in self.NetP.state_dict().items():
            param.requires_grad = True
        for name, param in self.NetE.state_dict().items():
            param.requires_grad = True
        for name, param in self.NetD.state_dict().items():
            param.requires_grad = False
        self.optimizer_G.zero_grad()
        g1 = F.mse_loss(torch.unsqueeze(y, 1), y_hat1)
        g2 = F.mse_loss(torch.unsqueeze(t, 1), t_hat)
        g3 = F.mse_loss(torch.unsqueeze(y, 1).repeat_interleave(self.args['m'], 0), y_hat2)
        G_loss = F.relu(g1 - self.args['gamma1'] * g2 + self.args['gamma2']*g3)
        G_loss.backward()
        self.optimizer_G.step()
        return g1

    def backward_D(self, t, t_hat):
        for name, param in self.NetP.state_dict().items():
            param.requires_grad = False
        for name, param in self.NetE.state_dict().items():
            param.requires_grad = False
        for name, param in self.NetD.state_dict().items():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        D_loss = F.mse_loss(torch.unsqueeze(t, 1), t_hat)
        D_loss.backward()
        self.optimizer_D.step()
        for name, param in self.NetD.named_parameters():
            if name == 'seqdisc.0.weight':
                err = torch.sum(param.grad, dim=1)
        return D_loss, err.detach()




