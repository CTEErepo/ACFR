import torch
from models.subnet import EncoderNet, DynamicPredNet, DensityNet
import torch.nn as nn
import torch.nn.functional as F


class vcnet(nn.Module):
    def __init__(self, args):
        super(vcnet, self).__init__()
        self.args = args
        self.NetE = EncoderNet(self.args['encoder_net']['input_dim'], self.args['encoder_net']['hidden_dims'],
                               self.args['encoder_net']['output_dim'])
        self.NetP = DynamicPredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                                   self.args['prediction_net']['output_dim'])
        self.NetD = DensityNet(self.args['density_net']['input_dim'], self.args['density_net']['hidden_dims'],
                               self.args['density_net']['output_dim'])
        self.optimizer = torch.optim.Adam(
            params=list(self.NetP.parameters()) + list(self.NetE.parameters()) + list(self.NetD.parameters()), lr=float(args['lr']), weight_decay=float(args['weight_decay']))

    def forward(self, x, t):
        z = self.NetE(x)
        z = F.normalize(z, p=2, dim=1)
        gps = self.NetD(z, t)
        y_hat = self.NetP(z, t)
        return y_hat, z, gps

    def get_loss(self, y, y_hat, gps):
        return F.mse_loss(torch.unsqueeze(y, 1), y_hat) - float(self.args['gamma']) * torch.mean(torch.log(gps))

    def backward(self, y, y_hat, gps):
        self.optimizer.zero_grad()
        loss = self.get_loss(y, y_hat, gps)
        loss.backward()
        self.optimizer.step()



