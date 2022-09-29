import torch.nn as nn
from models.subnet import EncoderNet, PredNet
import torch.nn.functional as F
from utils import *

class drnet(nn.Module):
    def __init__(self, args):
        super(drnet, self).__init__()
        self.args = args
        self.NetE = EncoderNet(self.args['encoder_net']['input_dim'], self.args['encoder_net']['hidden_dims'],
                               self.args['encoder_net']['output_dim'])
        self.NetP1 = PredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                             self.args['prediction_net']['output_dim'])
        self.NetP2 = PredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                             self.args['prediction_net']['output_dim'])
        self.NetP3 = PredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                             self.args['prediction_net']['output_dim'])
        self.NetP4 = PredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                             self.args['prediction_net']['output_dim'])
        self.NetP5 = PredNet(self.args['prediction_net']['input_dim'], self.args['prediction_net']['hidden_dims'],
                             self.args['prediction_net']['output_dim'])

        self.optimizer = torch.optim.Adam(
            params=list(self.NetP1.parameters()) + list(self.NetP2.parameters()) + list(self.NetP3.parameters()) + list(self.NetP4.parameters()) +
                   list(self.NetP5.parameters()) + list(self.NetE.parameters()),
            lr=float(self.args['lr']), weight_decay=float(self.args['weight_decay']))

    def forward(self, x, t, disc=None):
        z = self.NetE(x)
        z = F.normalize(z, p=2, dim=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        u = torch.cat([z, torch.unsqueeze(t, 1)], dim=1)
        v = torch.cat([z, torch.rand(t.shape[0], 1).to(device)], dim=1)
        if disc == "wass":
            disc, _ = wasserstein(u, v, 0.5)
        if disc == "hsic":
            disc = hsic_gam(z, torch.unsqueeze(t, 1))
        ys = torch.zeros(t.shape[0], 5).to(device)
        ys[:, 0] = torch.squeeze(self.NetP1(z, t))
        ys[:, 1] = torch.squeeze(self.NetP2(z, t))
        ys[:, 2] = torch.squeeze(self.NetP3(z, t))
        ys[:, 3] = torch.squeeze(self.NetP4(z, t))
        ys[:, 4] = torch.squeeze(self.NetP5(z, t))
        y_hat = ys.gather(1, torch.floor((t-1e-6)*5).long().unsqueeze(1))
        return y_hat, disc


    def backward(self, y, y_hat, disc):
        self.optimizer.zero_grad()
        loss = F.mse_loss(torch.unsqueeze(y, 1), y_hat) + float(self.args['gamma']) * disc
        loss.backward()
        self.optimizer.step()

