import copy
from iterator import get_iter
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from models.vcnet import vcnet
from models.acfr import acfr
from models.drnet import drnet

def val(model, val_iter, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        model.eval()
        for batch_idx, (t, x, y) in enumerate(tqdm(val_iter, desc='Batches', leave=False, disable=True)):
            x = x.flatten(start_dim=1).to(device).float()
            t = t.flatten(start_dim=0).to(device).float()
            y = y.flatten(start_dim=0).to(device).float()
            if model_type == 'acfr':
                y_hat = model(x, t)
            elif model_type == 'vcnet':
                y_hat, _, _ = model(x, t)
            elif model_type == "drnet" or model_type == "cfr-wass" or model_type=="cfr-hsic":
                y_hat, _ = model(x, t)
            elif model_type == "mlp":
                y_hat = model(x, t)
            loss = F.mse_loss(y, torch.squeeze(y_hat, 1))
            return loss


def train(train_iter, model, n_epoch, model_type, val_iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = list()
    model.to(device)
    model.train()
    best_val = 1e10
    best_model = None
    for epoch in trange(1, n_epoch + 1, desc='Epochs', leave=True):
        for batch_idx, (t, x, y) in enumerate(tqdm(train_iter, desc='Batches', leave=False, disable=True)):
            x = x.flatten(start_dim=1).to(device).float()
            t = t.flatten(start_dim=0).to(device).float()
            y = y.flatten(start_dim=0).to(device).float()
            if model_type == "acfr":
                if epoch % 1 == 0:
                    t_hat = model.forward_D(x)
                    l_t, err = model.backward_D(t, t_hat)
                if epoch % 1 == 0:
                    y_hat1, t_hat, y_hat2 = model.forward_G(x, t, err)
                    l = model.backward_G(t, y, t_hat, y_hat1, y_hat2, err)
                    loss.append(l.item())
            elif model_type == "vcnet":
                y_hat, _ , gps = model(x, t)
                model.backward(y, y_hat, gps)
            elif model_type == "drnet":
                y_hat, _ = model(x, t, )
                model.backward(y, y_hat, 0)
            elif model_type == "cfr-wass":
                y_hat, disc = model(x, t, "wass")
                model.backward(y, y_hat, disc)
                pass
            elif model_type == "cfr-hsic":
                y_hat, disc = model(x, t, "hsic")
                model.backward(y, y_hat, disc)
                pass
            elif model_type == "mlp":
                y_hat = model(x, t)
                model.backward(y, y_hat)

        val_loss = val(model, val_iter, model_type)
        if val_loss < best_val and epoch > n_epoch/2:
            best_model = copy.deepcopy(model)
            best_val = val_loss

    return best_model, best_val

def run(cfg, dataset_directory, model_type, dataset_type, file):
    models = dict()
    val_losses = list()
    model = None
    if model_type == 'vcnet':
        print("Results on: ", cfg, file=file)
        model = vcnet(cfg)
    elif model_type == 'acfr':
        print("Results on: ", cfg, file=file)
        model = acfr(cfg)
    elif model_type == 'drnet' or model_type == 'cfr-hsic' or model_type == 'cfr-wass':
        print("Results on: ", cfg, file=file)
        model = drnet(cfg)

    torch.save(model, 'init_models/' + model_type + "_" + dataset_type + "_init.pth")

    for i in range(int(cfg['realization_number'])):
        model = torch.load('init_models/' + model_type + "_" + dataset_type + "_init.pth")
        print("Training model on dataset: ", str(i), file=file)
        train_data_dir = dataset_directory + str(i) + "/train.csv"
        val_data_dir = dataset_directory + str(i) + "/val.csv"
        train_iter, val_iter = get_iter(train_data_dir, val_data_dir, dataset_type, batch_size=cfg['batch_size'])
        model, val_loss = train(train_iter, model, int(cfg['epoch_number']), model_type, val_iter)
        print("Validation loss", val_loss, file=file)
        models[i] = copy.deepcopy(model)
        val_losses.append(val_loss.item())

    return models



