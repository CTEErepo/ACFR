import torch
import pandas as pd
import torch.nn.functional as F

def dpe(o, po):
    ys = torch.amax(o, axis=1)
    argyhats = torch.argmax(po, axis=1)
    recys = torch.gather(o, 1, torch.unsqueeze(argyhats, 1))
    return F.mse_loss(ys, torch.squeeze(recys))


def test(model, test_data_dir, t_grids, t_grid, model_type, dataset_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = pd.read_csv(test_data_dir, header=None).to_numpy()
    if t_grid is not None:
        t_grid = pd.read_csv(t_grid, header=None).to_numpy()
        y = t_grid[1, :]
        y = torch.unsqueeze(torch.Tensor(y), 1).to(device)
        y_hat = torch.zeros(y.shape).to(device)

    t_grids = pd.read_csv(t_grids, header=None).to_numpy()

    t = t_grids[0, :]
    if dataset_type == 'ihdp':
        x = test_data[:, :25]
    elif dataset_type == 'news':
        x = test_data[:, :3477]
    elif dataset_type == 'tcga':
        x = test_data[:, :4000]
    ys = t_grids[1:, :]

    model.to(device)
    t = torch.unsqueeze(torch.Tensor(t), 1).to(device)
    x = torch.Tensor(x).to(device)
    ys = torch.Tensor(ys).to(device)
    y_hats = torch.zeros((x.shape[0], t.shape[0])).to(device)
    with torch.no_grad():
        model.eval()
        for i in range(len(t)):
            tensor_i = torch.ones(x.shape[0], 1).to(device)
            tensor_i *= t[i]
            if model_type == 'acfr':
                out = model(x, tensor_i)
            elif model_type == 'vcnet':
                tensor_i = torch.squeeze(tensor_i, 1)
                out, z, _ = model(x, tensor_i)
            elif model_type == "drnet" or model_type == "cfr-wass" or model_type == "cfr-hsic":
                tensor_i = torch.squeeze(tensor_i, 1)
                out, _ = model(x, tensor_i)
            elif model_type == "mlp":
                tensor_i = torch.squeeze(tensor_i, 1)
                out = model(x, tensor_i)
            y_hats[:, i] = torch.squeeze(out, 1)
            if t_grid is not None:
                y_hat[i] = torch.mean(out)
    if t_grid is not None:
        #print(torch.cat([y, y_hat], dim=1))
        AMSE = F.mse_loss(y, y_hat)


    MISE = torch.zeros(ys.shape[1], 1).to(device)

    for i in range(ys.shape[1]):
        MISE[i] = F.mse_loss(ys[:, i], y_hats[:, i])



    MISE = torch.mean(MISE)
    pe = dpe(ys, y_hats)

    return MISE, pe

def eval(models, run_number, dataset_directory, model_type, dataset_type, out_of_sample, file):
    MISE_values = list()
    pe_values = list()
    for i in range(run_number):
        if out_of_sample == 'yes':
            test_data_dir = dataset_directory + str(i) + "/test.csv"
            t_grid = dataset_directory + str(i) + "/out_t_grid.csv"
            t_grids = dataset_directory + str(i) + "/out_t_grids.csv"
        elif out_of_sample == 'no':
            test_data_dir = dataset_directory + str(i) + "/train.csv"
            t_grid = dataset_directory + str(i) + "/in_t_grid.csv"
            t_grids = dataset_directory + str(i) + "/in_t_grids.csv"
        if type(models) == dict:
            MISE, pe = test(models[i], test_data_dir, t_grids, t_grid, model_type, dataset_type)
        else:
            MISE, pe = test(models, test_data_dir, t_grids, t_grid, model_type, dataset_type)

        MISE_values.append(MISE.item())
        pe_values.append(pe.item())

    return MISE_values, pe_values
