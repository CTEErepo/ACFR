import numpy as np
import os
from sklearn.model_selection import train_test_split
NUMBER_OF_FEATURES = 3477

def make_count_matrix(dir):
    x_dir = dir + 'topic_doc_mean_n5000_k3477_seed_' + str(1) + '.csv.x'
    n = np.loadtxt(x_dir, delimiter=",", dtype=int)[1:, :]
    count_data = np.zeros((5000, 3477))
    for j in range(n.shape[0]):
        count_data[n[j, 0] - 1, n[j, 1] - 1] = n[j, 2]
    return count_data

def get_v():
    v1p, v2p, v3p = np.random.uniform(0, 1, NUMBER_OF_FEATURES), np.random.uniform(0, 1,
                                                                                 NUMBER_OF_FEATURES), np.random.uniform(
        0, 1, NUMBER_OF_FEATURES)
    v1, v2, v3 = v1p / np.linalg.norm(v1p, 2), v2p / np.linalg.norm(v2p, 2), v3p / np.linalg.norm(v3p, 2)
    return v1, v2, v3

def compute_beta(alpha, optimal_dosage):
    if (optimal_dosage <= 0.001 or optimal_dosage >= 1.0):
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(optimal_dosage) + (2.0 - alpha)

    return beta

def get_t(x, v2, v3):
    optimal_dosage = np.dot(v3, x) / (2.0 * np.dot(v2, x))
    alpha = 2
    t = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
    if t <= 0.001:
        t = 0.001
    elif t >= 0.999:
        t = 0.999
    return t


def get_y(t, x, v1, v2, v3):

    y = 10.0 * (np.dot(v1, x) + np.sin(
        np.pi * (np.dot(v2, x) / np.dot(v3, x)) * t))
    return y + np.random.normal(0, 0.2)

def po(test_data, v1, v2, v3, n_grids=2**6+1):
    n_test = test_data.shape[0]
    ts = np.linspace(0.01, 1, n_grids)
    t_grids = np.zeros((n_test + 1, n_grids))
    t_grids[0, :] = ts.squeeze()
    t_grid = np.zeros((2, n_grids))
    t_grid[0, :] = ts.squeeze()

    for j in range(n_grids):
        t = ts[j]
        psi = 0
        for i in range(n_test):
            x = test_data[i, 0:NUMBER_OF_FEATURES]
            y_hat_ij = get_y(t, x, v1, v2, v3)
            t_grids[i + 1, j] = y_hat_ij
            psi += y_hat_ij
        t_grid[1, j] = (psi / n_test)
    return t_grid, t_grids


def make_continuous(data, v1, v2, v3):
    tmp = np.zeros((data.shape[0], 2))
    continuous_data = np.append(data, tmp, axis=1)
    for row in range(data.shape[0]):
        ## adding continuous treatment
        t = get_t(data[row, :], v2, v3)
        continuous_data[row, data.shape[1]] = t

        ## adding outcome
        y = get_y(t, data[row, :], v1, v2, v3)
        continuous_data[row, data.shape[1] + 1] = y

    return continuous_data

def normalize_data(data):
    x = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-10)
    for i in range(x.shape[0]):
        x[i] = x[i] / np.linalg.norm(x[i])
    return x


def save_files(dir='../dataset/news/', run_n=50):
    data = make_count_matrix(dir)
    data = normalize_data(data)
    v1, v2, v3 = get_v()
    for _ in range(run_n):
        print("generating dataset: ", str(_))
        curr_dir = dir + str(_)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)


        continuous_data = make_continuous(data, v1, v2, v3)

        train, test = train_test_split(continuous_data, train_size=0.8)
        train, val = train_test_split(train, train_size=0.85)


        np.savetxt(curr_dir + "/train.csv", train, delimiter=",")
        np.savetxt(curr_dir + "/val.csv", val, delimiter=",")
        np.savetxt(curr_dir + "/test.csv", test, delimiter=",")

        out_t_grid, out_t_grids = po(test, v1, v2, v3)
        in_t_grid, in_t_grids = po(train, v1, v2, v3)


        np.savetxt(curr_dir + "/out_t_grids.csv", out_t_grids, delimiter=",")
        np.savetxt(curr_dir + "/out_t_grid.csv", out_t_grid, delimiter=",")

        np.savetxt(curr_dir + "/in_t_grids.csv", in_t_grids, delimiter=",")
        np.savetxt(curr_dir + "/in_t_grid.csv", in_t_grid, delimiter=",")


np.random.seed(5)
run_n = 20
save_files(run_n=run_n)

