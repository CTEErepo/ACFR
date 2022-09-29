import numpy as np
import statistics
import torch
import argparse
import yaml
import training
import test

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ihdp', 'news', 'tcga'], help='Dataset to use')
    parser.add_argument('--model', choices=['acfr', 'vcnet', 'drnet', 'cfr-hsic', 'cfr-wass'], help='Model to use')

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    yaml_file_directory = 'configs/config_' + args.dataset + "_" + args.model + ".yaml"
    with open(yaml_file_directory) as file:
        parameter_dic = yaml.load(file, Loader=yaml.FullLoader)
    cfg = parameter_dic['parameters']



    f = open(args.model + "_" + args.dataset + ".txt", 'a')
    print("#####################", file=f)
    dataset_directory = "dataset/" + args.dataset + "/"

    models = training.run(cfg, dataset_directory, args.model, args.dataset, f)

    print("#####################", file=f)
    print("out of sample evaluation: ", file=f)
    MISEs, pe = test.eval(models, int(cfg['realization_number']), dataset_directory, args.model, args.dataset, out_of_sample="yes", file=f)

    if len(MISEs) > 1:
        print(f'MISE = {statistics.mean(MISEs)} +-  {statistics.stdev(MISEs)}', file=f)
        print(f'pe = {statistics.mean(pe)} +-  {statistics.stdev(pe)}', file=f)

    else:
        print(f'MISE = {statistics.mean(MISEs)}', file=f)
        print(f'pe = {statistics.mean(pe)}', file=f)


    print("#########################", file=f)
    print("with in sample evaluation", file=f)
    MISEs, pe = test.eval(models, int(cfg['realization_number']), dataset_directory, args.model, args.dataset, out_of_sample="no", file=f)

    if len(MISEs) > 1:
        print(f'MISE = {statistics.mean(MISEs)} +-  {statistics.stdev(MISEs)}', file=f)
        print(f'pe = {statistics.mean(pe)} +-  {statistics.stdev(pe)}', file=f)
    else:
        print(f'MISE = {statistics.mean(MISEs)}', file=f)
        print(f'pe = {statistics.mean(pe)}', file=f)







