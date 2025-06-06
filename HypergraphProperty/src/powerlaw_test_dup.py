import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from tqdm import tqdm
import argparse

def read_data(dir_path, dataname, fname):
    data = []

    if "sv" in fname:
        with open(dir_path + fname + ".txt", "r") as f:
            tmp = {}
            X = []
            for li, line in enumerate(f.readlines()):
                sv = float(line.rstrip())
                tmp[li + 1] = sv
                X.append(li + 1)
            X = sorted(X)
            if dataname not in ["tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu", "threads-math-sx", "coauth-MAG-Geology-full"]:
                X = X[:min(1000, int(len(X) * 0.5))]
            elif dataname in ["tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu", "threads-math-sx"]:
                X = X[:1000]
            elif dataname in [ "coauth-MAG-Geology-full"]:
                X = X[:500]

            for x in X:
                data += [x] * round(tmp[x] * 100)

    else:
        with open(dir_path + fname + "_total.txt", "r") as f:
            total = float(f.readline().rstrip())
            total = min(100000000, total)
        with open(dir_path + fname + ".txt", "r") as f:
            for line in f.readlines():
                tmp = line.rstrip().split(",")
                x, y = int(tmp[0]), float(tmp[1])
                count = round(y * total)
                data += [x] * count

    return data

def powerlaw_test(test_type):
    fname = test_type

    dataset = ["email-Enron-full", "email-Eu-full", "contact-high-school", "contact-primary-school", "NDC-classes-full", "NDC-substances-full", "tags-math-sx", "tags-ask-ubuntu", "threads-math-sx" , "threads-ask-ubuntu", "coauth-MAG-Geology-full"]

    df = pd.DataFrame(index=range(len(dataset)), columns=['data name', 'power_law-ll', 'power_law-pval','truncated_power_law-ll', 'truncated_power_law-pval', 'lognormal-ll', 'lognormal-pval'], dtype="float") 

    idx = 0
    for data_name in tqdm(dataset):
        dir_path = '../results/answer/' + data_name + '/'
        
        all_data = read_data(dir_path, data_name, fname)
        fit = powerlaw.Fit(all_data, verbose=False, discrete=True)
        
        for c in ['power_law', 'truncated_power_law', 'lognormal']:
            try:
                r, p = fit.distribution_compare(c, 'exponential', normalized_ratio=True)
                # positive if the data is more likely in the first distribution
                
                df['data name'][idx] = data_name
                df[c+'-ll'][idx] = r
                df[c+'-pval'][idx] = p
            except:
                continue
            
        idx += 1

    df.to_csv('../results/answer/' + 'powerlaw_test_' + test_type + '.csv', sep=',')
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ttype', '--test_type', type=str, required=True, help='Select the observation result to test: [degree, size, pairdeg, intersection, sv]')
    args = parser.parse_args()

    powerlaw_test(args.test_type)
    print("Done " + args.test_type)