# explore the AMOS metadata
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl

local = False

if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"

input_folder = os.path.join(root_dir, "nnUNet_raw/Dataset200_AMOS")
splits_folder = os.path.join(root_dir, "splits")
meta_data_path = os.path.join(root_dir, "labeled_data_meta_0000_0599.csv")


def createDatasetInfo():
    df = pd.read_csv(meta_data_path)

    ids = df["amos_id"].values
    sex_mf = df["Patient's Sex"].values
    age_str = df["Patient's Age"].values
    scanner = df["Manufacturer's Model Name"].values
    site = df["Site"].values
    age = []
    ids_str = []

    # reformat age as integers
    for a in list(age_str):
        if type(a) == float and np.isnan(a):
            age.append(np.nan)
        else:
            age.append(int(a[1:3]))

    # change sex to a binary value (0=M, 1=F)
    sex = np.zeros(sex_mf.shape)
    sex[sex_mf == "F"] = 1

    # reformat ids as strings
    for id in list(ids):
        id_str = str(id)
        if len(id_str) == 1:
            ids_str.append("000" + id_str)
        elif len(id_str) == 2:
            ids_str.append("00" + id_str)
        elif len(id_str) == 3:
            ids_str.append("0" + id_str)

    info = {"id": np.array(ids_str),
            "age": np.array(age),
            "sex": np.array(sex),
            "scanner": scanner,
            "site": site}


    f = open(os.path.join(root_dir, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


def plot():
    df = pd.read_csv(meta_data_path)

    ids = df["amos_id"].values
    sex = df["Patient's Sex"].values
    age_str = df["Patient's Age"].values
    age = []

    # reformat age as integers
    for a in list(age_str):
        if type(a) == float and np.isnan(a):
            age.append(np.nan)
        else:
            age.append(int(a[1:3]))

    age = np.array(age)
    age_m = age[sex == "M"]
    ids_m = ids[sex == "M"]
    age_f = age[sex == "F"]
    ids_f = ids[sex == "F"]

    # age groups 20-40 and 60-80 (or under 40 and over 65?)
    print("Number of males under 40: {}".format(ids_m[age_m <= 40].shape[0]))
    print("Number of females under 40: {}".format(ids_f[age_f <= 40].shape[0]))

    print("Number of males over 65: {}".format(ids_m[age_m >= 65].shape[0]))
    print("Number of females over 65: {}".format(ids_f[age_f >= 65].shape[0]))

    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.clf()
    plt.hist(age_m, label="Male", alpha=0.6, bins=bins)
    plt.hist(age_f, label="Female", alpha=0.6, bins=bins)
    plt.legend()
    plt.show()


def main():
    createDatasetInfo()


if __name__ == "__main__":
    main()