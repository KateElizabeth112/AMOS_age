# explore the AMOS metadata
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

local = True

if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/vol/biomedic3/kc2322/data/AMOS_3D/"

input_folder = os.path.join(root_dir, "nnUNet_raw/Dataset200_AMOS")
splits_folder = os.path.join(root_dir, "splits")
meta_data_path = os.path.join(root_dir, "labeled_data_meta_0000_0599.csv")

def main():
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



if __name__ == "__main__":
    main()