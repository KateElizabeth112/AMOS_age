# A script to create train/test splits from the amos dataset with respect to age
# group 1 = 40 and under
# group 2 = 65 and over
import pandas as pd
import numpy as np
import os
import pickle as pkl
import shutil

local = False

if local:
    root_folder = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_folder = "/rds/general/user/kc2322/home/data/AMOS_3D"

input_folder = os.path.join(root_folder, "nnUNet_raw/Dataset200_AMOS")
output_folder = os.path.join(root_folder, "nnUNet_raw")
input_images_folder = os.path.join(input_folder, "imagesTr")
input_labels_folder = os.path.join(input_folder, "labelsTr")
splits_folder = os.path.join(root_folder, "splits")
meta_data_path = os.path.join(root_folder, "labeled_data_meta_0000_0599.csv")


def saveDatasetInfo():
    # open metadata
    df = pd.read_csv(meta_data_path)

    # List the images in the training dataset
    images = os.listdir(input_images_folder)

    amos_id = df["amos_id"].values
    sex_mf = df["Patient's Sex"].values
    age_str = df["Patient's Age"].values
    age = []
    patients = []
    genders = []
    patients_tr = []

    # Change the patient ID to be a 4-character string with zero padding where necessary
    for id in amos_id:
        id = str(id)
        if len(id) == 1:
            id = "000" + id
        elif len(id) == 2:
            id = "00" + id
        elif len(id) == 3:
            id = "0" + id

        patients.append(id)

    patients = np.array(patients)

    for image in images:
        if image.endswith(".nii.gz"):
            # Find the gender of the subject from the metadata
            id = image[5:9]
            if id in patients:
                x = np.where(patients == id)[0][0]
                patients_tr.append(id)
                if sex_mf[x] == "M":
                    genders.append(0)
                elif sex_mf[x] == "F":
                    genders.append(1)

                # convert age from string to float
                a = age_str[x]
                if type(a) == float and np.isnan(a):
                    age.append(np.nan)
                else:
                    age.append(int(a[1:3]))

    genders = np.array(genders)
    patients_tr = np.array(patients_tr)
    age = np.array(age)

    # Save lists
    info = {"patients": patients_tr,
            "genders": genders,
            "age": age}

    f = open(os.path.join(input_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


def generate_folds():
    f = open(os.path.join(input_folder, "info.pkl"), "rb")
    info = pkl.load(f)
    f.close()

    patients = info["patients"]
    age = info["age"]

    # split into group 1 and group 2
    ids_g1 = patients[age <= 40]
    ids_g2 = patients[age >= 65]

    # randomly shuffle indices
    np.random.shuffle(ids_g1)
    np.random.shuffle(ids_g2)

    # Find the block size
    block_size = np.floor(np.min([ids_g1.shape[0], ids_g2.shape[0]]) / 9)
    dataset_size = int(block_size * 8)

    print("Dataset size: {}".format(dataset_size))
    print("Test set size per fold: {}".format(block_size * 2))

    # create 9 training blocks overall (these will form 5 folds)
    blocks_g1 = []
    blocks_g2 = []

    for i in range(9):
        blocks_g1.append(ids_g1[int(i * block_size):int((i + 1) * block_size)])
        blocks_g2.append(ids_g2[int(i * block_size):int((i + 1) * block_size)])

    # create 5 training folds for three datasets
    ts = np.concatenate((blocks_g1[0], blocks_g2[0]), axis=0)
    tr1_g1 = np.concatenate(blocks_g1[1:5], axis=0)
    tr1_g2 = np.concatenate(blocks_g2[1:5], axis=0)
    tr1 = np.concatenate((tr1_g1, tr1_g2), axis=0)

    tr2 = np.concatenate(blocks_g1[1:9], axis=0)
    tr3 = np.concatenate(blocks_g2[1:9], axis=0)

    set_1_ids = {"train": tr1, "test": ts}
    set_2_ids = {"train": tr2, "test": ts}
    set_3_ids = {"train": tr3, "test": ts}

    f = open(os.path.join(splits_folder, "fold_0_age.pkl"), "wb")
    pkl.dump([set_1_ids, set_2_ids, set_3_ids], f)
    f.close()

    print(tr1.shape, tr2.shape, tr3.shape, ts.shape)

    for f in range(1, 4):
        ts = np.concatenate((blocks_g1[f], blocks_g2[f]), axis=0)
        tr1_g1 = np.concatenate((blocks_g1[0:f] + blocks_g1[f+1:5]), axis=0)
        tr1_g2 = np.concatenate((blocks_g2[0:f] + blocks_g2[f+1:5]), axis=0)
        tr1 = np.concatenate((tr1_g1, tr1_g2), axis=0)

        tr2 = np.concatenate((blocks_g1[0:f] + blocks_g1[f+1:9]), axis=0)
        tr3 = np.concatenate((blocks_g2[0:f] + blocks_g2[f+1:9]), axis=0)

        set_1_ids = {"train": tr1, "test": ts}
        set_2_ids = {"train": tr2, "test": ts}
        set_3_ids = {"train": tr3, "test": ts}

        f = open(os.path.join(splits_folder, "fold_{}_age.pkl".format(f)), "wb")
        pkl.dump([set_1_ids, set_2_ids, set_3_ids], f)
        f.close()

        print(tr1.shape, tr2.shape, tr3.shape, ts.shape)

    ts = np.concatenate((blocks_g1[4], blocks_g2[4]), axis=0)
    tr1_g1 = np.concatenate(blocks_g1[:4], axis=0)
    tr1_g2 = np.concatenate(blocks_g2[:4], axis=0)
    tr1 = np.concatenate((tr1_g1, tr1_g2), axis=0)

    tr2 = np.concatenate((blocks_g1[0:4] + blocks_g1[5:9]), axis=0)
    tr3 = np.concatenate((blocks_g2[0:4] + blocks_g2[5:9]), axis=0)

    set_1_ids = {"train": tr1, "test": ts}
    set_2_ids = {"train": tr2, "test": ts}
    set_3_ids = {"train": tr3, "test": ts}

    f = open(os.path.join(splits_folder, "fold_4_age.pkl"), "wb")
    pkl.dump([set_1_ids, set_2_ids, set_3_ids], f)
    f.close()


def copy_images(dataset_name, ids_tr, ids_ts):
    os.mkdir(os.path.join(output_folder, dataset_name))

    output_imagesTr = os.path.join(output_folder, dataset_name, "imagesTr")
    output_labelsTr = os.path.join(output_folder, dataset_name, "labelsTr")
    output_imagesTs = os.path.join(output_folder, dataset_name, "imagesTs")
    output_labelsTs = os.path.join(output_folder, dataset_name, "labelsTs")

    os.mkdir(output_imagesTr)
    os.mkdir(output_labelsTr)
    os.mkdir(output_imagesTs)
    os.mkdir(output_labelsTs)

    # copy over the files from Training Set
    for case in list(ids_tr):
        print("Case {}".format(case))
        img_name = "amos_" + case + "_0000.nii.gz"
        lab_name = "amos_" + case + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTr, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTr, lab_name))

    # copy over the files from Test Set
    for case in list(ids_ts):
        img_name = "amos_" + case + "_0000.nii.gz"
        lab_name = "amos_" + case + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTs, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTs, lab_name))


def sort():
    # Sort the case IDs according to the sets
    folds = [0, 1, 2, 3, 4]

    for fold in folds:
        f = open(os.path.join(splits_folder, "fold_{}_age.pkl".format(fold)), "rb")
        ids = pkl.load(f)
        f.close()

        for j in range(3):
            ids_tr = ids[j]["train"]
            ids_ts = ids[j]["test"]

            name = "Dataset{}0{}".format(5+fold, j) + "_Age{}".format(fold)

            print("Working on Set {}....".format(name))
            copy_images(name, ids_tr, ids_ts)


def main():
    #saveDatasetInfo()
    #generate_folds()

    sort()


if __name__ == "__main__":
    main()