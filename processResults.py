import numpy as np
import nibabel as nib
import os
import pickle as pkl

import pandas as pd
from monai.metrics import compute_hausdorff_distance
import argparse

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="Dataset303_Set3", help="Task to evaluate")
args = vars(parser.parse_args())

# set up variables
task = args["dataset"]

local = False
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"

fold = "all"

preds_dir = os.path.join(root_dir, "inference", task, fold)
gt_dir = os.path.join(root_dir, "nnUNet_raw", task, "labelsTs")

labels = {"background": 0,
          "spleen": 1,
          "right kidney": 2,
          "left kidney": 3,
          "gallbladder": 4,
          "esophagus": 5,
          "liver": 6,
          "stomach": 7,
          "aorta": 8,
          "inferior vena cava": 9,
          "pancreas": 10,
          "right adrenal gland": 11,
          "left adrenal gland": 12,
          "duodenum": 13,
          "bladder": 14,
          "prostate/uterus": 15}

n_channels = int(len(labels))


def getVolume(pred, gt, vox_vol):
    # Get the organ volumes given the ground truth mask and the validation mask
    vol_preds = []
    vol_gts = []
    for channel in range(n_channels):
        vol_preds.append(np.sum(pred[pred == channel]) * vox_vol)
        vol_gts.append(np.sum(gt[gt == channel]) * vox_vol)

    return np.array(vol_preds), np.array(vol_gts)


def oneHotEncode(array):
    one_hot = np.zeros((n_channels, array.shape[0], array.shape[1], array.shape[2]))

    for i in range(0, n_channels):
        one_hot[i, :, :, :][array == i] = 1

    return one_hot


def computeHDDIstance(pred, gt, vox_spacing):
    # To use the MONAI function pred must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
    # The values should be binarized.
    # gt: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
    # The values should be binarized.

    # Convert to one hot
    # covert predictions to one hot encoding
    pred_one_hot = oneHotEncode(pred)
    gt_one_hot = oneHotEncode(gt)

    # expand the number of dimensions to include batch as the first dimension (this is required by the MONAI function)
    pred_one_hot = np.expand_dims(pred_one_hot, axis=0)
    gt_one_hot = np.expand_dims(gt_one_hot, axis=0)

    # DEBUG: check dimensions. we should have 5
    print("Pred before HD: {}".format(pred_one_hot.shape))

    hd = compute_hausdorff_distance(pred_one_hot, gt_one_hot, include_background=False, distance_metric='euclidean', percentile=None,
                               directed=False, spacing=vox_spacing)

    return hd


def multiChannelDice(pred, gt, channels):

    dice = []

    for channel in range(channels):
        a = np.zeros(pred.shape)
        a[pred == channel] = 1

        b = np.zeros(gt.shape)
        b[gt == channel] = 1

        dice.append(np.sum(a[b == 1])*2.0 / (np.sum(a) + np.sum(a)))

    return np.array(dice)


def calculateMetrics():
    # get a list of male and female IDs
    f = open(os.path.join(root_dir, "info.pkl"), "rb")
    info = pkl.load(f)
    f.close()

    ids_all = info["id"]
    sex_all = info["sex"]  # male = 0, female = 1
    age_all = info["age"]

    # containers to store results
    case_id = []
    sex = []
    dice_scores = []
    age = []
    hausdorff = []
    vol_preds = []
    vol_gts = []

    ids_pred = os.listdir(preds_dir)

    for case in ids_pred:
        if case.endswith(".nii.gz"):
            id = case[5:9]
            print("Processing {}".format(id))

            pred_nii = nib.load(os.path.join(preds_dir, case))
            gt_nii = nib.load(os.path.join(gt_dir, case))

            # get the volume of 1 voxel in mm3
            sx, sy, sz = gt_nii.header.get_zooms()
            vox_vol = sx * sy * sz
            vox_spacing = np.array([sx, sy, sz])

            print(vox_spacing)
            print(type(vox_spacing))
            vox_spacing = vox_spacing.astype('float')

            pred = pred_nii.get_fdata()
            gt = gt_nii.get_fdata()

            if np.unique(gt).sum() == 0:
                print("Only background")

            # Get Dice and NSD and volumes
            dice = multiChannelDice(pred, gt, n_channels)
            hd = computeHDDIstance(pred, gt, vox_spacing)

            # DEBUG: shape of HD values
            print("Shape of HD array: {}".format(hd.numpy().shape))
            print("Shape of Dice array: {}".format(dice.shape))

            vol_pred, vol_gt = getVolume(pred, gt, vox_vol)

            if id in ids_all:
                case_id.append(id)
                sex.append(sex_all[ids_all == id])
                age.append(age_all[ids_all == id])
                dice_scores.append(dice)
                hausdorff.append(hd.numpy().squeeze())
                vol_preds.append(vol_pred)
                vol_gts.append(vol_gt)
            else:
                print("Not in list")

    # convert to numpy arrays (except for HD distance)
    case_id = np.array(case_id)
    dice_scores = np.array(dice_scores)
    hausdorff = np.array(hausdorff)
    sex = np.array(sex)
    age = np.array(age)
    vol_preds = np.array(vol_preds)
    vol_gts = np.array(vol_gts)

    print("Number of men: {}".format(sex.shape[0] - np.sum(sex)))
    print("Number of women: {}".format(np.sum(sex)))

    f = open(os.path.join(preds_dir, "results.pkl"), "wb")
    pkl.dump({"case_id": case_id,
              "sex": sex,
              "age": age,
              "dice": dice_scores,
              "hd": hausdorff,
              "vol_pred": vol_preds,
              "vol_gt": vol_gts,
              }, f)
    f.close()


def main():
    calculateMetrics()




if __name__ == "__main__":
    main()