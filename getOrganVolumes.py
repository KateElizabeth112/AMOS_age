# Get the volumes of the organs for the whole dataset and look for statistical differences based on characteristics
# in the metadata
import os
import nibabel as nib
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Polygon
import seaborn as sns

local = True

if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"

lblu = "#add9f4"
lred = "#f36860"

custom_palette = [lblu, lred]

gt_seg_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset200_AMOS", "labelsTr")
meta_data_path = os.path.join(root_dir, "labeled_data_meta_0000_0599.csv")

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


def calculate_volumes():
    # create containers to store the volumes
    volumes_f = []
    volumes_m = []

    # get a list of the files in the gt seg folder
    f_names = os.listdir(gt_seg_dir)

    # open the metadata
    meta = pd.read_csv(meta_data_path)
    ids_m = np.array(meta[meta["Patient's Sex"] == "M"]["amos_id"].values)
    ids_f = np.array(meta[meta["Patient's Sex"] == "F"]["amos_id"].values)

    for f in f_names:
        if f.endswith(".nii.gz"):
            # load image
            gt_nii = nib.load(os.path.join(gt_seg_dir, f))

            # get the volume of 1 voxel in mm3
            sx, sy, sz = gt_nii.header.get_zooms()
            volume = sx * sy * sz

            # find the number of voxels per organ in the ground truth image
            gt = gt_nii.get_fdata()
            volumes = []

            # cycle over each organ
            organs = list(labels.keys())

            for i in range(1, len(labels)):
                organ = organs[i]
                voxel_count = np.sum(gt == i)
                volumes.append(voxel_count * volume)

            # work out if the candidate is male or female
            subject = int(f[5:9])

            if subject in ids_f:
                print("F")
                volumes_f.append(np.array(volumes))
            elif subject in ids_m:
                print("M")
                volumes_m.append(np.array(volumes))
            else:
                print("Can't find subject in metadata list.")

    # Save the volumes ready for further processing
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "wb")
    pkl.dump([np.array(volumes_m), np.array(volumes_f)], f)
    f.close()


def plotVolumesHist():
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "rb")
    [volumes_m, volumes_f] = pkl.load(f)
    f.close()

    # For each organ, plot the volume distributions
    organs = list(labels.keys())

    for i in range(1, len(labels)):
        organ = organs[i]

        volumes_m_i = volumes_m[:, i-1]
        volumes_f_i = volumes_f[:, i-1]

        # First find the bins
        v_min_m = np.min(volumes_m_i)
        v_min_f = np.min(volumes_f_i)
        v_min = np.min((v_min_f, v_min_m))

        v_max_m = np.max(volumes_m_i)
        v_max_f = np.max(volumes_f_i)
        v_max = np.max((v_max_f, v_max_m))

        step = (v_max - v_min) / 20
        bins = np.arange(v_min, v_max + step, step)

        # Calculate averages to add to the plot
        v_av_men = np.mean(volumes_m_i)
        v_av_women = np.mean(volumes_f_i)

        plt.clf()
        plt.hist(volumes_m_i, color=lblu, alpha=0.6, label="Male", bins=bins)
        plt.axvline(x=v_av_men, color=lblu, label="Male average")
        plt.hist(volumes_f_i, color=lred, alpha=0.6, label="Female", bins=bins)
        plt.axvline(x=v_av_women, color=lred, label="Female average")
        plt.title(organ + " volume")
        plt.xlabel("Volume in voxels")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


def plotVolumesBoxAndWhiskers():
    # Build a list of data, labels and colors for plotting
    data = []
    box_labels = []
    box_colors = []

    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "rb")
    [volumes_m, volumes_f] = pkl.load(f)
    f.close()

    # For each organ, plot the volume distributions
    organs = list(labels.keys())

    for i in range(1, len(labels)):
        organ = organs[i]

        volumes_m_i = volumes_m[:, i-1]
        volumes_f_i = volumes_f[:, i-1]

        # Get overall maximum
        volumes_f_max = np.max(volumes_f_i)
        volumes_m_max = np.max(volumes_m_i)
        v_max = np.max((volumes_m_max, volumes_f_max))

        data.append((volumes_m_i / v_max))
        data.append((volumes_f_i / v_max))

        box_labels.append(organ)
        box_labels.append(organ)

        box_colors.append(lblu)
        box_colors.append(lred)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.2)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Dice scores for {}'.format(organ),
        xlabel='',
        ylabel='Dice Score',
    )

    num_boxes = len(data)
    medians = np.empty(num_boxes)

    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='k', marker='*', markeredgecolor='k',
                 markersize=10)

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.0
    bottom = 0.2
    # ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(box_labels, rotation=45, fontsize=8)

    # Finally, add a basic legend
    fig.text(0.80, 0.38, 'Male Test Set',
             backgroundcolor=box_colors[0], color='black', weight='roman',
             size='small')
    fig.text(0.80, 0.345, 'Female Test Set',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', size='small')
    fig.text(0.80, 0.295, '*', color='black',
             weight='roman', size='large')
    fig.text(0.815, 0.300, ' Average Value', color='black', weight='roman',
             size='small')

    plt.show()


def boxPlotSeaborn():
    # First create a dataframe from our results
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "rb")
    [volumes_m, volumes_f] = pkl.load(f)
    f.close()

    organs = list(labels.keys())

    organ_name = []
    normalised_volume = []
    gender = []

    for i in range(1, len(labels)-1):
        organ = organs[i]

        volumes_m_i = volumes_m[:, i-1]
        volumes_f_i = volumes_f[:, i-1]

        # Get overall maximum
        volumes_f_max = np.max(volumes_f_i)
        volumes_m_max = np.max(volumes_m_i)
        v_max = np.max((volumes_m_max, volumes_f_max))

        volumes_m_i_norm = (volumes_m_i / v_max)
        volumes_f_i_norm = (volumes_f_i / v_max)

        organ_name += [organ for _ in range(volumes_m_i.shape[0])]
        organ_name += [organ for _ in range(volumes_f_i.shape[0])]

        gender += ["M" for _ in range(volumes_m_i.shape[0])]
        gender += ["F" for _ in range(volumes_f_i.shape[0])]

        normalised_volume += list(volumes_m_i_norm)
        normalised_volume += list(volumes_f_i_norm)

    # Now build the data frame
    df = pd.DataFrame({'Normalised Volume': normalised_volume,
                       'Sex': gender,
                       'Organ Name': organ_name})
    sns.boxplot(y='Normalised Volume', x='Organ Name', data=df, hue='Sex', palette=custom_palette, showfliers=False)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.xlabel("")
    plt.show()


def significanceTesting():
    # perform Welch's t-test on the sample means
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "rb")
    [volumes_m, volumes_f] = pkl.load(f)
    f.close()

    # For each organ, plot the volume distributions
    organs = list(labels.keys())

    for i in range(1, len(labels)):
        organ = organs[i]

        # Calculate averages
        v_av_men = np.mean(volumes_m[:, i-1])
        v_av_women = np.mean(volumes_f[:, i-1])

        # difference in average (mm3)
        v_diff = v_av_men - v_av_women
        v_diff_prop = (v_diff / np.mean((v_av_women, v_av_men))) * 100

        # perform t-test
        res = stats.ttest_ind(volumes_m[:, i-1], volumes_f[:, i-1], equal_var=False)

        # save difference in mean, difference in mean as a proportion of the average volume, and p-value
        if res[1] < 0.01:
            sig = "**"
        elif res[1] < 0.05:
            sig = "*"
        else:
            sig = ""
        print("{0} & {1:.0f} {2} & {3:.2f} {4} ".format(organ, v_diff, sig, v_diff_prop, sig) + r"\\")



def plotOrganVolumeDistribution():
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "rb")
    [volumes_m, volumes_f] = pkl.load(f)
    f.close()

    volumes = np.vstack((volumes_m, volumes_f))

    # For each organ, plot the volume distributions. Ignore background
    organs = list(labels.keys())

    for i in range(1, len(labels)):
        organ = organs[i]
        volumes_i = volumes_m[:, i-1]

        # calculate median
        med = np.median(volumes_i) / 1000
        per_25 = np.percentile(volumes_i, 25) / 1000
        per_75 = np.percentile(volumes_i, 75) / 1000

        print("{0}: {1:.0f} ({2:.0f} - {3:.0f})".format(organ, med, per_25, per_75))


def main():
    #calculate_volumes()
    #plotVolumesBoxAndWhiskers()
    #boxPlotSeaborn()
    #significanceTesting()
    plotOrganVolumeDistribution()



if __name__ == "__main__":
    main()