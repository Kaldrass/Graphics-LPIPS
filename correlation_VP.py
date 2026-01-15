# Given a csv file containing the MOS
# Given a csv file containing the LPIPS values, we will compute the correlation between the two

import argparse
import os
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from rapidfuzz import fuzz
import re


def is_match_fuzz(name1, name2, threshold=90):
    """Verify if two names are similar."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    return n1 == n2 or fuzz.ratio(n1, n2) > threshold


def normalize_name(name):
    return name.lower().replace("_", "").strip()


def normalize_mos(mos_array):
    """Normalize MOS values from [1, 5] where 5 is best quality to [0, 1], where 0 is best quality."""
    return 1 - (mos_array - 1) / (5 - 1)

def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'\(.*?\)', '', name)     # remove parentheses content
    name = re.sub(r'_db$', '', name)
    name = re.sub(r'_kfolds$', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)   # keep only alphanumerics
    return name

# SECTION - GETTERS BEGIN
def get_MOS(MOSfile, distorted_obj_name, name_col, mos_col):
    mos = -1  # default value (golden ref ?)

    with open(MOSfile, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue
            name_candidate = row[name_col]
            mos_candidate = row[mos_col]

            if normalize_name(name_candidate) == normalize_name(distorted_obj_name):
                try:
                    mos = float(mos_candidate)
                    break
                except ValueError:
                    pass
    if mos == -1:
        print('[DEBUG] The object %s is not in the MOS file.' % distorted_obj_name)
    return mos


def get_test_MOS(test_list_csv, distorted_obj_name):  # For TMQ only
    mos = -1

    with open(test_list_csv, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 3:
                continue
            name_candidate = row[1]
            mos_candidate = row[2]

            if normalize_name(name_candidate) == normalize_name(distorted_obj_name):
                try:
                    mos = float(mos_candidate)
                    break
                except ValueError:
                    pass

    return mos


def get_testset_ref_list(test_list_csv):
    ref_list = []

    with open(test_list_csv, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 3:
                continue
            name_candidate = row[0]
            if name_candidate not in ref_list:
                ref_list.append(name_candidate)
    return ref_list


def get_testset_dis_list_from_ref(test_list_csv, ref_obj_name):
    dis_list = []
    with open(test_list_csv, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 3:
                continue
            dis_obj_name = row[1]
            if dis_obj_name.startswith(ref_obj_name):
                dis_list.append(dis_obj_name)
    return dis_list


# SECTION - GETTERS END


def calculate_correlation_all_vps_combined(base_dir, batchname, output_csv='global_combined_correlation.csv'):
    correlations = [("Object", "Pearson", "Spearman", "Slope", "CI_slope_lower", "CI_slope_upper", "Intercept", "R2")]

    def clamp01(a):
        a = np.asarray(a, dtype=float)
        np.clip(a, 0.0, 1.0, out=a)
        return a

    # Ensure output file is written inside base_dir if a relative path is given
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(base_dir, output_csv)

    for object_name in os.listdir(base_dir):
        object_dir = os.path.join(base_dir, object_name)
        csv_file = os.path.join(object_dir, 'GLPIPS_results_testset.csv')

        if not os.path.isfile(csv_file):
            continue

        with open(csv_file, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader)
            mos_list = []
            lpips_all_vps = []

            for row in reader:
                mos = float(row[1])
                lpips_vals = [float(x) for x in row[2:]]
                mos_list.append(mos)
                lpips_all_vps.append(lpips_vals)

        mos_array = np.array(mos_list)
        lpips_array = clamp01(np.array(lpips_all_vps))

        # MOS: from [1, 5] to [0, 1], where 0 is best quality
        mos_array = normalize_mos(mos_array)

        # Average LPIPS over all viewpoints
        avg_lpips = np.mean(lpips_array, axis=1)

        # Regression
        X = sm.add_constant(avg_lpips)
        model = sm.GLM(mos_array, X, family=sm.families.Binomial()).fit()
        predictions = model.predict(X)

        slope = model.params[1]
        intercept = model.params[0]
        pearson_corr = stats.pearsonr(predictions, mos_array)[0]
        spearman_corr = stats.spearmanr(predictions, mos_array)[0]
        ci = model.conf_int(alpha=0.05)

        correlations.append((
            object_name,
            round(pearson_corr, 4),
            round(spearman_corr, 4),
            round(slope, 4),
            round(ci[1, 0], 4),
            round(ci[1, 1], 4),
            round(intercept, 4),
        ))

    # Save per object correlations for this fold
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(correlations)

    print(f"\nCombined viewpoint correlations saved to: {output_csv}")

    # Global correlations over all objects (no plotting, only numeric results)
    all_mos = []
    all_lpips = []

    for object_name in os.listdir(base_dir):
        object_dir = os.path.join(base_dir, object_name)
        csv_file = os.path.join(object_dir, 'GLPIPS_results_testset.csv')

        if not os.path.isfile(csv_file):
            continue

        with open(csv_file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                mos = float(row[1])
                lpips_vals = [float(x) for x in row[2:] if float(x) != 0.0]
                avg_lpips = np.mean(lpips_vals)

                all_mos.append(mos)
                all_lpips.append(avg_lpips)

    all_mos = np.array(all_mos)
    all_mos = normalize_mos(all_mos)
    all_lpips = np.array(all_lpips)

    X = sm.add_constant(all_lpips)
    model = sm.GLM(all_mos, X, family=sm.families.Binomial()).fit()
    predictions = model.predict(X)

    pearson_corr = stats.pearsonr(predictions, all_mos)[0]
    spearman_corr = stats.spearmanr(predictions, all_mos)[0]

    # Print numeric summary for this fold / configuration
    print(f"Global correlations - Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

    # Previous per-fold stats file is removed to avoid one file per fold
    # global_stats_path = os.path.join(base_dir, "global_stats.csv")
    # with open(global_stats_path, mode='w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Pearson", "Spearman"])
    #     writer.writerow([round(pearson_corr, 4), round(spearman_corr, 4)])

    return pearson_corr, spearman_corr


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--use_gpu', action='store_true', default=True, help='turn on flag to use GPU')
    parser.add_argument('--version', type=str, default='0.1')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--use_folds', action='store_true')
    parser.add_argument('-v', '--views', type=int, required=True)
    parser.add_argument('-vm', '--view_method', type=str, required=True)
    parser.add_argument('-rm', '--render_method', type=str, required=True)
    parser.add_argument('-db', '--database', type=str, required=True)

    opt = parser.parse_args()

    model = opt.model
    modelpath = './checkpoints/' + model + '/latest_net_.pth'
    use_folds = opt.use_folds
    testing_views = opt.views
    view_method = opt.view_method
    render_method = opt.render_method
    database = opt.database

    batchname = f"{model}_{database}_{render_method}_{view_method}_{testing_views}VP"

    # Base experiment directory (without fold and without _METRIC_RESULTS_TESTSET_)
    experiment_dir = os.path.join(
        "D:/These/Graphics-LPIPS/out/",
        database,
        render_method,
        view_method,
        model,
        f"{testing_views}VP"
    )

    if use_folds:
        pcors = []
        scores_pearson = []
        scores_spearman = []

        for fold_idx in range(5):
            base_dir = os.path.join(
                experiment_dir,
                f"fold_k{fold_idx}",
                "_METRIC_RESULTS_TESTSET_"
            )
            fold_batchname = batchname + '_fold' + str(fold_idx)
            p_corr, s_corr = calculate_correlation_all_vps_combined(base_dir, fold_batchname)
            pcors.append(p_corr)
            scores_pearson.append(p_corr)
            scores_spearman.append(s_corr)

        pcorr_mean = float(np.mean(scores_pearson))
        scorr_mean = float(np.mean(scores_spearman))

        # Single file gathering all folds correlations
        folds_stats_path = os.path.join(experiment_dir, "correlation_folds_stats.csv")
        with open(folds_stats_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "pearson", "spearman"])
            for idx, (p, s) in enumerate(zip(scores_pearson, scores_spearman)):
                writer.writerow([idx, round(p, 4), round(s, 4)])
            writer.writerow(["mean", round(pcorr_mean, 4), round(scorr_mean, 4)])

        print(f"Fold correlations summary saved to: {folds_stats_path}")

        pcorr = pcorr_mean
    else:
        base_dir = os.path.join(
            experiment_dir,
            "_METRIC_RESULTS_TESTSET_"
        )
        p_corr, s_corr = calculate_correlation_all_vps_combined(base_dir, batchname)
        pcorr = p_corr
        scorr = s_corr

        # For non-fold case, still write a consistent file with a single fold
        folds_stats_path = os.path.join(experiment_dir, "correlation_folds_stats.csv")
        with open(folds_stats_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "pearson", "spearman"])
            writer.writerow([0, round(p_corr, 4), round(s_corr, 4)])
            writer.writerow(["mean", round(p_corr, 4), round(s_corr, 4)])

        print(f"Single configuration correlations saved to: {folds_stats_path}")

    print("pearson mean : {:.3f}".format(pcorr))

    # Save experiment level summary (including mean over folds) in the experiment directory
    summary_path = os.path.join(experiment_dir, "correlation_summary_kfolds.csv")
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "batchname", "model", "database",
                "render_method", "view_method",
                "testing_views", "n_folds", "pearson_mean"
            ])
        n_folds = 5 if use_folds else 1
        writer.writerow([
            batchname, model, database,
            render_method, view_method,
            testing_views, n_folds, round(pcorr, 4)
        ])

    print(f"Experiment summary appended to: {summary_path}")


if __name__ == "__main__":
    main()
