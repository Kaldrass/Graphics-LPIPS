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
import config

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

#SECTION - GETTERS BEGIN
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

    return mos
def get_test_MOS(test_list_csv, distorted_obj_name): # For TMQ only
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
#SECTION - GETTERS END 




def calculate_correlation_all_vps_combined(base_dir, batchname, output_csv='global_combined_correlation.csv'):
    correlations = [("Object", "Pearson", "Spearman", "Slope", "CI_slope_lower", "CI_slope_upper", "Intercept", "R²")]
    def clamp01(a):
        a = np.asarray(a, dtype=float)
        np.clip(a, 0.0, 1.0, out=a)
        return a
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
                lpips_vals = [float(x) for x in row[2:]]  # Exclude zero values
                mos_list.append(mos)
                lpips_all_vps.append(lpips_vals)
        mos_array = np.array(mos_list)
        lpips_array = clamp01(np.array(lpips_all_vps))
         

        #  MOS : from [1, 5] to [0, 1], where 0 = best quality
        mos_array = normalize_mos(mos_array) # Uncomment if MOS is in [1, 5] range, otherwise comment this line

        # Moyenne LPIPS sur toutes les vues
        avg_lpips = np.mean(lpips_array, axis=1)

        # Régression
        X = sm.add_constant(avg_lpips)
        
        model = sm.GLM(mos_array, X, family = sm.families.Binomial()).fit()
        predictions = model.predict(X)

        slope = model.params[1]
        intercept = model.params[0]
        # r_squared = model.rsquared
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
            # round(r_squared, 4)
        ))

    # Sauvegarde des résultats
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(correlations)

    print(f"\nCombined viewpoint correlations saved to: {output_csv}")

    # --- Graphique global (si on veut regrouper tous les objets en un seul nuage de points) ---
    # Moyenne LPIPS et MOS pour tout le dataset
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
                nvp = len(row[2:]) 
                mos = float(row[1])
                lpips_vals = [float(x) for x in row[2:] if float(x) != 0.0]  # Exclude zero values
                avg_lpips = np.mean(lpips_vals)

                all_mos.append(mos)
                all_lpips.append(avg_lpips)
    all_mos = np.array(all_mos)
    all_mos = normalize_mos(all_mos) # Uncomment if MOS is in [1, 5] range, otherwise comment this line
    
    all_lpips = np.array(all_lpips)

    X = sm.add_constant(all_lpips)
    model = sm.GLM(all_mos, X, family = sm.families.Binomial()).fit()
    predictions = model.predict(X)

    pearson_corr = stats.pearsonr(predictions, all_mos)[0]
    spearman_corr = stats.spearmanr(predictions, all_mos)[0]

    sorted_indices = np.argsort(X[:,1])
    x_sorted = X[sorted_indices,1]
    y_sorted = predictions[sorted_indices]
    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(all_lpips, all_mos, label="Data", color='blue')
    plt.plot(x_sorted, y_sorted, color='red', label='Logistic Fit')
    plt.title(f'Global Correlation ({nvp} Viewpoints)\nPearson={pearson_corr:.3f} | Spearman={spearman_corr:.3f}')
    plt.xlabel('GLPIPS')
    plt.ylabel('MOS (normalized)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_plot = os.path.join(base_dir, batchname + '_global_cc.png')
    plt.savefig(output_plot)
    plt.show()

    print(f"Global correlation plot saved to: {output_plot}")
    # --- Barplot des corrélations par objet ---
    object_names = []
    pearson_values = []
    
    for row in correlations[1:]:  # skip header
        object_names.append(row[0])
        pearson_values.append(row[1])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(object_names, pearson_values, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Correlation per Object (All Viewpoints Combined)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Optionnel : annotation des valeurs
    for bar, value in zip(bars, pearson_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    barplot = os.path.join(base_dir, batchname + '_barplot.png')
    plt.savefig(barplot)
    plt.show()
    
    print(f"Barplot saved to: {barplot}")


def main():
    # Path to the base directory containing all objects
    model = config.model#'TMQ_Gautier_NR_1VPn_fib'
    print('model:', model)
    testing_views = config.testing_views#1
    view_method = config.view_method#'Fibonacci' # 'Fibonacci', 'Y_fixed_0.3' or 'Polyhedron'
    render_method = config.render_method#'New_Render' # 'New_Render' or 'Old_render'
    database = config.database#'TMQ' # 'TSMD' or 'BASICS(PC)_DB' or 'TMQ'
    batchname = database + '_' + render_method + '_' + view_method + '_' + model + '_' + str(testing_views) + 'VP'

    base_dir = "D:/These/Graphics-LPIPS/out/" + database + "/" + render_method + "/" + view_method + "/" + model + "/" + str(testing_views) + "VP/_METRIC_RESULTS_TESTSET_/"
    # process_all_objects(base_dir)
    calculate_correlation_all_vps_combined(base_dir, batchname)
    # plot_global_correlations_per_viewpoint(base_dir, 1)
if __name__ == "__main__":
    main()