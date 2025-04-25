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


def normalize_name(name):
    """Nettoie un nom pour comparaison plus tolérante."""
    return name.lower().replace("_", "").strip()

def is_match_fuzz(name1, name2, threshold=90):
    """Vérifie si deux noms sont assez similaires."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    return n1 == n2 or fuzz.ratio(n1, n2) > threshold

def normalize_name(name):
    return name.lower().replace("_", "").strip()

def get_MOS(MOSfile, distorted_obj_name):
    mos = -1  # Valeur par défaut si on ne trouve rien (golden ref ?)

    with open(MOSfile, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue  # ligne vide ou incomplète
            name_candidate = row[0]
            mos_candidate = row[1]

            if normalize_name(name_candidate) == normalize_name(distorted_obj_name):
                try:
                    mos = float(mos_candidate)
                    break
                except ValueError:
                    pass  # colonne MOS non-numérique (ex: "NaN"), on ignore

    return mos

def calculate_correlation(GLPIPS_results_file):
    if not os.path.exists(GLPIPS_results_file):
        print('GLPIPS results file does not exist')
        return -1, -1

    with open(GLPIPS_results_file, mode='r') as GLPIPSfile:
        print('GLPIPS results file:', GLPIPS_results_file)
        reader = csv.reader(GLPIPSfile)
        next(reader)  # Skip header

        List_MOS = []
        List_GraphicsLPIPS = []

        for row in reader:
            distorted_obj_name = row[0]
            mos = float(row[1])
            List_MOS.append(mos)
            GLPIPS_values = [float(x) for x in row[2:]]
            List_GraphicsLPIPS.append(GLPIPS_values)

    List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
    List_MOS = np.array(List_MOS)

    # Normalize MOS values: from [1, 5] to [0, 1], with 0 being the best quality
    List_MOS = 1 - (List_MOS - 1) / (5 - 1)

    # Use LPIPS values from vp1 (first view point)
    vp1LPIPS = List_GraphicsLPIPS[:, 0]

    # Perform linear regression: MOS ~ LPIPS
    X = sm.add_constant(vp1LPIPS)  # Add intercept term
    model = sm.OLS(List_MOS, X).fit()
    predictions = model.predict(X)

    # Print regression summary
    print(model.summary())

    # Compute correlation coefficients between predicted and true MOS
    corrPears = stats.pearsonr(predictions, List_MOS)[0]
    corrSpear = stats.spearmanr(predictions, List_MOS)[0]
    print('pearson %.3f' % corrPears)
    print('spearman %.3f' % corrSpear)

    # Plot data and regression line
    # plt.figure(figsize=(8, 5))
    # plt.scatter(vp1LPIPS, List_MOS, color='blue', label='Data')
    # plt.plot(vp1LPIPS, predictions, color='red', linewidth=2, label='Linear regression')
    # plt.xlabel('LPIPS (vp1)')
    # plt.ylabel('MOS (normalized, 0 = best)')
    # plt.title('Linear Regression between LPIPS (vp1) and MOS')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return corrPears, corrSpear


# Function to calculate linear regression for all viewpoints
def calculate_linear_regression_all_vps(GLPIPS_results_file, output_csv='correlations_summary.csv', output_plot='regression_subplots.png'):
    if not os.path.exists(GLPIPS_results_file):
        print('GLPIPS results file does not exist')
        return

    with open(GLPIPS_results_file, mode='r') as GLPIPSfile:
        print('GLPIPS results file:', GLPIPS_results_file)
        reader = csv.reader(GLPIPSfile)
        header = next(reader)

        List_MOS = []
        List_GraphicsLPIPS = []

        for row in reader:
            mos = float(row[1])
            List_MOS.append(mos)
            GLPIPS_values = [float(x) for x in row[2:]]
            List_GraphicsLPIPS.append(GLPIPS_values)

    List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
    List_MOS = np.array(List_MOS)

    # Normalize MOS: from [1, 5] to [0, 1], where 0 is best quality
    List_MOS = 1 - (List_MOS - 1) / (5 - 1)

    num_vps = List_GraphicsLPIPS.shape[1]

    correlation_data = [(
        "Viewpoint", "Pearson", "Spearman",
        "Slope", "CI_slope_lower", "CI_slope_upper",
        "Intercept", "CI_intercept_lower", "CI_intercept_upper",
        "R2"
    )]
    
    best_vp_idx = -1
    best_pearson = -np.inf

    # Prepare subplots layout
    cols = 3
    rows = math.ceil(num_vps / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axs = axs.flatten()

    for vp_idx in range(num_vps):
        vp_lpips = List_GraphicsLPIPS[:, vp_idx]

        X = sm.add_constant(vp_lpips)
        model = sm.OLS(List_MOS, X).fit()
        predictions = model.predict(X)

        slope = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared
        pearson_corr = stats.pearsonr(predictions, List_MOS)[0]
        spearman_corr = stats.spearmanr(predictions, List_MOS)[0]

        # Save best viewpoint
        if pearson_corr > best_pearson:
            best_pearson = pearson_corr
            best_vp_idx = vp_idx

        # Confidence intervals
        conf = model.conf_int(alpha=0.05)  # 95% CI
        ci_slope_low, ci_slope_high = conf[1]
        ci_intercept_low, ci_intercept_high = conf[0]

        correlation_data.append(( 
            f"vp{vp_idx + 1}",
            round(pearson_corr, 4),
            round(spearman_corr, 4),
            round(slope, 4),
            round(ci_slope_low, 4),
            round(ci_slope_high, 4),
            round(intercept, 4),
            round(ci_intercept_low, 4),
            round(ci_intercept_high, 4),
            round(r_squared, 4)
        ))

        # Plot
        ax = axs[vp_idx]
        ax.scatter(vp_lpips, List_MOS, color='blue', label='Data')
        ax.plot(vp_lpips, predictions, color='red', label='Linear Fit')
        ax.set_title(f'vp{vp_idx + 1} | Pearson={pearson_corr:.2f}')
        ax.set_xlabel('LPIPS')
        ax.set_ylabel('MOS (normalized)')
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(num_vps, len(axs)):
        fig.delaxes(axs[i])

    fig.suptitle('Linear Regression for All Viewpoints', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_plot)
    # plt.show()

    # Export correlation summary
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(correlation_data)

    print(f"\nCorrelation summary saved to: {output_csv}")
    print(f"Regression plots saved to: {output_plot}")
    print(f"Best viewpoint: vp{best_vp_idx + 1} with Pearson = {best_pearson:.3f}")


# Function to process all objects
def process_all_objects(base_dir):
    object_folders = [name for name in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, name))]
    
    # Prepare a list to collect all regression summaries
    all_objects_summary = []

    for obj_name in object_folders:
        GLPIPS_file = os.path.join(base_dir, obj_name, "GLPIPS_results.csv")
        # Replace "\\" with "/" for cross-platform compatibility
        GLPIPS_file = GLPIPS_file.replace("\\", "/")
        if os.path.exists(GLPIPS_file):
            print(f"Processing {obj_name}...")
            output_csv = os.path.join(base_dir, obj_name, f"{obj_name}_correlations_summary.csv")
            output_plot = os.path.join(base_dir, obj_name, f"{obj_name}_regression_subplots.png")
            output_csv = output_csv.replace("\\", "/")
            output_plot = output_plot.replace("\\", "/")
            calculate_linear_regression_all_vps(GLPIPS_file, output_csv, output_plot)

            # Add object regression summary to the list
            with open(output_csv, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    all_objects_summary.append({
                        "ObjectName": obj_name,
                        "Viewpoint": row[0],
                        "Pearson": row[1],
                        "Spearman": row[2],
                        "Slope": row[3],
                        "CI_slope_lower": row[4],
                        "CI_slope_upper": row[5],
                        "Intercept": row[6],
                        "CI_intercept_lower": row[7],
                        "CI_intercept_upper": row[8],
                        "R2": row[9]
                    })

    # Combine all results into a single DataFrame
    if all_objects_summary:
        combined_df = pd.DataFrame(all_objects_summary)
        # Save the results to a CSV file
        combined_csv_path = os.path.join(base_dir, "all_objects_regression_summary.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined regression summary to {combined_csv_path}")
    else:
        print("No data to process.")
    # After processing all objects
    calculate_correlation_per_viewpoint_across_objects(base_dir)
    

def calculate_correlation_per_viewpoint_across_objects(base_dir, output_csv='global_viewpoint_correlations.csv'):
    object_folders = [name for name in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, name))]

    vp_lpips_data = {}
    vp_mos_data = {}

    for obj_name in object_folders:
        GLPIPS_file = os.path.join(base_dir, obj_name, "GLPIPS_results.csv")
        if not os.path.exists(GLPIPS_file):
            continue

        with open(GLPIPS_file, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                mos = float(row[1])
                lpips_values = [float(x) for x in row[2:]]

                for vp_idx, lpips in enumerate(lpips_values):
                    if vp_idx not in vp_lpips_data:
                        vp_lpips_data[vp_idx] = []
                        vp_mos_data[vp_idx] = []
                    vp_lpips_data[vp_idx].append(lpips)
                    # Normalize MOS to [0, 1]
                    vp_mos_data[vp_idx].append(1 - (mos - 1) / 4)

    summary = [("Viewpoint", "Pearson", "Spearman")]

    for vp_idx in sorted(vp_lpips_data.keys()):
        lpips_values = np.array(vp_lpips_data[vp_idx])
        mos_values = np.array(vp_mos_data[vp_idx])

        pearson_corr = stats.pearsonr(lpips_values, mos_values)[0]
        spearman_corr = stats.spearmanr(lpips_values, mos_values)[0]

        summary.append((
            f"vp{vp_idx + 1}",
            round(pearson_corr, 4),
            round(spearman_corr, 4)
        ))

    # Save the summary CSV
    output_path = os.path.join(base_dir, output_csv)
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(summary)

    print(f"\nGlobal correlation summary saved to: {output_path}")

def plot_global_correlations_per_viewpoint(correlations_csv, output_image='global_viewpoint_correlations.png'):
    vp_labels = []
    pearson_vals = []
    spearman_vals = []

    with open(correlations_csv, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            vp_labels.append(row[0])
            pearson_vals.append(float(row[1]))
            spearman_vals.append(float(row[2]))

    x = np.arange(len(vp_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, pearson_vals, width, label='Pearson')
    rects2 = ax.bar(x + width/2, spearman_vals, width, label='Spearman')

    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Global Correlation by Viewpoint (All Objects)')
    ax.set_xticks(x)
    ax.set_xticklabels(vp_labels, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()
    print(f"Correlation bar plot saved to: {output_image}")


def calculate_correlation_all_vps_combined(base_dir, output_csv='global_combined_correlation.csv', output_plot='global_combined_regression.png'):
    correlations = [("Object", "Pearson", "Spearman", "Slope", "CI_slope_lower", "CI_slope_upper", "Intercept", "R²")]

    for object_name in os.listdir(base_dir):
        object_dir = os.path.join(base_dir, object_name)
        csv_file = os.path.join(object_dir, 'GLPIPS_results.csv')

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
        lpips_array = np.array(lpips_all_vps)

        # Normalisation MOS : de [1, 5] vers [0, 1], où 0 = meilleure qualité
        mos_array = 1 - (mos_array - 1) / (5 - 1)

        # Moyenne LPIPS sur toutes les vues
        avg_lpips = np.mean(lpips_array, axis=1)

        # Régression
        X = sm.add_constant(avg_lpips)
        model = sm.OLS(mos_array, X).fit()
        predictions = model.predict(X)

        slope = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared
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
            round(r_squared, 4)
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
        csv_file = os.path.join(object_dir, 'GLPIPS_results.csv')

        if not os.path.isfile(csv_file):
            continue

        with open(csv_file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                mos = float(row[1])
                lpips_vals = [float(x) for x in row[2:]]
                avg_lpips = np.mean(lpips_vals)

                mos = 1 - (mos - 1) / (5 - 1)  # normaliser
                all_mos.append(mos)
                all_lpips.append(avg_lpips)

    all_mos = np.array(all_mos)
    all_lpips = np.array(all_lpips)

    X = sm.add_constant(all_lpips)
    model = sm.OLS(all_mos, X).fit()
    predictions = model.predict(X)

    pearson_corr = stats.pearsonr(predictions, all_mos)[0]

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(all_lpips, all_mos, label="Data", color='blue')
    plt.plot(all_lpips, predictions, color='red', label='Linear Fit')
    plt.title(f'Global Correlation (All Viewpoints)\nPearson={pearson_corr:.3f}')
    plt.xlabel('Average LPIPS across all viewpoints')
    plt.ylabel('MOS (normalized)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir + output_plot)
    plt.show()

    print(f"Global correlation plot saved to: {base_dir + output_plot}")
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
    
    plt.savefig(base_dir + 'barplot_combined_correlation_per_object.png')
    plt.show()
    
    print(f"Barplot saved to: barplot_combined_correlation_per_object.png")
    
# Path to the base directory containing all objects
base_dir = "D:/These/Vscode/Graphics-LPIPS/out/TSMD/_METRIC_RESULTS_/"
# process_all_objects(base_dir)
calculate_correlation_all_vps_combined(base_dir)