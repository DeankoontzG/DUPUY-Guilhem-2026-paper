from pipeline_utils import *
from models import *
import numpy as np
import pandas as pd
import json
import os
import sys
import platform
import subprocess
import scipy.stats as stats
from joblib import Parallel, delayed

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print("No display found. Running in headless mode (saving plots to disk only).")
    matplotlib.use('Agg')
else:
    print("Display detected. Plots will open in a new window.")
    matplotlib.use('TkAgg')
    
import matplotlib.pyplot as plt
import seaborn as sns

def compute_features(G, G_name, spatial_ref = "GT_pos"):

    validate_input_graph(G)
    print("[PREP] Validation du Graphe terminée. Lancement des calculs...")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
    
        GT = {
            'GT_pos': np.array(gt_raw.get('GT_pos', [])),
            'GT_sbm_id': np.array(gt_raw.get('GT_sbm_id', [])),
            'GT_sbm_matrix': np.array(gt_raw.get('GT_sbm_matrix', []))
            
        }
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    if GT is not None and 'GT_pos' in GT:
            for i, node_id in enumerate(G.nodes()):
                G.nodes[node_id]['GT_pos'] = GT['GT_pos'][i]
                G.nodes[node_id]['GT_sbm_id'] = GT['GT_sbm_id'][i]
    
    G_kept, G_hidden = hide_graph_links(G, test_size=0.10)
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)
        
    G_train_with_communities = computeCommunityFeatures(G_train, spatial_ref=spatial_ref)
    G_kept_with_communities = computeCommunityFeatures(G_kept, spatial_ref=spatial_ref)

    G_train_with_communities = computeDistanceFeatures(G_train, spatial_ref=spatial_ref)
    G_kept_with_communities = computeDistanceFeatures(G_kept, spatial_ref=spatial_ref)
        
    print("Save : graph pour exploration")
    loadsave_data_joblib(data=G_kept_with_communities, filename=f"G_kept_w_struct_com_dist_{G_name}", mode="save")
    loadsave_data_joblib(data=G_train_with_communities, filename=f"G_train_w_struct_com_dist_{G_name}", mode="save")
    
    dataset_train = prepare_balanced_data(G_test, G_train_with_communities,  negative_ratio=10.0, GroundTruth=GT)
    dataset_hidden = prepare_balanced_data(G_hidden, G_kept_with_communities, negative_ratio=50.0, GroundTruth=GT)

    print("Vérif : colonnes du dataset :")
    print(dataset_train.columns)

    print("Sauvegarde des datasets")
    save_dataset(dataset=dataset_train, filename=f"dataset_train_{G_name}")
    save_dataset(dataset=dataset_hidden, filename=f"dataset_hidden_{G_name}")

def analyze_features(G_name_short, nb_iterations, spatial_ref = "GT_pos", i_min =0.00, i_max = 1.00, nb_i=11, name_export_results="DATE"):

    features_GT_pos = ['GT_pos_dist']
    features_GT_sbm = ["GT_sbm_density"]
    features_louvain = ["louvain_density"]
    features_spatial_louvain = ["spatial_louvain_density"]
    features_deepwalk = ["deepwalk_dist"]
    features_SiNEcustom_spatial = ["SiNEcustom_spatial_dist"]
    # ---- Tests ----
    features_deepwalk_residuals = ["deepwalk_residuals_dist"]
    features_SiNEcustom_sbm = ["SiNEcustom_sbm_dist"]
    features_sbm_louvain = ["sbm_louvain_density"]

    experiments = {
        "GT_pos": features_GT_pos,
        "GT_sbm": features_GT_sbm,
        "Louvain": features_louvain,
        "Louvain spatial": features_spatial_louvain,
        "Deepwalk": features_deepwalk,
        "SiNEcustom_spatial": features_SiNEcustom_spatial,
        # ---- Tests ----
        "SiNEcustom_sbm": features_SiNEcustom_sbm,
        "Deepwalk on residuals": features_deepwalk_residuals,
        "Louvain sbm": features_sbm_louvain,

        "GT_pos + louvain": features_GT_pos + features_louvain,
        "GT_pos + spatial louvain": features_GT_pos + features_spatial_louvain,
        "GT_pos + Deepwalk": features_GT_pos + features_deepwalk,
        "GT_pos + SiNEcustom_spatial": features_GT_pos + features_SiNEcustom_spatial,
        # ---- Tests ----
        "GT_pos + Deepwalk on residuals": features_GT_pos + features_deepwalk_residuals,
        "GT_sbm + SiNEcustom_sbm": features_GT_sbm + features_SiNEcustom_sbm,
        "GT_sbm + Louvain sbm": features_GT_sbm + features_sbm_louvain,
        "GT_sbm + Louvain": features_GT_sbm + features_louvain,
        "GT_sbm + Deepwalk": features_GT_sbm + features_deepwalk,
    }

    all_results = []

    tasks = [
        (nb_iter, i) 
        for nb_iter in range(1, nb_iterations + 1) 
        for i in np.linspace(i_max, i_min, nb_i)
    ]

    cores_to_use = max(1, os.cpu_count() -2)

    print(f"Lancement de la parallélisation sur {cores_to_use} coeurs pour {len(tasks)} tâches...")

    # Exécution parallèle
    results_nested = Parallel(n_jobs=cores_to_use)(
        delayed(run_single_experiment)(nb_iter, i, spatial_ref, G_name_short, experiments) 
        for nb_iter, i in tasks
    )

    # Aplatir la liste de listes
    all_results = [item for sublist in results_nested for item in sublist]

    all_results = pd.DataFrame(all_results)
    
    output_dir = "your_results/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"link_prediction_perfs_{G_name_short}_{nb_iterations}iter_{name_export_results}.csv")
    all_results.to_csv(output_path, index=False)
    print(f" Succès ! Fichier sauvegardé dans : {output_path}")
    return all_results

def run_single_experiment(nb_iter, i, spatial_ref, G_name_short, experiments):
        """
        Fonction exécutée par un cœur unique pour une valeur de i et une itération donnée.
        """
        sbm_val = f"{i:.2f}"
        pos_val = f"{1-i:.2f}"
        if spatial_ref == "GT_Pos" or spatial_ref == "GT_pos" : 
            spatial_ref = ""
        else :
            spatial_ref = f"_{spatial_ref}"
        G_name = f"{G_name_short}_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_{nb_iter}{spatial_ref}"
    

        # 1. Chargement des données d'entraînement
        _, dataset_train, dataset_eval, _ = load_all_data_for_graph(G_name)
        local_results = []

        for exp_name, feat_list in experiments.items():
            missing = set(feat_list) - set(dataset_train.columns)
            if missing:
                print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
                print(set(dataset_train.columns))
                continue

            #print(f" Running: {exp_name} for SBM={i}")
        
            Params = {
                'max_depth': 3,             # Faible profondeur pour éviter l'overfitting sur 2 variables
                'learning_rate': 0.1,       # Compromis idéal vitesse/précision
                'n_estimators': 1000,       # On met beaucoup, l'early stopping fera le reste
                'subsample': 1.0,           # On garde 100% des lignes (plus stable pour peu de features)
                'colsample_bytree': 1.0,    # On garde les 2 features à chaque split
                'objective': 'binary:logistic', 
                'tree_method': 'hist',      # Accélère l'entraînement sur de gros datasets
                'reg_lambda': 1,            # Régularisation L2 pour stabiliser les poids
                'n_jobs': 1                # Utilise 1 seul coeur, pour la parallélisation
            }

            stats_df, model, _, _, _, _ = train_and_test_xgboost(dataset_train, features=feat_list, parameters = Params, plot=False)

            importances = model.feature_importances_
            feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
                
            # Évaluation sur le dataset de référence FIXE (Graphe SBM 1.0)
            X_eval_fixed = dataset_eval[feat_list] 
            stats_eval_df = get_performance_metrics(model, X_eval_fixed, dataset_eval["target"], "EXP_")
            
            local_results.append({
                "G_name" : G_name,
                "Ratio_SBM": i,
                "Iter": nb_iter,
                "Experiment": exp_name,
                "AP_train": stats_df["Test_AP"].iloc[0],
                "AUC-ROC_train": stats_df["Test_AUC-ROC"].iloc[0],
                "AP_eval": stats_eval_df["EXP_AP"].iloc[0],
                "AUC-ROC_eval": stats_eval_df["EXP_AUC-ROC"].iloc[0],
                "Top_Feature": feat_imp_series.index[0], # On stocke la #1 pour analyse
                "Top_Importance": feat_imp_series.iloc[0]
            })

        return local_results

#############################################
### FONCTIONS POUR AFFICHER LES RESULTATS ###
#############################################

def confidence_interval_95(data):
    n = len(data)
    if n < 2: return 0
    sem = stats.sem(data)
    return sem * stats.t.ppf((1 + 0.95) / 2., n - 1)

def generate_and_show_plot(csv_path, metric="AUC-ROC_eval", name="plot_output"):
    """Loads data, aggregates performance metrics, and displays an interactive pop-up plot."""
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' could not be found.")
        return

    df_compare = pd.read_csv(csv_path)

    feat_to_plot = [
        "GT_pos",
        "GT_sbm",
        "Louvain",
        "Louvain spatial",
        "Deepwalk",
        "SiNEcustom_spatial",
        # ---- Tests ----
        "SiNEcustom_sbm",
        "Deepwalk on residuals",
        "Louvain sbm",

        "GT_pos + louvain",
        "GT_pos + spatial louvain",
        "GT_pos + Deepwalk",
        "GT_pos + SiNEcustom_spatial",
        # ---- Tests ----
        "GT_pos + Deepwalk on residuals",
        "GT_sbm + SiNEcustom_sbm",
        "GT_sbm + Louvain sbm",
        "GT_sbm + Louvain",
        "GT_sbm + Deepwalk",
    ]

    
    df_compare = df_compare[df_compare["Experiment"].isin(feat_to_plot)]
    df_compare = df_compare.sort_values(by="Experiment", ascending=True)

    # 1. Aggregation
    metrics = ["AP_train", "AUC-ROC_train", "AP_eval", "AUC-ROC_eval"] 
    df_avg = df_compare.groupby(["Ratio_SBM", "Experiment"])[metrics].agg([
        ('mean', np.mean),
        ('ci95', confidence_interval_95)
    ]).reset_index()

    df_avg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_avg.columns.values]

    # 2. Plotting
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        'font.size': 11
    })

    experiments = df_avg['Experiment'].unique()
    palette = sns.color_palette("tab10", len(experiments))

    for i, exp in enumerate(experiments):
        subset = df_avg[df_avg["Experiment"] == exp]
        mean_col = f"{metric}_mean"
        ci_col = f"{metric}_ci95" 
        
        color = palette[i]

        # Utilisation directe du nom original 'exp' pour le label
        plt.plot(subset["Ratio_SBM"], subset[mean_col], 
                 marker='o', label=exp, color=color, linewidth=2, markersize=6)
        
        plt.fill_between(
            subset["Ratio_SBM"], 
            subset[mean_col] - subset[ci_col], 
            subset[mean_col] + subset[ci_col], 
            color=color, alpha=0.12
        )

    plt.xlabel(r"$\alpha$ (SBM weight ratio in hybridization)", labelpad=10)
    plt.ylabel(f"{metric.replace('_', ' ')}", labelpad=10)
    plt.legend(title="Configurations", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0.48, 1.02)
    plt.tight_layout()

    # Automatic Saving
    output_dir = os.path.join("your_results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{metric}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print("="*80)
    print(f"Plot successfully saved to: {save_path}")
    
    # FORCE INTERACTIVE DISPLAY WITHOUT FREEZING THE TERMINAL UPON CLOSING
    print("Opening plot window... Close the window to return control to the terminal.")
    plt.ion()  
    plt.show(block=True)