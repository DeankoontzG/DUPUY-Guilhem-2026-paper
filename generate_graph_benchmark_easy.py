import pandas as pd
import numpy as np
import networkx as nx
import graph_tool.all as gt
import json
import os
import joblib
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve, minimize_scalar

# Configuration globale pour le benchmark synthétique
BENCHMARK_SIZE = 10
HYBRID_RATIO_LIST = np.arange(1.00, -0.10, -0.10)
HYBRIDATION_METHOD = "Somme"  # Ou "Custom_exposant"

# Paramètres du nouveau générateur analytique
N_NODES = 198
N_COMMUNITIES = 9  # 9 communautés * 22 nœuds = 198 nœuds (parfaitement égalitaire)
P_IN = 1.00
P_OUT = 0.00

#######################################
###### NOUVEAUX GENERATEURS PURS ######
#######################################

def generate_analytic_sbm_probs(n_nodes, n_communities, p_in, p_out):
    """
    Génère une matrice de probabilités SBM analytique et propre
    avec des blocs de tailles identiques.
    """
    nodes_per_comm = n_nodes // n_communities
    comm_labels = np.array([i // nodes_per_comm for i in range(n_nodes)])
    # Gérer le reste éventuel si n_nodes n'est pas un multiple parfait
    comm_labels = np.clip(comm_labels, 0, n_communities - 1)
    
    P_sbm = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if comm_labels[i] == comm_labels[j]:
                P_sbm[i, j] = p_in
                P_sbm[j, i] = p_in
            else:
                P_sbm[i, j] = p_out
                P_sbm[j, i] = p_out
                
    # Calcul théorique des liens attendus
    counts = np.bincount(comm_labels)
    intra_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_pairs = n_nodes * (n_nodes - 1) // 2
    inter_pairs = total_pairs - intra_pairs
    expected_links = (intra_pairs * p_in) + (inter_pairs * p_out)
    
    print(f"[SBM] {n_communities} blocs de {counts[0]} nœuds. Liens théoriques attendus : {expected_links:.1f}")
    return P_sbm, comm_labels, expected_links

def generate_analytic_spatial_positions(n_nodes):
    """
    Distribue les nœuds de manière uniforme dans un carré [0, 1]^2.
    L'assignation est orthogonale par défaut (indépendante du SBM).
    """
    rng = np.random.default_rng(seed=42)
    positions = rng.uniform(0.0, 1.0, size=(n_nodes, 2))
    return positions

def get_probs_spatial_non_DC(positions, n_liens_target, sigma=1.0):
    """
    Calcule les probabilités spatiales via une fonction de dissuasion logistique.
    """
    n = len(positions)
    dist_matrix = squareform(pdist(positions, 'euclidean'))

    deterrence = sigma * dist_matrix
    iu = np.triu_indices(n, k=1)
    det_vec = deterrence[iu]

    def objective(alpha):
        logits = alpha - det_vec
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.sum(probs) - n_liens_target

    alpha_opt = fsolve(objective, x0=0.0)[0]
    
    logit_final = alpha_opt - deterrence
    P = 1.0 / (1.0 + np.exp(-logit_final))
    np.fill_diagonal(P, 0)
    
    return P

#######################################
###### FONCTIONS POUR METRIQUES #######
#######################################

def get_variance_from_P(P):
    n = P.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    return np.var(P[upper_idx])

def get_entropy_from_p(P):
    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = np.clip(P[upper_idx], 1e-12, 1 - 1e-12)
    h_binaire = -(p_vector * np.log2(p_vector) + (1 - p_vector) * np.log2(1 - p_vector))
    return np.sum(h_binaire)

def get_log_likelihood(G, P):
    from graph_tool.spectral import adjacency
    if isinstance(G, nx.Graph):
        adj = nx.to_numpy_array(G, nodelist=range(len(P)))
    else:
        adj = adjacency(G).toarray()
    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = np.clip(P[upper_idx], 1e-12, 1 - 1e-12)
    adj_vector = adj[upper_idx]
    return np.sum(adj_vector * np.log2(p_vector) + (1 - adj_vector) * np.log2(1 - p_vector))

#######################################
###### CALIBRATION ET HYBRIDATION #####
#######################################

def match_spatial_to_sbm_variance(P_sbm, positions, target_links):
    target_variance = get_variance_from_P(P_sbm)
    print(f"\n[Calibration] Variance cible (SBM) : {target_variance:.8f}")
    print("-" * 50)

    history = {'step': 0}

    def objective(sigma_test):
        history['step'] += 1
        P_test = get_probs_spatial_non_DC(positions, n_liens_target=target_links, sigma=sigma_test)
        current_var = get_variance_from_P(P_test)
        diff_pct = (abs(current_var - target_variance) / target_variance) * 100
        
        if history['step'] % 5 == 0 or history['step'] == 1:
            print(f"Étape {history['step']:02d} | Sigma: {sigma_test:.4f} | Var: {current_var:.8f} | Écart: {diff_pct:.2f}%")
        
        return (current_var - target_variance)**2

    res = minimize_scalar(objective, bounds=(0.01, 100.0), method='bounded')
    
    opt_sigma = res.x
    final_P_spatial = get_probs_spatial_non_DC(positions, n_liens_target=target_links, sigma=opt_sigma)
    final_var = get_variance_from_P(final_P_spatial)
    
    # Calcul précis de l'écart final en %
    final_diff_pct = (abs(final_var - target_variance) / target_variance) * 100
    
    print("-" * 50)
    print(f"✨ Sigma optimal trouvé : {opt_sigma:.4f}")
    print(f"📊 Variance finale Spatial : {final_var:.8f}")
    print(f"📢 ÉCART DE VARIANCE FINAL : {final_diff_pct:.4f} %")
    print("-" * 50)
    
    return opt_sigma, final_P_spatial

def generate_graph_from_probs(P):
    n = P.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    upper_idx = np.triu_indices(n, k=1)
    probs_vector = P[upper_idx]
    mask = np.random.random(len(probs_vector)) < probs_vector
    edges = np.column_stack((upper_idx[0][mask], upper_idx[1][mask]))
    g.add_edge_list(edges)
    return g

def convert_to_nx_with_metadata(gt_graph, positions, comm_labels, Probas_mtx=None):
    edges = gt_graph.get_edges()
    n_nodes = len(comm_labels)
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_nodes))
    G_nx.add_edges_from(edges)
    
    # 1. Recréer les listes de degrés attendues par le JSON
    degrees_list = [G_nx.degree(n) for n in G_nx.nodes()]
    
    # 2. Construction de la matrice SBM THÉORIQUE (9x9)
    # C'est beaucoup plus propre pour la Ground Truth !
    num_blocks = len(np.unique(comm_labels))
    sbm_density_matrix = np.zeros((num_blocks, num_blocks))
    
    for r in range(num_blocks):
        for s in range(num_blocks):
            if r == s:
                sbm_density_matrix[r, s] = P_IN   # 0.45
            else:
                sbm_density_matrix[r, s] = P_OUT  # 0.06

    # 3. Payload d'origine strict, converti en listes standards pour le JSON
    gt_payload = {
        'GT_degrees_sbm': degrees_list,
        'GT_degrees_spatial': degrees_list,
        'GT_pos': positions.tolist() if hasattr(positions, 'tolist') else list(positions),
        'GT_sbm_id': [int(x) for x in comm_labels], # Liste d'entiers standard
        'GT_sbm_matrix': sbm_density_matrix.tolist() # Matrice 9x9 de floats
    }

    # 4. Injection
    G_nx.graph['GroundTruth_JSON'] = json.dumps(gt_payload)
    if Probas_mtx is not None:
        G_nx.graph['P_matrix_JSON'] = json.dumps(Probas_mtx.tolist())

    return G_nx

def generate_graph_benchmarks(Hybrid_ratios_list, P_sbm, P_spatial, positions, comm_labels, name, nb_iter):
    results_list = []
    all_P_matrices = {}

    for alpha in Hybrid_ratios_list:
        G_name = f"{name}_{f'{alpha:.2f}'.replace('.', '_')}_pos_{f'{1-alpha:.2f}'.replace('.', '_')}_{nb_iter}.graphml"
        
        # Fusion des matrices
        if HYBRIDATION_METHOD == "Somme":
            P_hybride = P_sbm * alpha + P_spatial * (1 - alpha)
        elif HYBRIDATION_METHOD == "Custom_exposant":
            kpow = 4
            tol = 1e-2
            target_expectation = alpha * np.sum(P_sbm) + (1 - alpha) * np.sum(P_spatial)
            P_base = alpha * (P_sbm**kpow) + (1 - alpha) * (P_spatial**kpow)
            
            j_min, j_max = 0.1, 50.0
            for _ in range(100):  
                j_mid = (j_min + j_max) / 2
                current_exp = np.sum(P_base**j_mid)
                if abs(current_exp - target_expectation) < tol:
                    break
                if current_exp > target_expectation:
                    j_min = j_mid
                else:
                    j_max = j_mid
            P_hybride = P_base**j_mid
            
        alpha_key = round(alpha, 2)
        all_P_matrices[alpha_key] = P_hybride.copy()
        
        # Échantillonnage
        g_hybride = generate_graph_from_probs(P_hybride)
        g_hybride_nx = convert_to_nx_with_metadata(g_hybride, positions, comm_labels, P_hybride)
        
        # Sauvegarde
        os.makedirs("graph_library", exist_ok=True)
        nx.write_graphml(g_hybride_nx, os.path.join("graph_library", G_name))
        
        # Métriques pour le log
        var_h = get_variance_from_P(P_hybride)
        ent_h = get_entropy_from_p(P_hybride)
        ll_h = get_log_likelihood(g_hybride, P_hybride)
        clustering = gt.global_clustering(g_hybride)[0]
        
        results_list.append({
            "Ratio SBM (α)": f"{alpha:.2f}",
            "N": g_hybride.num_vertices(),
            "E": g_hybride.num_edges(),
            "Variance": f"{var_h:.8f}",
            "Entropy": f"{ent_h:.2f}",
            "Clustering": f"{clustering:.4f}"
        })

    print(pd.DataFrame(results_list).to_string(index=False))
    return all_P_matrices

#######################################
###### EXECUTION DU BENCHMARK #########
#######################################

if __name__ == "__main__":
    print("=========================================================================")
    print(f"LANCEMENT DU GÉNÉ SIMPLE (N={N_NODES}, {HYBRIDATION_METHOD})")
    print("=========================================================================")

    for nb_iter in range(1, BENCHMARK_SIZE + 1):
        print(f"\n▶️ ITÉRATION DE GÉNÉRATION N° {nb_iter} / {BENCHMARK_SIZE}")
        
        # 1. Génération SBM Pur
        P_sbm, comm_labels, target_links = generate_analytic_sbm_probs(N_NODES, N_COMMUNITIES, P_IN, P_OUT)
        
        # 2. Génération Spatial Pur (Positions Aléatoires Uniformes)
        positions = generate_analytic_spatial_positions(N_NODES)
        
        # 3. Calibration de la variance spatiale sur la variance SBM
        sigma_opt, P_spatial_calibrated = match_spatial_to_sbm_variance(P_sbm, positions, target_links)
        
        # 4. Mixage et écriture des fichiers du benchmark
        if HYBRIDATION_METHOD == "Somme": 
            all_P_matrices = generate_graph_benchmarks(
                Hybrid_ratios_list=HYBRID_RATIO_LIST,
                P_sbm=P_sbm,
                P_spatial=P_spatial_calibrated,
                positions=positions,
                comm_labels=comm_labels,
                name="artificial_graph_simple_somme",
                nb_iter=nb_iter
            )
        elif HYBRIDATION_METHOD == "Custom_exposant": 
            all_P_matrices = generate_graph_benchmarks(
                Hybrid_ratios_list=HYBRID_RATIO_LIST,
                P_sbm=P_sbm,
                P_spatial=P_spatial_calibrated,
                positions=positions,
                comm_labels=comm_labels,
                name="artificial_graph_simple_power",
                nb_iter=nb_iter
            )
        
    print("\nGénération terminée avec succès. Fichiers disponibles dans 'graph_library/'.")