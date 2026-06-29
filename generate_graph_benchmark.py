import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx
import graph_tool.all as gt
from graph_tool.spectral import adjacency
import os
import joblib
import json

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve, minimize_scalar

BENCHMARK_SIZE = 30 # Nombre de graphe de chaque ratio d'hybridation créés
HYBRID_RATIO_LIST = np.arange(1.00, -0.10, -0.10) # Ratios d'hybridation générés

#######################################
###### FONCTIONS POUR GENERAZAO #######
#######################################

def get_real_graph_properties_sbm_V2(G_train):
    nodes_map = {node: i for i, node in enumerate(G_train.nodes())}
    edges = [(nodes_map[u], nodes_map[v]) for u, v in G_train.edges()]    
    communities = nx.get_node_attributes(G_train, 'sbm_id')
    unique_comms = sorted(list(set(communities.values())))
    mapping = {raw_id: i for i, raw_id in enumerate(unique_comms)}
    
    g = gt.Graph(directed=False)
    g.add_vertex(len(G_train.nodes()))
    g.add_edge_list(edges)
    
    b_array = np.array([mapping[communities[node]] for node in G_train.nodes()])
    b_prop = g.new_vertex_property("int", b_array)

    state = gt.BlockState(g, b=b_prop, deg_corr=True)

    # La matrice de liens entre blocs
    e_rs = state.get_matrix().toarray()
    
    # Les degrés de chaque nœud
    k = g.get_out_degrees(g.get_vertices())
    
    return e_rs, k, b_array

def get_real_graph_properties_pos_V2(G_train, n_components=4, shuffle=True):
    nodes = list(G_train.nodes())
    embeddings_attr = nx.get_node_attributes(G_train, 'deepwalk')
    raw_embeddings = np.array([embeddings_attr[node] for node in nodes])
    degrees = []
    
    for node in nodes:
        degrees.append(G_train.degree(node))
    
    # On réduit les dimensions des embeddings pour éviter des distances trop grandes. Utile ?
    pca = PCA(n_components=n_components, random_state=42)
    pos_reduced = pca.fit_transform(raw_embeddings)
    
    # Normalisation dans [0;1], standard pour modèles spatiaux askip
    pos_min = pos_reduced.min(axis=0)
    pos_max = pos_reduced.max(axis=0)
    pos_normalized = (pos_reduced - pos_min) / (pos_max - pos_min)
        
    if shuffle:
        rng_pos = np.random.default_rng(seed=42)
        idx_pos = rng_pos.permutation(len(pos_normalized))
        pos_final = pos_normalized[idx_pos]
        
        rng_deg = np.random.default_rng(seed=99) 
        idx_deg = rng_deg.permutation(len(degrees))
        degrees_final = np.array(degrees)[idx_deg]
    else:
        degrees_final = degrees
        
    return degrees_final, pos_final


def get_probs_sbm_non_DC(e_rs, b):
    n = len(b)
    n_blocks = e_rs.shape[0]
    P = np.zeros((n, n))
    
    counts = np.bincount(b)
    
    for r in range(n_blocks):
        for s in range(r, n_blocks):
            idx_r = np.where(b == r)[0]
            idx_s = np.where(b == s)[0]
            
            # Calcul du nombre de liens maximum possibles entre ces blocs
            if r == s:
                possible = counts[r] * (counts[r] - 1) / 2
                p_rs = e_rs[r, s] / (2*possible)
            else:
                possible = counts[r] * counts[s]
                p_rs = e_rs[r, s] / possible
                
            P[np.ix_(idx_r, idx_s)] = p_rs
            if r != s:
                P[np.ix_(idx_s, idx_r)] = p_rs

                
    np.fill_diagonal(P, 0)
    n_clipped = np.sum(P > 1.0)
    if n_clipped > 0:
        print(f"Warning: {int(n_clipped/2)} probabilités spatiales plafonnées à 1.0")
    P = np.clip(P, 0, 1)
    
    return P

def get_probs_spatial_non_DC(positions, n_liens_target, sigma=1.0):
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
    
    print(f" Alpha trouvé : {alpha_opt:.4f} pour {n_liens_target} liens visés.")
    return P


def generate_graph_from_probs(P, sbm_groups=None, positions=None):
    n = P.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    
    upper_idx = np.triu_indices(n, k=1)
    probs_vector = P[upper_idx]
    
    mask = np.random.random(len(probs_vector)) < probs_vector
    edges = np.column_stack((upper_idx[0][mask], upper_idx[1][mask]))
    
    g.add_edge_list(edges)
        
    return g

def generate_graph_benchmarks(Hybrid_ratios_list, P_sbm, P_spatial, position, k, degrees, commu, e_rs, name="00_OUBLI_DE_NOM", save_P_matrix = False, nb_iter = "", agg_method=""):
    results_list = []
    all_P_matrices = {}

    for alpha in Hybrid_ratios_list:
        G_name = f"{name}_{f'{alpha:.2f}'.replace('.', '_')}_pos_{f'{1-alpha:.2f}'.replace('.', '_')}{nb_iter}.graphml"

        print("\n" + "="*90)
        print(f"Pour ratio_sbm = {alpha}")
        print("\n" + "="*90)
        
        P_hybride = P_sbm * alpha + P_spatial * (1 - alpha)
        if agg_method == "Custom_exposant" : 
            kpow = 4
            tol = 1e-2
            target_expectation = alpha * np.sum(P_sbm) + (1 - alpha) * np.sum(P_spatial)
            P_base = alpha * (P_sbm**kpow) + (1 - alpha) * (P_spatial**kpow)
            def get_expectation(j):
                return np.sum(P_base**j)
        
            j_min, j_max = 0.1, 50.0 
            if get_expectation(j_max) > target_expectation:
                j_max = 800.0
        
            for _ in range(100):  
                j_mid = (j_min + j_max) / 2
                current_exp = get_expectation(j_mid)
                
                if abs(current_exp - target_expectation) < tol:
                    break
                
                if current_exp > target_expectation:
                    j_min = j_mid
                else:
                    j_max = j_mid
                    
            P_hybride = P_base**j_mid
            norm_facteur = target_expectation/np.sum(P_base)
            print(f"-- Aggrégation par exposant réussie, avec une puissance j={j_mid}")
            print(f"-- Remplace un facteur de normalisation classique de {norm_facteur}")
            
        alpha_key = round(alpha, 2)
        all_P_matrices[alpha_key] = P_hybride.copy()
        g_hybride = generate_graph_from_probs(P_hybride)

        if save_P_matrix : 
            g_hybride_nx = convert_to_nx_with_metadata(g_hybride, position, k, degrees, commu, e_rs, P_hybride)
        else:  
            g_hybride_nx = convert_to_nx_with_metadata(g_hybride, position, k, degrees, commu, e_rs)

        save_as_graphml(g_hybride_nx, filename=G_name)
        
        var_h = get_variance_from_P(P_hybride)
        ent_h = get_entropy_from_p(P_hybride)
        ll_h = get_log_likelihood(g_hybride, P_hybride)
        
        clustering = gt.global_clustering(g_hybride)[0]  
        
        results_list.append({
            "Modèle": f"Hybride (α={alpha:.2f})",
            "N": g_hybride.num_vertices(),
            "E": g_hybride.num_edges(),
            "Variance": f"{var_h:.8f}",
            "Entropy": f"{ent_h:.2f}",
            "Log-Likelihood": f"{ll_h:.2f}",
            "Clustering": f"{clustering:.4f}"
        })

    # --- Affichage final ---
    df_results = pd.DataFrame(results_list)

    print("\n" + "="*90)
    print("📊 TABLEAU RÉCAPITULATIF DE L'HYBRIDATION")
    print("="*90)
    print(df_results)
    print("="*90)

    save_dir = "../../graph_library"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{name}_P_matrices.joblib"
    joblib.dump(all_P_matrices, save_path)
    print(f" Dataset de matrices sauvegardé avec succès : {save_path}")
    print(f"Taille du fichier : {os.path.getsize(save_path) / 1e6:.2f} MB")

    return all_P_matrices


#####################################
###### FONCTIONS POUR ANALYSE #######
#####################################

def get_variance_from_P(P):
    n = P.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    
    p_vector = P[upper_idx].copy()
    variance = np.var(p_vector)
    
    return variance

def get_entropy_from_p(P):
    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = P[upper_idx]
    
    epsilon = 1e-12
    p_vector = np.clip(p_vector, epsilon, 1 - epsilon)
    
    # Formule de l'entropie binaire : H = - [p*log2(p) + (1-p)*log2(1-p)]
    h_binaire = -(p_vector * np.log2(p_vector) + (1 - p_vector) * np.log2(1 - p_vector))
    total_entropy = np.sum(h_binaire)
    
    return total_entropy

def get_log_likelihood(G, P):
    if isinstance(G, nx.Graph):
        adj = nx.to_numpy_array(G, nodelist=range(len(P)))
    else:
        adj = adjacency(G).toarray()

    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = np.clip(P[upper_idx], 1e-12, 1 - 1e-12)
    adj_vector = adj[upper_idx]

    log_likelihood = np.sum(adj_vector * np.log2(p_vector) + (1 - adj_vector) * np.log2(1 - p_vector))
    
    return log_likelihood

def convert_to_nx_with_metadata(gt_graph, positions, k, degrees, sbm_labels, e_rs, Probas_mtx=None):
    edges = gt_graph.get_edges()
    n_nodes = len(sbm_labels)
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_nodes))
    G_nx.add_edges_from(edges)
    
    gt_payload = {
        'GT_degrees_sbm': k.tolist() if hasattr(k, 'tolist') else list(k),
        'GT_degrees_spatial': degrees.tolist() if hasattr(degrees, 'tolist') else list(degrees),
        'GT_pos': positions.tolist() if hasattr(positions, 'tolist') else list(positions),
        'GT_sbm_id': [int(x) for x in sbm_labels],
    }

    num_blocks = e_rs.shape[0]
    counts = Counter(sbm_labels)
    sbm_density_matrix = np.zeros((num_blocks, num_blocks))
    
    for r in range(num_blocks):
        for s in range(r, num_blocks):
            n_r, n_s = counts[r], counts[s]
            links = e_rs[r, s]
            if r == s:
                possible = n_r * (n_r - 1) / 2
                dens = links / (2 * possible) if possible > 0 else 0
            else:
                possible = n_r * n_s
                dens = links / possible if possible > 0 else 0
            
            sbm_density_matrix[r, s] = dens
            sbm_density_matrix[s, r] = dens
    
    gt_payload['GT_sbm_matrix'] = sbm_density_matrix.tolist()


    G_nx.graph['GroundTruth_JSON'] = json.dumps(gt_payload)
    if Probas_mtx is not None:
        G_nx.graph['P_matrix_JSON'] = json.dumps(Probas_mtx.tolist())

    return G_nx


def save_as_graphml(G_nx, filename="mon_graphe.graphml", folder="graph_library"):
    path = os.path.join(folder, filename)
    nx.write_graphml(G_nx, path)
    print(f"Graphe exporté avec succès dans : {path}")


def match_spatial_to_sbm_variance(P_sbm, degrees, positions, DC=True):
    target_variance = get_variance_from_P(P_sbm)
    target_links = np.sum(degrees) / 2
    print(f"Variance cible (SBM) : {target_variance:.8f}")
    print("-" * 30)

    history = {'step': 0}

    def objective(sigma_test):
        history['step'] += 1
        
        if DC : 
            P_test, alphas = get_probs_spatial_DC(degrees, positions, sigma=sigma_test)
        else : 
            P_test = get_probs_spatial_non_DC(positions, n_liens_target= target_links,sigma=sigma_test)
        current_var = get_variance_from_P(P_test)
        diff = abs(current_var - target_variance)
        
        print(f"Step {history['step']:02d} | Sigma testé: {sigma_test:.4f} | Var: {current_var:.8f} | Δ: {diff:.2e}")
        
        return (current_var - target_variance)**2

    res = minimize_scalar(objective, bounds=(0.005, 50), method='bounded')
    
    print("-" * 30)
    opt_sigma = res.x
    
    if DC : 
        final_P_spatial, alphas = get_probs_spatial_DC(degrees, positions, sigma=opt_sigma)
    else : 
        final_P_spatial = get_probs_spatial_non_DC(positions, n_liens_target= target_links, sigma=opt_sigma)
    final_var = get_variance_from_P(final_P_spatial)
    
    print(f"✨ Sigma optimal trouvé : {opt_sigma:.4f}")
    print(f"📊 Variance finale Spatial : {final_var:.8f} (Écart: {abs(final_var-target_variance):.2e})")
    
    return opt_sigma, final_P_spatial

###########################################
###### MAIN QUI GENERE LE BENCHMARK #######
###########################################

if __name__ == "__main__":
    G_name = "reel_jazz_collab_w_attributes.joblib"
    path = f"graph_library/{G_name}"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de base est introuvable à l'emplacement : {path}")
        
    print(f"Fichier source détecté : {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    G = joblib.load(path)
    print(f"Graphe d'origine chargé : {G.number_of_nodes()} nœuds, {G.number_of_edges()} liens.")

    # 2. Boucle de génération principale (30 itérations)
    for nb_iter in range(1, BENCHMARK_SIZE + 1):
        print(f"DÉBUT DE L'ITÉRATION DE GÉNÉRATION N° {nb_iter} / 30")
        
        nb_iter_name = f"_{nb_iter}"
        
        # Extraction des propriétés SBM (Non Degré-Corrigé)
        e_rs, k, commus = get_real_graph_properties_sbm_V2(G)
        P_sbm = get_probs_sbm_non_DC(e_rs, commus)
        
        # Extraction des propriétés Spatiales (Non Degré-Corrigé)
        degrees, position = get_real_graph_properties_pos_V2(G)
        
        # Calibrage du modèle spatial (recherche de sigma pour matcher la variance du SBM)
        sigma_opt, P_spatial_calibrated = match_spatial_to_sbm_variance(
            P_sbm, 
            degrees, 
            position, 
            DC=False
        )
        
        # Génération des graphes hybrides et exports des fichiers (.graphml et matrices .joblib)
        all_P_matrices = generate_graph_benchmarks(
            Hybrid_ratios_list=HYBRID_RATIO_LIST, 
            P_sbm=P_sbm, 
            P_spatial=P_spatial_calibrated, 
            position=position, 
            k=k, 
            degrees=degrees, 
            commu=commus, 
            e_rs=e_rs, 
            name="artificial_graph_sbmv_4", 
            save_P_matrix=True, 
            nb_iter=nb_iter_name, 
            agg_method="" # Agrégation par somme
        )

        print(f"\nItération {nb_iter} terminée. Échantillon de probabilités (alpha={list(all_P_matrices.keys())[0]}) :")
        print(all_P_matrices[list(all_P_matrices.keys())[0]][:3, :3])
        print("#"*100)
