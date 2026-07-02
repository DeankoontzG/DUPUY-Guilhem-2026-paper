from NullModelsInference import get_gravity_null_model_manual_iterative

import time
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import multiprocessing


def _append_DeepwalkOnResiduals_with_sampling(G_train, pos_attr="GT_pos", attr_name="deepwalk_residuals_sampl", 
                                             NullModel_method="ManualIter", emb_dim=64, 
                                             k_neighbors=15, keep_original_weights=True):
    print(f"Calcul de Deepwalk on Residuals avec Échantillonnage (NullModel type ={NullModel_method})...")
    start_time = time.time()

    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    N = len(nodes)
    cores = multiprocessing.cpu_count() -1
    
    if NullModel_method == "ManualIter":
        P, _ = get_gravity_null_model_manual_iterative(G_train, pos_attr)
        P_symmetric = (P + P.T) / 2
        R_matrix = A - P_symmetric
    else:
        print(f"NullModel_method {NullModel_method} non reconnue")
        return G_train

    R_pos = np.maximum(R_matrix, 0)
    R_sparse = np.zeros_like(R_pos)
    
    for idx in range(N):
        row = R_pos[idx].copy()
        valid_indices = np.where(row > 0)[0]
        
        if len(valid_indices) == 0:
            continue
            
        if len(valid_indices) <= k_neighbors:
            chosen_indices = valid_indices
        else:
            row_probs = row[valid_indices] / np.sum(row[valid_indices])
            
            # --- TIRAGE SANS REMISE ---
            chosen_indices = np.random.choice(
                valid_indices, 
                size=k_neighbors, 
                replace=False, 
                p=row_probs
            )
            
        # Remplissage de la ligne selon la stratégie choisie
        if keep_original_weights:
            R_sparse[idx, chosen_indices] = row[chosen_indices]
        else:
            R_sparse[idx, chosen_indices] = 1.0  # Graphe non-pondéré (évite le double impact)

    G_res_tmp = nx.from_numpy_array(R_sparse, create_using=nx.DiGraph)
    mapping = {i: nodes[i] for i in range(N)}
    G_res_tmp = nx.relabel_nodes(G_res_tmp, mapping)
    
    # Nettoyage des arêtes à 0
    zero_edges = [(u, v) for u, v, d in G_res_tmp.edges(data=True) if d.get('weight', 1) == 0]
    G_res_tmp.remove_edges_from(zero_edges)
    
    print(f"Graphe temporaire généré : {G_res_tmp.number_of_nodes()} nœuds et {G_res_tmp.number_of_edges()} arêtes (k={k_neighbors}).")

    print("Génération des marches aléatoires sur le graphe échantillonné...")
    
    # Si keep_original_weights=True, on passe 'weight', sinon Node2Vec ignore le poids de toute façon
    weight_key = 'weight' if keep_original_weights else None
    
    node2vec = Node2Vec(G_res_tmp, 
                        dimensions=emb_dim, 
                        walk_length=30, 
                        num_walks=100, 
                        workers=cores, 
                        p=1.0, q=1.0, # p=1, q=1 -> Comportement DeepWalk pur
                        weight_key=weight_key)

    print("Entraînement du modèle Skip-gram (Word2Vec)...")
    model = node2vec.fit(window=10, min_count=1, batch_words=1000)
    
    embeddings = {}
    for node in G_train.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node]
        elif str(node) in model.wv:
            embeddings[node] = model.wv[str(node)]
        else:
            # Cas rare où le nœud n'avait aucun résidu positif et s'est retrouvé isolé
            embeddings[node] = np.zeros(emb_dim)

    nx.set_node_attributes(G_train, embeddings, attr_name)
    
    print(f"Deepwalk on Residuals terminé en {time.time() - start_time:.2f}s ({emb_dim} dimensions).")
    return G_train


def _append_DeepwalkOnResiduals_top_quantile(G_train, pos_attr="GT_pos", attr_name="deepwalk_residuals_quantile", 
                                            NullModel_method="ManualIter", emb_dim=64, 
                                            local_quantile=0.75):
    """
    Sparsifie la matrice de résidus en ne gardant, pour chaque nœud, que les arêtes 
    supérieures au quantile local défini (ex: 0.75 pour le top 25% des résidus positifs).
    """
    print(f"Calcul de Deepwalk on Residuals (Top {int((1-local_quantile)*100)}% locaux, NullModel={NullModel_method})...")
    start_time = time.time()

    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    N = len(nodes)
    cores = multiprocessing.cpu_count() -1
    
    if NullModel_method == "ManualIter":
        P, _ = get_gravity_null_model_manual_iterative(G_train, pos_attr)
        P_symmetric = (P + P.T) / 2
        R_matrix = A - P_symmetric
    else:
        print(f"NullModel_method {NullModel_method} non reconnue")
        return G_train

    R_pos = np.maximum(R_matrix, 0)
    R_sparse = np.zeros_like(R_pos)
    
    for idx in range(N):
        row = R_pos[idx].copy()
        valid_indices = np.where(row > 0)[0]
        
        if len(valid_indices) == 0:
            continue
            
        threshold = np.percentile(row[valid_indices], local_quantile * 100)
        top_indices = valid_indices[row[valid_indices] >= threshold]
        R_sparse[idx, top_indices] = row[top_indices]

    # 3. Création du graphe NetworkX temporaire
    # Note : Le top 25% de A vers B n'implique pas forcément que A soit dans le top 25% de B.
    # On utilise donc un DiGraph (graphe orienté) pour respecter cette asymétrie locale.
    G_res_tmp = nx.from_numpy_array(R_sparse, create_using=nx.DiGraph)
    
    # Remappage des labels des nœuds
    mapping = {i: nodes[i] for i in range(N)}
    G_res_tmp = nx.relabel_nodes(G_res_tmp, mapping)
    
    # Nettoyage
    zero_edges = [(u, v) for u, v, d in G_res_tmp.edges(data=True) if d.get('weight', 1) == 0]
    G_res_tmp.remove_edges_from(zero_edges)
    
    print(f"Graphe temporaire généré : {G_res_tmp.number_of_nodes()} nœuds et {G_res_tmp.number_of_edges()} arêtes.")

    print("Génération des marches aléatoires sur le graphe filtré...")
    node2vec = Node2Vec(G_res_tmp, 
                        dimensions=emb_dim, 
                        walk_length=30, 
                        num_walks=100, 
                        workers=cores, 
                        p=1.0, q=1.0, # Comportement DeepWalk pur
                        weight_key='weight') # Prise en compte de l'intensité du résidu

    print("Entraînement du modèle Skip-gram (Word2Vec)...")
    model = node2vec.fit(window=10, min_count=1, batch_words=1000)
    
    embeddings = {}
    for node in G_train.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node]
        elif str(node) in model.wv:
            embeddings[node] = model.wv[str(node)]
        else:
            embeddings[node] = np.zeros(emb_dim)

    nx.set_node_attributes(G_train, embeddings, attr_name)
    
    print(f"Deepwalk on Residuals terminé en {time.time() - start_time:.2f}s ({emb_dim} dimensions).")
    return G_train