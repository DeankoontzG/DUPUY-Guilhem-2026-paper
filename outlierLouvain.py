import time
import networkx as nx
import numpy as np
import inspect

from SiNEcustom import train_custom_signed_embedding

def _append_OutlierLouvain_SiNEcustom(G_train, attr_com="outlier_com_id", attr_pos="outlier_pos", emb_dim=64, outlier_threshold=0.15, temperature=0.5, epochs=100, lr=0.1):
    print("Calcul de Outlier Louvain suivi de SiNE custom sur résidus SBM filtrés...")
    start_time = time.time()
    
    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    N = A.shape[0]
    
    # 1. Communautés initiales via ta méthode robuste
    best_p = _find_best_partition(
        G_train, 
        nx.community.louvain_communities, 
        K_min=3, 
        min_edge_ratio=0.01
    )
    
    # On aligne com_labels avec l'ordre exact de la liste 'nodes' (pour la suite avec la matrice A)
    com_labels = np.array([best_p[node] for node in nodes])

    # Étape 2 : Identification des outliers
    final_com_labels = com_labels.copy()
    for i, node in enumerate(nodes):
        neighbors = list(G_train.neighbors(node))
        if len(neighbors) == 0:
            continue
            
        com_counts = {}
        for n in neighbors:
            # CORRECTION BUG & PERF : On interroge directement le dictionnaire de partition
            c = best_p[n] 
            com_counts[c] = com_counts.get(c, 0) + 1
            
        max_internal_ratio = max(com_counts.values()) / len(neighbors)
        if max_internal_ratio < outlier_threshold:
            final_com_labels[i] = -1

    # 3. Construction du modèle nul SBM (les outliers n'ont pas de prédiction SBM, P_ij reste à 0)
    degrees = np.sum(A, axis=1)
    P = np.zeros((N, N))
    valid_mask = (final_com_labels != -1)
    
    if np.sum(valid_mask) > 0:
        unique_coms, inverse_coms = np.unique(final_com_labels[valid_mask], return_inverse=True)
        K = len(unique_coms)
        M = np.zeros((K, K))
        kappa = np.zeros(K)
        A_valid = A[valid_mask][:, valid_mask]
        degrees_valid = degrees[valid_mask]
        
        for r in range(K):
            mask_r = (inverse_coms == r)
            kappa[r] = np.sum(degrees_valid[mask_r])
            for s in range(K):
                mask_s = (inverse_coms == s)
                M[r, s] = np.sum(A_valid[mask_r][:, mask_s])
                
        with np.errstate(divide='ignore', invalid='ignore'):
            Omega = M / (np.outer(kappa, kappa) + 1e-12)
            
        Omega_expanded = Omega[inverse_coms[:, None], inverse_coms]
        P_valid = np.outer(degrees_valid, degrees_valid) * Omega_expanded
        P[np.ix_(valid_mask, valid_mask)] = P_valid

    # 4. Résidu signé réel
    R_matrix = A - P
    
    # 5. Apprentissage géométrique PyTorch via ton échantillonnage multinomial signé
    embedding_matrix = train_custom_signed_embedding(
        R_matrix=R_matrix, 
        embedding_dim=emb_dim, 
        epochs=epochs, 
        lr=lr, 
        temperature=temperature
    )
    
    # 6. Sauvegarde des attributs
    for i, node_id in enumerate(nodes):
        G_train.nodes[node_id][attr_com] = int(final_com_labels[i])
        G_train.nodes[node_id][attr_pos] = embedding_matrix[i]
        
    print(f"Outlier Louvain + SiNE terminé en {time.time() - start_time:.2f}s")
    return G_train
    

##############################################
## FONCTIONS POUR VALIDATION DE COMMUNAUTES ##
##############################################

def _find_best_partition(G, partition_func, K_min=3, min_edge_ratio=0.01, resolutions=None, **kwargs):
    """
    Explore les résolutions de manière bidirectionnelle à partir du pivot physique 1.0.
    S'arrête dès qu'une partition robuste (K_min) est trouvée.
    """
    null_model = kwargs.get('null_model', None)
    sig = inspect.signature(partition_func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    if null_model is not None and 'null_model' not in filtered_kwargs:
        filtered_kwargs['null_model'] = null_model

    # Génération d'une séquence de résolutions alternée à partir de 1.0 : 
    #[1.0, 1.2, 0.83, 1.44, 0.69, 1.73, 0.58, 2.07, 0.48]
    if resolutions is None:
        resolutions = [1.0]
        res_up = 1.0
        res_down = 1.0
        for _ in range(5):
            res_up *= 1.2
            res_down /= 1.2
            resolutions.append(round(res_up, 2))
            resolutions.append(round(res_down, 2))

    best_overall_partition = None
    best_res = 1.0
    
    for res in resolutions:
        # Sécurité : Louvain n'accepte pas les résolutions négatives ou nulles
        if res <= 0:
            continue
            
        communities_raw = partition_func(G, resolution=res, **filtered_kwargs)
        
        if isinstance(communities_raw, dict):
            partition_dict = communities_raw.copy()
        else:
            partition_dict = {}
            for i, community in enumerate(communities_raw):
                for node in community:
                    partition_dict[node] = i

        num_commus = len(set(partition_dict.values()))
        print(f"RES LOGS - ({num_commus} commus inférées pour res = {res:.2f})")
        
        # Sauvegarde par défaut (sur le premier élément de la liste, donc 1.0)
        if best_overall_partition is None:
            best_overall_partition = partition_dict.copy()
            best_res = res

        # Dès qu'une résolution (qu'elle soit plus haute ou plus basse) offre une partition robuste, on valide
        if is_partition_robust(G, partition_dict, K_min=K_min, min_edge_ratio=min_edge_ratio):
            best_overall_partition = partition_dict.copy()
            best_res = res
            print(f" Structure robuste trouvée à res = {best_res:.2f}")
            return best_overall_partition

    print(f"Attention : Aucun niveau de résolution n'a satisfait K_min={K_min}.")
    print(f"Retour de la partition par défaut (res = {best_res:.2f})")
    return best_overall_partition

def is_partition_robust(G, partition_dict, K_min=3, min_edge_ratio=0.01):
    """
    Vérifie si la partition contient au moins K_min communautés 'significatives' en termes de nombre de liens internes (%age du nb de liens totaux du graphe)
    """
    community_edge_counts = {}
    total_edges = G.number_of_edges()
    min_edges = total_edges * min_edge_ratio
    
    for comm_id in set(partition_dict.values()):
        community_edge_counts[comm_id] = 0
        
    for u, v in G.edges():
        if partition_dict[u] == partition_dict[v]:
            community_edge_counts[partition_dict[u]] += 1
            
    robust_commus = [count for count in community_edge_counts.values() if count >= min_edges]
    
    return len(robust_commus) >= K_min