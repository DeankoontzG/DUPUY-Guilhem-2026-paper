import time
import networkx as nx
import numpy as np
import inspect

from SiNEcustom import train_custom_signed_embedding

def robust_pca_graph_decomposition(A, lmbda=None, max_iter=200, tol=1e-5):
    """Décomposition Robuste en Composantes Principales (RPCA) via ADMM."""
    N = A.shape[0]
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(N)
    L = np.zeros((N, N))
    S = np.zeros((N, N))
    Y = np.zeros((N, N))
    mu = 1.25 / np.linalg.norm(A, 2)
    rho = 1.5
    for it in range(max_iter):
        U, s, Vt = np.linalg.svd(A - S + (Y / mu), full_matrices=False)
        s_th = np.maximum(s - (1.0 / mu), 0)
        L_next = np.dot(U * s_th, Vt)
        X_S = A - L_next + (Y / mu)
        S_next = np.sign(X_S) * np.maximum(np.abs(X_S) - (lmbda / mu), 0)
        err = np.linalg.norm(A - L_next - S_next, 'fro') / np.linalg.norm(A, 'fro')
        L = L_next
        S = S_next
        if err < tol:
            break
        Y = Y + mu * (A - L - S)
        mu = mu * rho
    return (L + L.T) / 2, (S + S.T) / 2

def compute_dcsbm_null_model(A, com_labels):
    """Calcule analytiquement le modèle nul théorique DC-SBM."""
    N = A.shape[0]
    degrees = np.sum(A, axis=1)
    total_volume = np.sum(degrees)
    if total_volume == 0:
        return np.zeros((N, N))
    unique_coms, inverse_coms = np.unique(com_labels, return_inverse=True)
    K = len(unique_coms)
    M = np.zeros((K, K))
    kappa = np.zeros(K)
    for r in range(K):
        mask_r = (inverse_coms == r)
        kappa[r] = np.sum(degrees[mask_r])
        for s in range(K):
            mask_s = (inverse_coms == s)
            M[r, s] = np.sum(A[mask_r][:, mask_s])
    with np.errstate(divide='ignore', invalid='ignore'):
        Omega = M / (np.outer(kappa, kappa) + 1e-12)
    Omega_expanded = Omega[inverse_coms[:, None], inverse_coms]
    P = np.outer(degrees, degrees) * Omega_expanded
    np.fill_diagonal(P, 0.0)
    return P

def _append_RPCA_SiNEcustom(G_train, attr_com="rpca_com_id", attr_pos="rpca_pos", emb_dim=64, temperature=0.5, epochs=100, lr=0.1):
    print("Calcul de la décomposition RPCA suivie de SiNE custom sur résidus SBM...")
    start_time = time.time()
    
    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    
    # 1. Séparation par rang faible pour isoler les communautés
    L, _ = robust_pca_graph_decomposition(A)

    G_L_raw = nx.from_numpy_array(np.clip(L, 0, None))
    mapping = {i: node for i, node in enumerate(nodes)}
    G_L = nx.relabel_nodes(G_L_raw, mapping)
    
    best_p = _find_best_partition(
        G_L, 
        nx.community.louvain_communities, 
        K_min=3, 
        min_edge_ratio=0.01
    )
    com_labels = np.array([best_p[node] for node in nodes])
    
    # 2. Calcul du modèle nul basé sur les blocs trouvés
    P = compute_dcsbm_null_model(A, com_labels)
    R_matrix = A - P
    
    # 3. Entraînement de ton embedding signé (SiNE) sur le résidu réel
    embedding_matrix = train_custom_signed_embedding(
        R_matrix=R_matrix, 
        embedding_dim=emb_dim, 
        epochs=epochs, 
        lr=lr, 
        temperature=temperature
    )
    
    # 4. Stockage des attributs
    for i, node_id in enumerate(nodes):
        G_train.nodes[node_id][attr_com] = int(com_labels[i])
        G_train.nodes[node_id][attr_pos] = embedding_matrix[i]
        
    print(f"RPCA + SiNE terminé en {time.time() - start_time:.2f}s")
    return G_train

def _append_RPCA_DirectSiNEcustom(G_train, attr_com="rpca_com_id", attr_pos="rpca_pos", emb_dim=64, temperature=0.5, epochs=100, lr=0.1):
    print("Calcul de la décomposition RPCA suivie de SiNE direct sur la composante Sparse S...")
    start_time = time.time()
    
    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    
    # 1. Séparation RPCA
    L, S = robust_pca_graph_decomposition(A)

    # 2. Extraction des communautés sur L (comme avant)
    G_L_raw = nx.from_numpy_array(np.clip(L, 0, None))
    mapping = {i: node for i, node in enumerate(nodes)}
    G_L = nx.relabel_nodes(G_L_raw, mapping)
    
    best_p = _find_best_partition(
        G_L, 
        nx.community.louvain_communities, 
        K_min=3, 
        min_edge_ratio=0.01
    )
    com_labels = np.array([best_p[node] for node in nodes])
    
    # 3. Utilisation DIRECTE de S comme matrice de résidus signés
    # S contient déjà les écarts positifs et négatifs par rapport au rang faible.
    # On force la diagonale à 0 pour éviter l'auto-échantillonnage.
    R_matrix = S.copy()
    np.fill_diagonal(R_matrix, 0.0)
    
    # 4. Entraînement de SiNE custom directement sur ce signal épuré
    embedding_matrix = train_custom_signed_embedding(
        R_matrix=R_matrix, 
        embedding_dim=emb_dim, 
        epochs=epochs, 
        lr=lr, 
        temperature=temperature
    )
    
    # 5. Stockage des attributs (sans .tolist() pour éviter le bug de reshape)
    for i, node_id in enumerate(nodes):
        G_train.nodes[node_id][attr_com] = int(com_labels[i])
        G_train.nodes[node_id][attr_pos] = embedding_matrix[i]
        
    print(f"RPCA + DirectSiNE terminé en {time.time() - start_time:.2f}s")
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