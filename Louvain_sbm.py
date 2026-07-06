from MetaLouvain import standardized_residual_best_partition

import networkx as nx
import numpy as np
import inspect


def compute_dcsbm_null_model(A, com_labels):
    """
    Calcule la matrice de modèle nul théorique DC-SBM (Degree-Corrected Stochastic Block Model)
    à partir d'une matrice d'adjacence A et d'un vecteur de labels de communautés.
    
    Formulation analytique exacte préservant la symétrie.
    """
    N = A.shape[0]
    degrees = np.sum(A, axis=1)
    total_volume = np.sum(degrees)
    
    if total_volume == 0:
        return np.zeros((N, N))
        
    # Identification et numérotation des blocs uniques de 0 à K-1
    unique_coms, inverse_coms = np.unique(com_labels, return_inverse=True)
    K = len(unique_coms)
    
    # 1. Matrice d'arêtes inter-blocs M (K, K) et volumes des blocs kappa (K,)
    M = np.zeros((K, K))
    kappa = np.zeros(K)
    
    for r in range(K):
        mask_r = (inverse_coms == r)
        kappa[r] = np.sum(degrees[mask_r])
        for s in range(K):
            mask_s = (inverse_coms == s)
            # Somme des arêtes entre le bloc r et le bloc s
            M[r, s] = np.sum(A[mask_r][:, mask_s])
            
    # 2. Calcul des probabilités de connexion normalisées par bloc : Omega_rs
    # Évite les divisions par zéro si un bloc est complètement isolé
    with np.errstate(divide='ignore', invalid='ignore'):
        # P(r, s) attendu sans correction de degré individuel
        Omega = M / (np.outer(kappa, kappa) + 1e-12)
        
    # 3. Expansion matricielle et application de la correction de degré de chaque nœud
    # P_ij = k_i * k_j * Omega[com_i, com_j]
    # Pour vectoriser proprement, on extrait les lignes et colonnes de la matrice Omega
    Omega_expanded = Omega[inverse_coms[:, None], inverse_coms]
    
    # Produit externe des degrés : (N, 1) x (1, N) -> (N, N)
    P = np.outer(degrees, degrees) * Omega_expanded
    
    # Masquage optionnel des auto-boucles si ton graphe d'origine n'en contient pas
    np.fill_diagonal(P, 0.0)
    
    return P

def _appendSbmLouvainCommunities(G_train, sbm_attr="GT_sbm_id", attr_name="sbm_louvain_id", K_min=3, min_edge_ratio=0.01):
    """
    Détecte les communautés Louvain décorrélées des groupes SBM réels (GT_sbm_id).
    Utilise le modèle nul DC-SBM (Degree-Corrected Stochastic Block Model) pour intégrer 
    la topologie induite par les blocs existants.
    """
    print(f"Calcul de Louvain décorrélé du SBM (attribut source : {sbm_attr})...")
    
    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    
    # 2. Récupération des labels SBM dans le même ordre que la matrice d'adjacence
    try:
        com_labels = [G_train.nodes[node][sbm_attr] for node in nodes]
    except KeyError as e:
        print(f"Erreur : L'attribut SBM '{sbm_attr}' est manquant pour certains nœuds.")
        raise e

    # 3. Calcul du modèle nul théorique DC-SBM
    P = compute_dcsbm_null_model(A, com_labels)
    
    # Le modèle analytique DC-SBM pour graphes non-orientés est intrinsèquement symétrique, 
    # mais on applique la symétrisation par précaution numérique.
    P_symmetric = (P + P.T) / 2

    asymmetry_sum = np.sum(np.abs(P - P_symmetric))
    max_diff = np.max(np.abs(P - P_symmetric))

    print(f"--- ANALYSE DE L'ASYMÉTRIE ---")
    print(f"Somme de la valeur absolue des différences (|P - P_sym|) : {asymmetry_sum:.2e}")
    print(f"Écart maximal ponctuel : {max_diff:.2e}")
    print(f"------------------------------")

    # 4. Définition de la fonction de mapping pour l'évaluation élémentaire du modèle nul
    def my_matrix_null_model(u, v):
        idx_u = mapping[u]
        idx_v = mapping[v]
        return P_symmetric[idx_u, idx_v]

    # 5. Recherche de la meilleure partition via le modèle nul DC-SBM personnalisé
    partition = _find_best_partition(
        G_train, 
        standardized_residual_best_partition, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio,
        null_model=my_matrix_null_model
    )
    
    print("--- Diagnostic de l'objet partition ---")
    print(f"Nombre de nœuds assignés : {len(partition)}")
    print(f"Nombre de communautés trouvées : {len(set(partition.values()))}")
    print("---------------------------------------")

    nx.set_node_attributes(G_train, partition, attr_name)
    
    return G_train


###################################################################
#### 2 Fonctions utilitaire : pour trouver une bonne partition ####
###################################################################

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