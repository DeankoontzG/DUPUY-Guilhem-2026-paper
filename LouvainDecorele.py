from .MetaLouvain import best_partition, standardized_residual_best_partition
from .NullModelsInference import get_gravity_null_model_manual_iterative

import inspect
import networkx as nx
import numpy as np


def _appendSpatialLouvainCommunities(G_train, pos_attr="GT_pos", attr_name = "spatial_louvain_id", K_min=3, min_edge_ratio=0.01):
    
    P, nodes = get_gravity_null_model_manual_iterative(G_train, pos_attr)
    P_symetric = (P + P.T) / 2

    asymmetry_sum = np.sum(np.abs(P - P_symetric))
    max_diff = np.max(np.abs(P - P_symetric))

    print(f"--- ANALYSE DE L'ASYMÉTRIE ---")
    print(f"Somme de la valeur absolue des différences (|B_avant - B_après|) : {asymmetry_sum:.2e}")
    print(f"Écart maximal ponctuel : {max_diff:.2e}")

    mapping = {node: i for i, node in enumerate(nodes)}

    def my_matrix_null_model(u, v):
        idx_u = mapping[u]
        idx_v = mapping[v]
        return P_symetric[idx_u, idx_v]

    # Appel de l'algorithme développé dans MetaLouvain.py, dans la loop qui cherche la best partition
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