import time
import torch
import numpy as np
import networkx as nx

from SiNEcustom import train_custom_signed_embedding

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

def _append_SineSBMcustom(G_train, com_attr="spatial_louvain_id", attr_name="SBMcustom", temperature=0.5, emb_dim=64, epochs=100, lr=0.1):
    """
    Calcule les embeddings décorrélés des structures communautaires macroscopiques.
    Utilise un modèle nul DC-SBM pour extraire la matrice de résidus.
    """
    print(f"Calcul de SBM custom (Attribut communauté = {com_attr})...")
    start_time = time.time()

    # Extraction de la matrice d'adjacence et des métadonnées
    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    
    # Récupération des ID de communautés pour chaque nœud dans l'ordre de la matrice
    try:
        com_labels = np.array([G_train.nodes[node][com_attr] for node in nodes])
    except KeyError:
        raise KeyError(f"L'attribut de communauté '{com_attr}' est manquant sur certains nœuds du graphe.")

    # Inférence du modèle nul DC-SBM
    P = compute_dcsbm_null_model(A, com_labels)
    
    # Symétrisation par sécurité (l'implémentation ci-dessus est nativement symétrique pour un graphe non-orienté)
    P_symmetric = (P + P.T) / 2
    R_matrix = A - P_symmetric

    # Diagnostics de structure
    asymmetry_sum = np.sum(np.abs(P - P_symmetric))
    max_diff = np.max(np.abs(P - P_symmetric))
    print(f"--- ANALYSE DE L'ASYMÉTRIE du modèle DC-SBM ---")
    print(f"Somme de la valeur absolue des différences : {asymmetry_sum:.2e}")
    print(f"Écart maximal ponctuel : {max_diff:.2e}")
    print(f"Moyenne des résidus : {np.mean(R_matrix):.4f} | Min : {np.min(R_matrix):.4f} | Max : {np.max(R_matrix):.4f}")

    # Apprentissage des représentations signées via SiNE (réutilisation de tes fonctions d'origine)
    embedding_matrix = train_custom_signed_embedding(
        R_matrix=R_matrix, 
        embedding_dim=emb_dim, 
        epochs=epochs, 
        lr=lr, 
        temperature=temperature
    )
    
    # Stockage des embeddings sous forme de dictionnaire d'attributs
    embeddings_dict = {node_id: embedding_matrix[i] for i, node_id in enumerate(nodes)}
    nx.set_node_attributes(G_train, embeddings_dict, attr_name)

    duration = time.time() - start_time
    print(f"SBMcustom terminé en {duration:.2f}s")
    print(f"-> Succès : {embedding_matrix.shape[1]} dimensions ajoutées à l'attribut '{attr_name}'.\n")
    
    return G_train