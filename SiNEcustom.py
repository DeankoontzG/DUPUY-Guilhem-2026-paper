from .NullModelsInference import get_gravity_null_model_manual_iterative

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


########################################
#### METHODE SiNE CUSTOM AC PAIRES #####
########################################

class PairwiseSignLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, pairs, signs):
        """
        z : Embeddings des noeuds (N, embedding_dim)
        pairs : Tenseur (M, 2) des paires (noeud_i, noeud_j)
        signs : Tenseur (M,) contenant le signe original (+1.0 ou -1.0)
        """
        if pairs.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)
            
        anchors = z[pairs[:, 0]]
        neighbors = z[pairs[:, 1]]

        # Distance euclidienne entre les embeddings de la paire
        distances = torch.norm(anchors - neighbors, p=2, dim=1)

        # Si signe == 1  -> On minimise la distance
        loss_pos = distances
        
        # Si signe == -1 -> On veut distance >= margin
        loss_neg = torch.clamp(self.margin - distances, min=0.0)

        # Projection des signes [-1, 1] vers [0, 1] pour le masque
        is_pos = (signs + 1) / 2.0
        
        # Combinaison linéaire des pertes
        total_loss = is_pos * loss_pos + (1.0 - is_pos) * loss_neg

        return total_loss.mean()

def generate_pairwise_samples(R, num_samples_per_node=15, temperature=0.5):
    """
    R : Matrice d'adjacence ou de résidus (N, N)
    """
    pairs = []
    signs = []
    N = R.shape[0]
    
    for i in range(N):
        row = R[i]
        abs_residues = np.abs(row)
        
        # Masquage : le noeud i ne peut pas se piocher lui-même
        abs_residues_tensor = torch.tensor(abs_residues, dtype=torch.float32)
        abs_residues_tensor[i] = float('-inf')

        # Comptage du nb de voisins "valides" pour le noeud étudié (i)
        valid_candidates_mask = (abs_residues_tensor > 1e-6) & (abs_residues_tensor != float('-inf'))
        valid_indices = torch.where(valid_candidates_mask)[0].numpy()

        # GESTION DES NOEUDS ISOLÉS / COMPORTEMENT NORMAL
        if len(valid_indices) == 0:
            # Le noeud n'a aucun signal utile. On lui donne une probabilité uniforme 
            # sur TOUS les autres noeuds du graphe (pour éviter de bloquer l'algo)
            probs = np.ones(N) / (N - 1)
            probs[i] = 0.0
            # Le pool de choix devient tous les noeuds sauf i
            pool_to_sample = np.delete(np.arange(N), i)
        else:
            # Le noeud a du signal. On applique le Softmax UNIQUEMENT sur les candidats valides
            sub_scores = abs_residues_tensor[valid_indices]
            sub_probs = F.softmax(sub_scores / temperature, dim=0).numpy()
            
            # On reconstruit un vecteur de probabilité de taille N
            probs = np.zeros(N)
            probs[valid_indices] = sub_probs
            pool_to_sample = N

        probs_sum = probs.sum()
        if probs_sum > 0:
            normalization_ratio = 1/probs_sum
            probs = probs*normalization_ratio
            if abs(1-normalization_ratio)>1.05:
                print(f"ATTTENTION :SiNE probas normalisées par un facteur de {normalization_ratio}")
            
        
       # TIRAGE : Le nombre maximum de tirages possibles sans remise est limité par notre pool de candidats réels
        available_candidates = len(valid_indices) if len(valid_indices) > 0 else (N - 1)
        eff_num_samples = min(num_samples_per_node, available_candidates)
        
        # TIRAGE ALÉATOIRE SANS REMISE
        sampled_nodes = np.random.choice(
            pool_to_sample, 
            size=eff_num_samples, 
            p=probs if isinstance(pool_to_sample, int) else None, 
            replace=False
        )
        
        # 6. ENREGISTREMENT DES PAIRES
        for j in sampled_nodes:
            pairs.append([i, j])
            # Signe réel : 1.0 pour les affinités/liens existants, -1.0 pour les inimitiés
            sign = 1.0 if R[i, j] >= 0 else -1.0
            signs.append(sign)
            
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(signs, dtype=torch.float32)

def train_custom_signed_embedding(R_matrix, embedding_dim=64, epochs=100, lr=0.01, temperature=0.5):
    N = R_matrix.shape[0]
    
    # Initialisation des embeddings libres (Paramètre PyTorch)
    # Les noeuds isolés resteront proches de leur position d'initialisation 
    # car ils ne recevront aucun gradient significatif.
    node_embeddings = torch.nn.Parameter(torch.randn(N, embedding_dim) * 0.1)
    
    optimizer = torch.optim.Adam([node_embeddings], lr=lr)
    loss_fn = PairwiseSignLoss(margin=2.0)
    
    for epoch in range(epochs):
        # 1. Échantillonnage stochastique custom
        pairs, signs = generate_pairwise_samples(R_matrix, num_samples_per_node=15, temperature=temperature)
        
        optimizer.zero_grad()
        
        # 2. Calcul de la perte sur les paires échantillonnées
        loss = loss_fn(node_embeddings, pairs, signs)
        
        # 3. Rétropropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return node_embeddings.detach().numpy()

def _append_SiNEcustom(G_train, pos_attr="GT_pos", attr_name = "SiNEcustom", NullModel_method = "ManualIter", temperature=0.5, emb_dim=64):
    print(f"Calcul de SiNE custom (NullModel type ={NullModel_method})...")
    start_skip = time.time()

    A = nx.to_numpy_array(G_train)
    nodes = list(G_train.nodes())
    
    if NullModel_method == "ManualIter":
        P, _ = get_gravity_null_model_manual_iterative(G_train, pos_attr)
        P_symetric = (P + P.T) / 2
        R_matrix = A - P_symetric

        asymmetry_sum = np.sum(np.abs(P - P_symetric))
        max_diff = np.max(np.abs(P - P_symetric))

        print(f"--- ANALYSE DE L'ASYMÉTRIE du modèle spatial pour SiNEcustom ---")
        print(f"Somme de la valeur absolue des différences (|B_avant - B_après|) : {asymmetry_sum:.2e}")
        print(f"Écart maximal ponctuel : {max_diff:.2e}")
    else:
        print(f"NullModel_method {NullModel_method} non reconnue")

    embedding_matrix = train_custom_signed_embedding(R_matrix=R_matrix, embedding_dim=emb_dim, epochs=100, lr=0.1, temperature=temperature)
    
    embeddings_dict = {}
    for i, node_id in enumerate(nodes):
        embeddings_dict[node_id] = embedding_matrix[i]

    nx.set_node_attributes(G_train, embeddings_dict, attr_name)

    end_skip = time.time()
    SiNEcustom_duration = end_skip - start_skip
    print(f"SiNEcustom terminé en {SiNEcustom_duration:.2f}s")
    print(f"-> Succès : {embedding_matrix.shape[1]} dimensions ajoutées à l'attribut '{attr_name}' de chaque nœud.")
    
    return G_train