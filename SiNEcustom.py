from NullModelsInference import get_gravity_null_model_manual_iterative

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
        if pairs.shape[0] == 0:
            return z.sum() * 0.0
            
        anchors = z[pairs[:, 0]]
        neighbors = z[pairs[:, 1]]

        distances = torch.norm(anchors - neighbors, p=2, dim=1)
        
        loss_pos = distances
        loss_neg = torch.clamp(self.margin - distances, min=0.0)

        # Transformation des signes {-1, 1} vers {0, 1}
        is_pos = (signs + 1) / 2.0
        total_loss = is_pos * loss_pos + (1.0 - is_pos) * loss_neg

        return total_loss.mean()

def generate_pairwise_samples(R_tensor, num_samples_per_node=15, temperature=0.5):
    """
    R : Matrice d'adjacence ou de résidus (N, N)
    """
    N = R_tensor.shape[0]
    device = R_tensor.device
    
    abs_residues = torch.abs(R_tensor)
    
    # Masquage de la diagonale (auto-échantillonnage)
    abs_residues.diagonal().fill_(0.0)
    
    all_pairs = []
    all_signs = []
    
    for i in range(N):
        row = abs_residues[i]
        valid_mask = row > 1e-6
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # Nœud isolé : probabilités uniformes sur tous les autres nœuds
            probs = torch.ones(N, device=device)
            probs[i] = 0.0
            probs /= probs.sum()
            eff_num_samples = min(num_samples_per_node, N - 1)
        else:
            # Softmax ou normalisation directe. 
            # Si les scores sont très petits, attention à l'échelle avant Softmax
            sub_scores = row[valid_indices] / temperature
            sub_probs = F.softmax(sub_scores, dim=0)
            
            probs = torch.zeros(N, device=device)
            probs[valid_indices] = sub_probs
            eff_num_samples = min(num_samples_per_node, len(valid_indices))
        
        if eff_num_samples > 0:
            # Échantillonnage sans remise via multinomiale
            sampled_nodes = torch.multinomial(probs, num_samples=eff_num_samples, replacement=False)
            
            # Construction des paires
            src = torch.full((eff_num_samples,), i, dtype=torch.long, device=device)
            pairs = torch.stack([src, sampled_nodes], dim=1)
            
            # Signes correspondants
            sampled_signs = torch.where(R_tensor[i, sampled_nodes] >= 0, 1.0, -1.0)
            
            all_pairs.append(pairs)
            all_signs.append(sampled_signs)
            
    if not all_pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device), torch.empty((0,), device=device)
        
    return torch.cat(all_pairs, dim=0), torch.cat(all_signs, dim=0)

def train_custom_signed_embedding(R_matrix, embedding_dim=64, epochs=100, lr=0.01, temperature=0.5):
    N = R_matrix.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    R_tensor = torch.from_numpy(R_matrix).float().to(device)
    node_embeddings = torch.nn.Parameter(torch.randn(N, embedding_dim, device=device) * 0.1)
    
    optimizer = torch.optim.Adam([node_embeddings], lr=lr)
    loss_fn = PairwiseSignLoss(margin=2.0)
    
    for epoch in range(epochs):
        pairs, signs = generate_pairwise_samples(R_tensor, num_samples_per_node=15, temperature=temperature)
        
        optimizer.zero_grad()
        loss = loss_fn(node_embeddings, pairs, signs)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return node_embeddings.detach().cpu().numpy()

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