import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np

class ContinuousMixedNetworkModel(nn.Module):
    def __init__(self, num_nodes, num_communities, embedding_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.K = num_communities
        
        # Séparation des propensions de degré
        self.raw_k_spatial = nn.Parameter(torch.ones(num_nodes) * 0.5) # k_i pour le modèle Spatial
        self.raw_k_sbm = nn.Parameter(torch.ones(num_nodes) * 0.5)     # K_i pour le modèle SBM
        
        self.raw_beta = nn.Parameter(torch.tensor(0.0)) 
        self.z = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.1)
        
        # Logits SBM
        self.h = nn.Parameter(torch.randn(num_nodes, num_communities) * 0.5)
        
        # Matrice B d'affinité communautaire
        B_init = torch.eye(num_communities) * 0.5 + 0.1
        self.raw_B = nn.Parameter(B_init)

    def forward(self, tau=1.0):
        # Contraintes de positivité strictes
        k_spatial = torch.clamp(self.raw_k_spatial, min=1e-3)
        k_sbm = torch.clamp(self.raw_k_sbm, min=1e-3)
        #beta = torch.exp(self.raw_beta)
        beta = torch.sigmoid(self.raw_beta) * 5.0
        B = (self.raw_B + self.raw_B.t()) / 2.0 
        
        pi = torch.softmax(self.h / tau, dim=1)
        
        # Calcul des distances géométriques
        dist_sq = torch.cdist(self.z, self.z, p=2) ** 2
        K_spatial = torch.exp(-beta * dist_sq)
        K_sbm = torch.matmul(torch.matmul(pi, B), pi.t())
        
        # Application asymétrique des propensions de degré
        E_spatial = torch.outer(k_spatial, k_spatial) * K_spatial
        E_sbm = torch.outer(k_sbm, k_sbm) * K_sbm
        E_total = E_spatial + E_sbm
        
        return E_total, E_spatial, E_sbm, pi, k_spatial, k_sbm, beta, dist_sq

def fit_continuous_mixed_model(G_train, K=3, epochs=300, lr=0.01):
    A_np = nx.to_numpy_array(G_train)
    A = torch.tensor(A_np, dtype=torch.float32)
    num_nodes = A.shape[0]
    mask = ~torch.eye(num_nodes, dtype=torch.bool)
    
    model = ContinuousMixedNetworkModel(num_nodes=num_nodes, num_communities=K)
    
    # =========================================================================
    # WARM-START DU SBM VIA LOUVAIN
    # =========================================================================
    try:
        # On extrait une partition rapide en K communautés (ou proche)
        from networkx.community import louvain_communities
        communities_list = list(louvain_communities(G_train))
        
        # On remplit une matrice one-hot cible (N, K)
        h_init = torch.zeros(num_nodes, K)
        for block_idx, nodes_set in enumerate(communities_list[:K]): # On se limite à K blocs
            for node in nodes_set:
                node_idx = list(G_train.nodes()).index(node)
                h_init[node_idx, block_idx] = 2.0 # On donne un poids fort au bloc cible
                
        # On injecte ces logits dans le modèle
        with torch.no_grad():
            model.h.copy_(h_init + torch.randn(num_nodes, K) * 0.2)
        print("-> Warm-start du SBM réussi via Louvain.")
    except Exception as e:
        print(f"-> Impossible de faire le warm-start ({e}), initialisation aléatoire.")
    # =========================================================================

    optimizer = optim.AdamW([
        {'params': [model.h, model.raw_B], 'lr': lr * 10},     
        {'params': [model.z, model.raw_beta], 'lr': lr * 1.0}, 
        {'params': [model.raw_k_spatial, model.raw_k_sbm], 'lr': lr * 1.0}
    ], weight_decay=0.01)
    
    print(f"Ajustement avec propensions découplées (Flexible k_i / K_i)...")
    print("=" * 125)
    print(f"{'Époque':<7} | {'MSE':<10} | {'Loss Ortho':<12} | {'Loss Proxim':<12} | {'R² Global':<11} | {'R² Spatial':<12} | {'R² SBM':<10} | {'Beta':<6}")
    print("=" * 125)
    
    total_variance = torch.var(A[mask]).item()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        tau = max(0.1, 1.0 - (0.9 * (epoch - 1) / (epochs - 1)))
        
        E_total, E_spatial, E_sbm, pi, k_spatial, k_sbm, beta, dist_sq = model(tau=tau)
        
        # 1. Perte de reconstruction
        loss_mse = torch.mean((A[mask] - E_total[mask]) ** 2)
        entropy_reg = -0.1 * torch.mean(torch.sum(pi ** 2, dim=1))
        
        # 2. Régularisation élastique de proximité entre k_spatial et k_sbm
        # Permet la flexibilité locale tout en interdisant une divergence aberrante globale
        loss_proximity = torch.mean((k_spatial - k_sbm) ** 2)
        
        # 3. Orthogonalité
        P_blocks = torch.matmul(pi, pi.t())
        D_centered = dist_sq - torch.mean(dist_sq[mask])
        P_centered = P_blocks - torch.mean(P_blocks[mask])
        covariance = torch.mean((D_centered * P_centered)[mask])
        loss_ortho = torch.abs(covariance)
        
        # Perte globale combinée (Ajustement du poids de l'orthogonalité pour libérer le SBM)
        loss = loss_mse + entropy_reg + (2.0 * loss_ortho) + (5.0 * loss_proximity) + (0.005 * model.raw_beta ** 2)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            with torch.no_grad():
                r2_total = 1 - (torch.var((A - E_total)[mask]).item() / total_variance)
                r2_spatial = 1 - (torch.var((A - E_spatial)[mask]).item() / total_variance)
                r2_sbm = 1 - (torch.var((A - E_sbm)[mask]).item() / total_variance)
                
                pct_tot = f"{max(0, r2_total * 100):.1f}%"
                pct_spat = f"{max(0, r2_spatial * 100):.1f}%"
                pct_sbm = f"{max(0, r2_sbm * 100):.1f}%"
                
                print(f"{epoch:<7} | {loss_mse.item():<10.4f} | {loss_ortho.item():<12.4f} | {loss_proximity.item():<12.4f} | {pct_tot:<11} | {pct_spat:<12} | {pct_sbm:<10} | {beta.item():<6.2f}")
                
    print("=" * 125)
    return model

def _appendContinuousMixedMetrics(G_train, K=3, epochs=1000, lr=0.01):
    trained_model = fit_continuous_mixed_model(G_train, K=K, epochs=epochs, lr=lr)
    
    with torch.no_grad():
        _, _, _, pi, k_spatial, k_sbm, beta, _ = trained_model(tau=0.01) 
        
        positions = trained_model.z.cpu().numpy()
        poids_k_spatial = k_spatial.cpu().numpy()
        poids_k_sbm = k_sbm.cpu().numpy()
        commu_ids = torch.argmax(pi, dim=1).cpu().numpy()
        
    spatial_positions_dict = {node: positions[i] for i, node in enumerate(G_train.nodes())}
    community_assignment_dict = {node: int(commu_ids[i]) for i, node in enumerate(G_train.nodes())}
    propensity_spatial_dict = {node: float(poids_k_spatial[i]) for i, node in enumerate(G_train.nodes())}
    propensity_sbm_dict = {node: float(poids_k_sbm[i]) for i, node in enumerate(G_train.nodes())}
    
    nx.set_node_attributes(G_train, spatial_positions_dict, "gd_embedding")
    nx.set_node_attributes(G_train, community_assignment_dict, "gd_sbm_id")
    nx.set_node_attributes(G_train, propensity_spatial_dict, "latent_propensity_spatial")
    nx.set_node_attributes(G_train, propensity_sbm_dict, "latent_propensity_sbm")
    
    G_train.graph["spatial_friction_beta"] = float(beta.item())
    
    return G_train