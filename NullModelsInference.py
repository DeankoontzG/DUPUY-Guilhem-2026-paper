import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar

def get_gravity_null_model_manual_iterative(G, pos_attr='pos', tol=0.01, max_iter=1000):
    """
    Inférence par maximisation de la vraisemblance (descente de coordonnées et optim scalaire)
    du modèle gravitaire / ERGM géométrique non dirigé.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    adj = nx.to_numpy_array(G)
    degrees = np.sum(adj, axis=1)
    
    # Matrice de distance (N, N)
    pos_array = np.array([G.nodes[u][pos_attr] for u in nodes])
    if len(pos_array.shape) == 1 or (len(pos_array.shape) == 2 and pos_array.shape[1] == 1):
        # Cas 1D : Différence absolue directe
        dist_matrix = np.abs(pos_array[:, np.newaxis] - pos_array[np.newaxis, :])
    else:
        dist_matrix = np.linalg.norm(pos_array[:, np.newaxis] - pos_array[np.newaxis, :], axis=2)
    
    # Initialisation des paramètres
    alphas = np.zeros(n)
    beta = 1.0
    
    def total_log_likelihood_beta(b, current_alphas):
        """ Log-vraisemblance négative pour l'optimisation de beta """
        theta = current_alphas[:, np.newaxis] + current_alphas[np.newaxis, :] - b * dist_matrix
        log_q = np.logaddexp(0, theta)
        # Triangle supérieur (réseau non dirigé, pas de boucles)
        ll = np.sum(np.triu(adj * theta - log_q, k=1))
        return -ll

    for iteration in range(max_iter):
        
        # 1. Mise à jour séquentielle des alphas (Descente de coordonnées vectorisée)
        for _ in range(5):
            theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
            theta = np.clip(theta, -50, 50)  # Stabilité numérique
            
            P = 1 / (1 + np.exp(-theta))
            np.fill_diagonal(P, 0)
            
            f_x = np.sum(P, axis=1) - degrees
            f_prime = np.sum(P * (1 - P), axis=1) + 1e-5  # Approximation de la Hessienne
            
            step = f_x / f_prime
            alphas -= 0.5 * np.clip(step, -2.0, 2.0)
            
        # 2. Mise à jour de beta
        res_beta = minimize_scalar(
            total_log_likelihood_beta, 
            args=(alphas,), 
            bounds=(0, 20), 
            method='bounded'
        )
        beta = res_beta.x
        
        # Évaluation de la convergence
        theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
        current_P = 1 / (1 + np.exp(-theta))
        np.fill_diagonal(current_P, 0)
        
        predicted_degrees = np.sum(current_P, axis=1)
        mae_degrees = np.mean(np.abs(predicted_degrees - degrees))
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: MAE = {mae_degrees:.6f}, Beta = {beta:.4f}")
        
        if mae_degrees < tol:
            break

    print(f"Modèle gravitaire inféré. MAE = {mae_degrees:.4f}, alpha moy = {np.mean(alphas):.4f}, beta = {beta:.4f}")  
    
    A_sum = len(G.edges())
    normalization_factor = 2 * A_sum / current_P.sum()
    print(f"Vérification : Null Model donne 2*E / P.sum = {normalization_factor:.4f}")
    
    return current_P, nodes