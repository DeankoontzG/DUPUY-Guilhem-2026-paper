from .NullModelsInference import get_gravity_null_model_manual_iterative

import random
import time
from collections import defaultdict
import networkx as nx
import numpy as np


def pair_residual_table(G, null_model=None, nodelist=None):
    """
    Build pair arrays for B_ij = A_ij - P_ij.
        Parameters
        ----------
        G : networkx.Graph
            Undirected graph.
        null_model : callable or None
            Function null_model(u, v) returning the expected edge weight/probability.
            If None, P_ij is zero.
        nodelist : list or None
            Node order. If None, uses list(G.nodes()).

        Returns
        -------
        nodes, left, right, residual
            `left`, `right`, and `residual` are numpy arrays over unordered pairs.
    """
    nodes = list(G.nodes()) if nodelist is None else list(nodelist)
    left = []
    right = []
    residual = []
    for i, u in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            observed = 1.0 if G.has_edge(u, v) else 0.0
            expected = 0.0 if null_model is None else float(null_model(u, v))
            left.append(i)
            right.append(j)
            residual.append(observed - expected)
    return (
        nodes,
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(residual, dtype=np.float32),
    )



def assortative_gt_sbm_null_model(G, com_attr="com"):
    """Assortative SBM null model from ground-truth communities.

    Internal density is estimated separately for each community. A single
    external density is shared by all inter-community pairs.
    """
    nodes = list(G.nodes())
    labels = {node: int(G.nodes[node][com_attr]) for node in nodes}
    communities = sorted(set(labels.values()))

    possible_internal = defaultdict(int)
    observed_internal = defaultdict(int)
    possible_external = 0
    observed_external = 0

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if labels[u] == labels[v]:
                possible_internal[labels[u]] += 1
            else:
                possible_external += 1

    for u, v in G.edges():
        if labels[u] == labels[v]:
            observed_internal[labels[u]] += 1
        else:
            observed_external += 1

    internal_probability = {
        community: observed_internal[community] / possible_internal[community]
        for community in communities
        if possible_internal[community] > 0
    }
    external_probability = observed_external / possible_external if possible_external else 0.0

    def null_model(u, v):
        if u == v:
            return 0.0
        if labels[u] == labels[v]:
            return internal_probability.get(labels[u], 0.0)
        return external_probability

    diagnostics = {
        "n_blocks": len(communities),
        "external_probability": float(external_probability),
        "mean_internal_probability": (
            float(np.mean(list(internal_probability.values()))) if internal_probability else 0.0
        ),
        "n_edges": G.number_of_edges(),
    }
    return null_model, diagnostics


def residual_deterrence_embedding(
    G,
    null_model=None,
    dim=2,
    tau=2,
    epochs=800,
    lr=0.03,
    pair_batch_size=20000,
    n_restarts=3,
    seed=0,
    init="random",
    init_attr=None,
    attr_name=None,
    verbose=True,
):
    """Optimize latent positions against a residual modularity objective.

    The optimized objective is approximately

        max_z sum_ij (A_ij - P_ij) exp(-||z_i - z_j|| / tau)

    where positions are centered and standardized during optimization. Positive
    residual pairs are pulled together; negative residual pairs are pushed apart
    with a bounded, saturating effect.

    This is a continuous relaxation, not a combinatorial search over node
    orderings. The pair sums are evaluated by chunks, so the function is usable
    on the synthetic graphs in this repository without materializing a dense
    torch matrix.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "residual_deterrence_embedding requires torch. Install with `python -m pip install torch`."
        ) from exc

    if dim < 1:
        raise ValueError("dim must be >= 1")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    nodes, left_np, right_np, residual_np = pair_residual_table(G, null_model=null_model)
    n = len(nodes)
    if n == 0:
        return np.zeros((0, dim)), nodes, {"best_objective": 0.0}

    abs_mass = float(np.abs(residual_np).sum())
    if abs_mass == 0:
        embedding = np.zeros((n, dim), dtype=float)
        if attr_name is not None:
            nx.set_node_attributes(G, {node: embedding[i] for i, node in enumerate(nodes)}, attr_name)
        return embedding, nodes, {"best_objective": 0.0, "abs_residual_mass": 0.0}

    device = torch.device("cpu")
    left = torch.as_tensor(left_np, dtype=torch.long, device=device)
    right = torch.as_tensor(right_np, dtype=torch.long, device=device)
    residual = torch.as_tensor(residual_np, dtype=torch.float32, device=device)
    normalization = torch.as_tensor(abs_mass, dtype=torch.float32, device=device)

    def normalized_positions(raw_z):
        z = raw_z - raw_z.mean(dim=0, keepdim=True)
        scale = z.std(dim=0, keepdim=True).clamp_min(1e-6)
        return z / scale

    def objective(raw_z):
        z = normalized_positions(raw_z)
        total = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, len(residual_np), pair_batch_size):
            stop = min(start + pair_batch_size, len(residual_np))
            diff = z[left[start:stop]] - z[right[start:stop]]
            dist = torch.linalg.norm(diff, dim=1)
            deterrence = torch.exp(-dist / tau)
            total = total + torch.sum(residual[start:stop] * deterrence)
        return total / normalization

    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    best_embedding = None
    best_objective = -float("inf")
    history = []

    for restart in range(n_restarts):
        if init == "attr" and init_attr is not None:
            initial = np.asarray([G.nodes[node][init_attr] for node in nodes], dtype=float)
            initial = np.atleast_2d(initial)
            if initial.shape[0] != n:
                initial = initial.T
            if initial.shape[1] < dim:
                pad = rng.normal(scale=0.01, size=(n, dim - initial.shape[1]))
                initial = np.column_stack([initial, pad])
            initial = initial[:, :dim]
            initial = initial + rng.normal(scale=0.01 * (restart + 1), size=initial.shape)
        else:
            initial = rng.normal(size=(n, dim))

        raw_z = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW([raw_z], lr=lr, weight_decay=0.0)

        restart_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            score = objective(raw_z)
            loss = -score
            loss.backward()
            optimizer.step()

            current = float(score.detach().cpu())
            restart_history.append(current)
            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                # Correction des guillemets ici
                print(f"restart {restart:02d} epoch {epoch + 1:04d} objective={current:.6f}")

        final_score = restart_history[-1]
        history.append(restart_history)
        if final_score > best_objective:
            best_objective = final_score
            best_embedding = normalized_positions(raw_z).detach().cpu().numpy()

    diagnostics = {
        "best_objective": float(best_objective),
        "abs_residual_mass": abs_mass,
        "n_pairs": int(len(residual_np)),
        "positive_pair_fraction": float(np.mean(residual_np > 0)),
        "tau": float(tau),
        "epochs": int(epochs),
        "n_restarts": int(n_restarts),
        "history": history,
    }

    if attr_name is not None:
        nx.set_node_attributes(G, {node: best_embedding[i] for i, node in enumerate(nodes)}, attr_name)

    return best_embedding, nodes, diagnostics


def _append_residual_deterrence_embedding(G_train, pos_attr="GT_pos", attr_name="ResDeterEmb", NullModel_method="ManualIter", dim=64):
    """
    Calcule l'embedding de dissuasion des résidus (Residual Deterrence) en se basant
    sur un modèle nul spatial Degré Corrigé.
    """
    print(f"Calcul de Residual Deterrence emb (NullModel type = {NullModel_method})...")
    start_skip = time.time()

    nodes = list(G_train.nodes())
    
    if NullModel_method == "ManualIter":
        P, _ = get_gravity_null_model_manual_iterative(G_train, pos_attr)
        P_symetric = (P + P.T) / 2
    else:
        raise ValueError(f"NullModel_method '{NullModel_method}' non reconnue.")

    mapping = {node: i for i, node in enumerate(nodes)}

    def my_matrix_null_model(u, v):
        idx_u = mapping[u]
        idx_v = mapping[v]
        return P_symetric[idx_u, idx_v]

    # Appel de la fonction principale du fichier
    embedding, _, diagnostics = residual_deterrence_embedding(
        G_train,
        null_model=my_matrix_null_model,
        attr_name=attr_name,
        dim=dim,
    )

    end_skip = time.time()
    duration = end_skip - start_skip
    print(f"ResDeterEmb terminé en {duration:.2f}s")
    print(f"-> Succès : {embedding.shape[1]} dimensions ajoutées à l'attribut '{attr_name}' de chaque nœud.")
    
    return G_train