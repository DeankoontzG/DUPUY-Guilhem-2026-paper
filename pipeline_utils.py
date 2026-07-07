from LouvainDecorele import _appendSpatialLouvainCommunities
from SiNEcustom import _append_SiNEcustom

from Louvain_sbm import _appendSbmLouvainCommunities
from DeepWalk_on_residuals import _append_DeepwalkOnResiduals_full
from SiNE_on_sbm_residuals import _append_SineSBMcustom

import os
import gc
import time
import random
import json
import html
import io
import inspect
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from node2vec import Node2Vec
import joblib
from joblib import Parallel, delayed
from xgboost import XGBClassifier
import optuna
import multiprocessing

###############################################################
## CONTANTES, DONT MAPPING VERS ALGOS DE CALCUL DE MÉTRIQUES ##
###############################################################
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

"""
EMBEDDINGS = ["deepwalk", "SiNEcustom_spatial", "SiNEcustom_sbm", "deepwalk_residuals"]
COMMUNITY_ALGOS = ['louvain', 'spatial_louvain', "sbm_louvain"]
"""
EMBEDDINGS = ["deepwalk", "SiNEcustom_spatial"]
COMMUNITY_ALGOS = ['louvain', 'spatial_louvain']

#################################################
# FONCTIONS DE VALIDATION DES DONNES EN ENTREE ##
#################################################

def validate_input_graph(G, min_nodes=2, min_edges=1, require_undirected=True):
    """
    Vérifie la validité du graphe d'entrée avant les calculs de prédiction de liens.
    """
    # 1. Vérification du type de base
    if not isinstance(G, nx.Graph):
        raise TypeError(
            f"L'entrée doit être un objet networkx.Graph. Reçu: {type(G)}. "
            "Pour d'autres formats, convertissez-les d'abord avec networkx."
        )

    if require_undirected and G.is_directed():
        raise ValueError(
            "Le graphe est dirigé (DiGraph). L'algorithme actuel supporte uniquement "
            "les graphes non-dirigés pour garantir la validité des métriques topo."
        )

    # 3. Vérification de la taille
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < min_nodes:
        raise ValueError(f"Graphe trop petit: {n_nodes} nœuds (minimum requis: {min_nodes}).")

    if n_edges < min_edges:
        raise ValueError(f"Le graphe n'a pas assez de liens ({n_edges}) pour l'entraînement.")

    # 4. Vérification optionnelle : Self-loops (peuvent fausser SP et CN)
    n_self_loops = nx.number_of_selfloops(G)
    if n_self_loops > 0:
        print(f"Warning: {n_self_loops} boucles sur soi (self-loops) détectées. "
              "Il est recommandé de les supprimer avec G.remove_edges_from(nx.selfloop_edges(G)).")

    return True


########################################
# FONCTIONS DE CALCUL DES FEATURES #####
########################################
def hide_graph_links(G, test_size = 0.15):
    all_edges = list(G.edges())
    random.seed(42)
    random.shuffle(all_edges)
    
    split_idx = int(len(all_edges) * (1 - test_size))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    
    # 2. Création du graphe d'entraînement (G sans le test set)
    # C'est sur ce graphe qu'on va tout calculer
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(train_edges)
    G_train.graph.update(G.graph)

    G_eval = nx.Graph()
    G_eval.add_nodes_from(G.nodes(data=True))
    G_eval.add_edges_from(test_edges)
    G_eval.graph.update(G.graph)
    
    print(f"Graphe original: {G.number_of_edges()} liens")
    print(f"Graphe d'entraînement: {G_train.number_of_edges()} liens")
    print(f"Liens cachés pour le test: {len(test_edges)}")

    return G_train, G_eval


def _extract_pair_features(G_train, u, v, densities):
    """
    Aggrège les infos de noeuds (IDs de blocs, centralités) et 
    calcule les métriques de paires à la volée.
    """
    nu = G_train.nodes[u]
    nv = G_train.nodes[v]

    features = {}

    for algo in COMMUNITY_ALGOS :
        id_u = nu.get(f'{algo}_id')
        id_v = nv.get(f'{algo}_id')
        if id_u is None or id_v is None:
            #print(f"ALERTE : Noeud u={u} ou v={v} a un ID None pour {algo} !")
            #print(f"DEBUG: Attr cherché: {algo}_id | Présents dans nu: {list(nu.keys())}")
            id_u = 0
            id_v = 0
        pair = tuple(sorted((id_u, id_v)))
        
        try:
            features[f'{algo}_density'] = densities[algo].get(pair, 0)
            
        except Exception as e:
            #print(f"ERREUR ({algo}) : {e} -> Métrique passée.")
            continue

    for emb in EMBEDDINGS:
        if emb in nu and emb in nv:
            vec_u = nu[emb].reshape(1, -1)
            vec_v = nv[emb].reshape(1, -1)
            hadamard_prod = vec_u * vec_v
            features[f'{emb}_cos'] = cosine_similarity(vec_u, vec_v)[0][0]
            features[f'{emb}_dist'] = np.linalg.norm(vec_u - vec_v)
        
    return features

def _worker_extract(u, v, target, G_train, densities):
    """
    Fonction isolée pour un processus : extrait les features d'une paire unique.
    """
    features = _extract_pair_features(G_train, u, v, densities)

    return {'u': u, 'v': v, 'target': target, **features}

def prepare_balanced_data(G, G_train, negative_ratio=10.0, GroundTruth = None, n_jobs=-2):
    """
    Prépare le dataset final en utilisant G_train pour les features
    et G pour vérifier l'existence réelle des liens (target).
    """
    total_cores = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, total_cores + n_jobs)
    else:
        n_jobs = min(n_jobs, total_cores) if n_jobs > 0 else total_cores

    all_edges = list(G.edges())
    nodes = list(G.nodes())
    n_pos = len(all_edges)
    densities = prepare_all_densities(G_train)

    print(f"Préparation des listes de paires...")
    tasks = [(u, v, 1) for u, v in all_edges]
    
    n_neg_target = int(n_pos * negative_ratio)
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if u != v and not G.has_edge(u, v) and not G_train.has_edge(u, v):
            tasks.append((u, v, 0))
            neg_count += 1

    print(f"Extraction parallèle sur {len(tasks)} paires (n_jobs={n_jobs})...")
    
    results = Parallel(n_jobs=n_jobs, batch_size=1000, backend="loky")(
        delayed(_worker_extract)(u, v, target, G_train, densities) 
        for u, v, target in tasks
    )

    df = pd.DataFrame(results)

        
    if GroundTruth is not None:
        print(f"Injection de la Ground Truth ({len(GroundTruth)} sources)...")
        node_list = list(G.nodes()) # L'ordre utilisé lors de la création de GT_pos
        mapping = {node_id: i for i, node_id in enumerate(node_list)}
        
        indices_u = df['u'].map(mapping).values.astype(int)
        indices_v = df['v'].map(mapping).values.astype(int)
        
        for feat_name, data in GroundTruth.items():
            if data is None:
                continue
                
            # Cas spécifiques (nominatifs) 
            if feat_name == 'GT_pos':
                pos_u = data[indices_u]
                pos_v = data[indices_v]
                df['GT_pos_dist'] = np.linalg.norm(pos_u - pos_v, axis=1)
            elif feat_name == 'GT_sbm_matrix':
                ids_u = GroundTruth['GT_sbm_id'][indices_u]
                ids_v = GroundTruth['GT_sbm_id'][indices_v]
                df['GT_sbm_density'] = data[ids_u, ids_v]
     
            # Cas 1 : Matrice de Paires (N x N)
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0]==data.shape[1] and data.shape[0] > 100: 
                df[feat_name] = data[indices_u, indices_v]

            # Cas 2 : Vecteurs de Nœuds (N,) -> Ex: GT_degrees_sbm, GT_degrees_spatial
            elif isinstance(data, np.ndarray) and data.ndim == 1:
                df[f"{feat_name}_u"] = data[indices_u]
                df[f"{feat_name}_v"] = data[indices_v]

        print(f"DataFrame enrichi. Colonnes GT : {[c for c in df.columns if c.startswith('GT_')]}")

    print(f"DataFrame créé avec succès : {df.shape[0]} lignes.")
    return df

#############################################
## FONCTIONS POUR INFERENCE DE COMMUNAUTES ##
#############################################

def _appendLouvainCommunities(G_train, K_min=3, min_edge_ratio=0.01):
    best_p = _find_best_partition(
        G_train, 
        nx.community.louvain_communities, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio,
    )
    
    nx.set_node_attributes(G_train, best_p, "louvain_id")
    _normalize_community_assignment(G_train, "louvain_id")
    
    return G_train


def _normalize_community_assignment(G, attr_name):
    """ Remplace les NaN par des IDs uniques (singletons) """
    nodes_data = nx.get_node_attributes(G, attr_name)
    
    current_ids = [int(v) for v in nodes_data.values() if pd.notnull(v)]
    next_id = max(current_ids) + 1 if current_ids else 0
    
    mapping = {}
    for node in G.nodes():
        val = nodes_data.get(node)
        if pd.isnull(val):
            mapping[node] = next_id
            next_id += 1
        else:
            mapping[node] = val
            
    nx.set_node_attributes(G, mapping, attr_name)
    

COMMUNITY_MAPPING = {
    'louvain': _appendLouvainCommunities,
    "spatial_louvain" : _appendSpatialLouvainCommunities,
    "sbm_louvain": _appendSbmLouvainCommunities,
}


def computeCommunityFeatures(G_train, algos="All", spatial_ref = "GT_pos"):
    print("\n--- Enrichissement du Graphe avec les Communautés ---")
    to_run = COMMUNITY_ALGOS if algos == "All" else algos
    
    for algo in to_run:
        if algo in COMMUNITY_MAPPING:
            print(f"Calcul des communautés via {algo}...")
            if algo.startswith("spatial_"):
                COMMUNITY_MAPPING[algo](G_train, pos_attr= spatial_ref)
            else :
                COMMUNITY_MAPPING[algo](G_train)
                
        else:
            print(f"Attention : L'algorithme {algo} n'est pas reconnu.")
            
    return G_train


def prepare_all_densities(G_train):
    """
    Pré-calcule les densités de blocs pour tous les algorithmes et embeddings concernés.
    """
    all_densities = {}
    targets = []
    
    for algo in COMMUNITY_ALGOS:
        targets.append((algo, f"{algo}_id"))
            
    for key, attr_name in targets:
        node_to_block = nx.get_node_attributes(G_train, attr_name)
        
        # Si le graphe n'a pas cet attribut, on évite le crash
        if not node_to_block:
            continue
        
        # Compter les membres par bloc
        block_sizes = pd.Series(node_to_block).value_counts().to_dict()
        blocks = list(block_sizes.keys())
        
        # Compter les liens réels entre blocs (triangle supérieur)
        counts = {(b1, b2): 0 for i, b1 in enumerate(blocks) for b2 in blocks[i:]}
        
        for u, v in G_train.edges():
            bu, bv = node_to_block.get(u), node_to_block.get(v)
            if bu is not None and bv is not None:
                pair = tuple(sorted((bu, bv)))
                if pair in counts:
                    counts[pair] += 1
        
        # Calculer les densités
        algo_densities = {}
        for (b1, b2), real_count in counts.items():
            n1, n2 = block_sizes[b1], block_sizes[b2]
            if b1 == b2:
                possible = (n1 * (n1 - 1)) / 2  # Intra
            else:
                possible = n1 * n2              # Inter
            
            algo_densities[(b1, b2)] = real_count / possible if possible > 0 else 0
            
        all_densities[key] = algo_densities
        
    return all_densities

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

###########################################
## FONCTIONS POUR INFERENCE D'EMBEDDINGS ##
###########################################

def _append_node2vec_features(G_train, p, q, attr_name, dimensions=64):
    """
    Génère les embeddings Node2Vec et retourne un dictionnaire {node_id: vector}
    """
    print(f"Calcul de Node2Vec (p={p}, q={q})...")
    print(f"Génération des marches aléatoires (dim={dimensions})...")

    cores = multiprocessing.cpu_count() -1
    
    # Configuration de Node2Vec
    # p=1, q=1 => équivalent à DeepWalk
    node2vec = Node2Vec(G_train, 
                        dimensions=dimensions, 
                        walk_length=30, 
                        num_walks=100, 
                        workers=cores, 
                        p=p, q=q)

    print("Entraînement du modèle Skip-gram...")
    start_skip = time.time()
    model = node2vec.fit(window=10, min_count=1, batch_words=1000, vector_size=dimensions, workers=cores)
    
    embeddings = {}
    for node in G_train.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            embeddings[node] = model.wv[str(node)]

    nx.set_node_attributes(G_train, embeddings, attr_name)

    end_skip = time.time()
    skipgram_duration = end_skip - start_skip
    print(f"Skip-gram terminé en {skipgram_duration:.2f}s")
    
EMBEDDING_MAPPING = {
    'deepwalk': lambda G: _append_node2vec_features(G, p=1, q=1, attr_name="deepwalk"),
    'SiNEcustom_spatial' : lambda G, pos_attr="GT_pos": _append_SiNEcustom(G, pos_attr, attr_name="SiNEcustom_spatial", NullModel_method="ManualIter", temperature=0.5),
    "SiNEcustom_sbm": lambda G:_append_SineSBMcustom(G, com_attr="GT_sbm_id", attr_name="SiNEcustom_sbm", temperature=0.5, emb_dim=64, epochs=100, lr=0.1),
    "deepwalk_residuals": lambda G:_append_DeepwalkOnResiduals_full(G, pos_attr="GT_pos", attr_name="deepwalk_residuals", NullModel_method="ManualIter", emb_dim=64),
}

def computeDistanceFeatures(G_train, embeddings="All", spatial_ref="GT_pos"):
    to_run = EMBEDDINGS if embeddings == "All" else embeddings
    print("\n--- Enrichissement du Graphe avec les Embeddings ---")

    for emb in to_run:
        if emb in EMBEDDING_MAPPING:
            print(f"Calcul des embeddings via {emb}...")
            if emb.endswith("_spatial") or emb.endswith("_spatial_bined"):
                EMBEDDING_MAPPING[emb](G_train, pos_attr=spatial_ref)
            else:
                EMBEDDING_MAPPING[emb](G_train)
        else:
            print(f"Attention : L'algorithme {emb} n'est pas reconnu.")
    return G_train


#################################################
######### FONCTIONS DE CROSS VALIDATION #########
#################################################

def k_fold_cross_validation(G, k=2, features_list=None, n_trials=50, GroundTruth =None, graph_name="G_NAME"):
    folds_data = _prepare_precalculated_folds(G, k=k, GroundTruth=GroundTruth)
    study = _run_optuna_tuning(folds_data, features_list, n_trials=n_trials)
    
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                'Trial': trial.number,
                'Avg_AUC': trial.value,
                'Std_AUC': trial.user_attrs.get('std_auc'),
                'Avg_AP': trial.user_attrs.get('avg_ap'),
                'Delta_AUC': trial.user_attrs.get('delta_auc'),
                'Params': trial.params
            })
    
    summary_df = pd.DataFrame(results).sort_values(by='Avg_AUC', ascending=False)

    print("\n" + "="*80)
    print(f"{'RÉSULTATS OPTUNA : BASELINE VS TOP CONFIGURATIONS':^80}")
    print("="*80)

    cols = ['Trial', 'Avg_AUC', 'Std_AUC', 'Avg_AP', 'Delta_AUC']
    print(summary_df[summary_df['Trial'] == 0][cols].to_string(index=False))
    print("-" * 80)
    print(summary_df.head(3)[cols].to_string(index=False))
    print("="*80)

    save_dir = "outputs/results"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"optuna_results_{graph_name}.csv"
    full_path = os.path.join(save_dir, filename)
    summary_df.to_csv(full_path, index=False)
    print(f"Résultats sauvegardés dans : {full_path}")

    best_params = study.best_params.copy()
    best_params.update({'tree_method': 'hist', 'n_estimators': 150})
    
    return best_params, summary_df

def _process_single_fold(f_idx, t_idx, v_idx, edges, nodes_data, GroundTruth=None):
    print(f"--- Démarrage Parallèle Fold {f_idx + 1} ---")
    # Construction du graphe kept
    kept_edges = [edges[i] for i in t_idx]
    G_kept = nx.Graph()
    G_kept.add_nodes_from(nodes_data)
    G_kept.add_edges_from(kept_edges)

    # Séparation en graphe de train/test
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)
    
    # G_hiden : pour le validation set
    hidden_edges = [edges[i] for i in v_idx]
    G_hidden = nx.Graph()
    G_hidden.add_nodes_from(nodes_data)
    G_hidden.add_edges_from(hidden_edges)

    # Enrichissement du graphe de train
    G_train = computeDistanceFeatures(G_train)
    G_train = computeCommunityFeatures(G_train)

    # Enrichissement du graphe de validation finale
    G_kept = computeDistanceFeatures(G_kept)
    G_kept = computeCommunityFeatures(G_kept)

    # Création des datasets
    ds_train = prepare_balanced_data(G_test, G_train, negative_ratio=10.0, GroundTruth=GroundTruth) 
    ds_val = prepare_balanced_data(G_hidden, G_kept, negative_ratio=25.0, GroundTruth=GroundTruth)
    
    return (ds_train, ds_val)

def _prepare_precalculated_folds(G, k=1, GroundTruth = None):
    edges = list(G.edges())
    nodes_data = list(G.nodes(data=True))

    if k == 1:
        folds_idx = [train_test_split(range(len(edges)), test_size=0.2, random_state=42)]
    else:
        kf = KFold(n_splits=k, shuffle=True)
        folds_idx = list(kf.split(edges))

    print(f"[K-FOLD] Préparation séquentielle de {len(folds_idx)} folds...")

    # Anciennement //isé, plus efficace comme ça pour éviter //isations imbriquées.
    precalculated_folds = [
        _process_single_fold(i, t_idx, v_idx, edges, nodes_data, GroundTruth=GroundTruth)
        for i, (t_idx, v_idx) in enumerate(folds_idx)
    ]
    
    return precalculated_folds

def _run_optuna_tuning(precalculated_folds, features_list=None, n_trials=50, n_jobs = -2):

    if features_list is None or len(features_list) == 0:
        exclude = ['u', 'v', 'target', 'label']
        features = [
            col for col in precalculated_folds[0][0].columns
            if (col not in exclude and not col.startswith('GT_'))
            #or col in ['GT_sbm_density', 'GT_pos_dist','GT_spatial_deg_product', 'GT_sbm_deg_product']
        ]
        print(f"Features détectées ({len(features)}) : {features}")
    else:
        features = features_list

    optimized_folds = []
    for ds_train, ds_val in precalculated_folds:
        optimized_folds.append({
            'X_train': ds_train[features].values.astype('float32'),
            'y_train': ds_train['target'].values,
            'X_val': ds_val[features].values.astype('float32'),
            'y_val': ds_val['target'].values
        })

    total_cores = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, total_cores + n_jobs)
    else:
        n_jobs = min(n_jobs, total_cores) if n_jobs > 0 else total_cores

    def objective(trial):
        params = {
            'n_estimators': 150,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist',
            "n_jobs" : n_jobs,
            'random_state': 42
        }

        f_auc_v, f_auc_t, f_ap_v = [], [], []

        for fold in optimized_folds:
            model = XGBClassifier(**params)
            model.fit(fold['X_train'], fold['y_train'])

            p_val = model.predict_proba(fold['X_val'])[:, 1]
            p_train = model.predict_proba(fold['X_train'])[:, 1]
            
            f_auc_v.append(roc_auc_score(fold['y_val'], p_val))
            f_auc_t.append(roc_auc_score(fold['y_train'], p_train))
            f_ap_v.append(average_precision_score(fold['y_val'], p_val))
        
        avg_auc_v = np.mean(f_auc_v)
        trial.set_user_attr("std_auc", np.std(f_auc_v))
        trial.set_user_attr("avg_ap", np.mean(f_ap_v))
        trial.set_user_attr("delta_auc", np.mean(f_auc_t) - avg_auc_v)

        del model 
        gc.collect()

        return avg_auc_v

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Pour ne garder que les logs d'erreur d'optuna
    study = optuna.create_study(direction='maximize')
    baseline = {'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 6,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 1e-3, 'reg_lambda': 1.0
    }
    study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials)
    
    return study


########################################
## FONCTIONS UTILITAIRES DE LOAD SAVE ##
########################################

def save_dataset(dataset, filename="dataset"):
    output_dir = os.path.join(os.getcwd(), "your_results", "data")
    output_path = os.path.join(output_dir, filename)
    
    # Création du dossier (absolu)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset.to_parquet(output_path, index=False)
    print(f"Dataset (DataFrame) sauvegardé : {output_path}")

    return output_path

def load_dataset(filename="dataset", talk = False):
    input_dir = os.path.join(os.getcwd(), "your_results", "data")
    input_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(input_path) :
        print(f"Erreur : Le fichier n'existe pas : {input_path}")
        return None
    
    dataset = pd.read_parquet(input_path)
    if talk :
        print(f" Dataset chargé avec succès depuis : {input_path}")
        print(f" Taille : {dataset.shape[0]} lignes, {dataset.shape[1]} colonnes.")
    
    return dataset

def loadsave_data_joblib(data=None, filename="data.joblib", mode="save", talk=False):
    """
    Gère la sauvegarde et le chargement d'objets en .joblib (SHAP, XGBoost, etc.).
    """
    base_path = Path.cwd()
    target_path = base_path / "your_results" / "data" / filename

    if mode == "save":
        if data is None :
            print("Erreur : Aucun objet fourni pour la sauvegarde.")
            return None
        
        # Création du dossier
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(data, target_path, compress=3)
        if talk :
            print(f"Objet sauvegardé dans : {target_path}")
        return target_path

    elif mode == "load":
        if not target_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {target_path}")
        
        obj = joblib.load(target_path)
        if talk :
            print(f"Objet chargé avec succès depuis : {target_path}")
        
        return obj

def load_all_data_for_graph(G_name, talk=False):
    # 1. G_train (avec structure, communautés et distances)
    try:
        G_train = loadsave_data_joblib(data=None, filename=f"G_train_w_struct_com_dist_{G_name}", mode="load", talk = talk)
    except Exception:
        #print(f"G_train introuvable pour {G_name}, création d'un graphe vide.")
        G_train = nx.Graph()

    # 2. Dataset de Train (via load_dataset)
    try:
        dataset_train = load_dataset(filename=f"dataset_train_{G_name}", talk = talk)
    except Exception:
        print(f"Dataset de Train introuvable pour {G_name}.")
        dataset_train = None

    # 3. Dataset d'Évaluation (via load_dataset)
    try:
        dataset_hidden = load_dataset(filename=f"dataset_hidden_{G_name}", talk = talk)
    except Exception:
        print(f"Dataset d'Évaluation introuvable pour {G_name}.")
        dataset_hidden = None

    # 4. Données XGBoost (Modèle, X_test, etc.)
    try:
        xgboost_data = loadsave_data_joblib(data=None, filename=f"xgboost_data_{G_name}.joblib", mode="load", talk = talk)
    except Exception:
        #print(f"Données XGBoost introuvables pour {G_name}.")
        xgboost_data = None

    return G_train, dataset_train, dataset_hidden, xgboost_data

class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def load_graphml_safe(path, speak=False):
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    clean_data = html.unescape(raw_data)
    G = nx.read_graphml(io.StringIO(clean_data))

    if speak : 
        print(f"✅ Graphe chargé : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
    
    return G

def save_graph(G, filename):
    base_path = Path.cwd()
    target_path = base_path / "your_results" / "data" / filename

    data = nx.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, cls=GraphEncoder)
    print(f"Graphe sauvegardé dans {filename}")

def load_graph(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)