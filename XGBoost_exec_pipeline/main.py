from pipeline_exec import *
from pipeline_utils import *

import time
import numpy as np
import pandas as pd

# nohup python -u main.py 2>&1 | grep --line-buffered -vE "it/s|%|\[.*\]|^----" | grep --line-buffered "." > myoutfile &
# 

if __name__ == "__main__":

    execution_stats = []

    for nbiter in range(1,5) : 
        for sbm_ratio in np.arange(0.00, 1.10, 0.10):
            
            G_name = f"artificial_graph_sbmv_4_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            G_name_bis = f"artificial_graph_sbmv_4_AllFeatures_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            
            print("######################################")
            print(f"#### graph {G_name} :  ####")
            print("######################################")
            print (f"G bname bis : {G_name_bis}")
            
            path = f"graph_library/{G_name}.graphml"
            try:
                G = load_graphml_safe(path)
                print(f"Graphe chargé avec succès : {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")
            except Exception as e:
                print(f"Erreur lors du chargement de {path} : {e}")

            start_time = time.time()
            compute_commus(G, G_name_bis, "GT_pos", computeEmb=True)      
            end_time = time.time()
            duration = end_time - start_time
    
            execution_stats.append({
                    "Graph": G_name,
                    "Nodes": G.number_of_nodes(),
                    "Edges": G.number_of_edges(),
                    "Time_sec": round(duration, 2),
                    "Time_per_node": round(duration / G.number_of_nodes(), 4) if G.number_of_nodes() > 0 else 0,
                    "Time_per_link": round(duration / G.number_of_edges(), 4) if G.number_of_edges() > 0 else 0
                })
                
            print(f"⏱️ Terminé en {duration:.2f} secondes.")
    
    df = pd.DataFrame(execution_stats)
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DES STATISTIQUES D'EXÉCUTION")
    print("="*50)
    print(df.to_string(index=False))
    
    
    start_time = time.time()
    all_results = analyze_commus(G_name_short = "artificial_graph_sbmv_4_embDebiased", nb_iterations=1, spatial_ref = "GT_pos", i_min = 0.00, i_max = 1.00, nb_i=11, name_export_results="")
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "="*50)
    print("📊 TEMPS D'EXEC POUR ANALYSIS PAS HALAL :")
    print("="*50)
    print(f"{duration} secs")