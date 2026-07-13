from pipeline_exec import *
from pipeline_utils import *

import time
import numpy as np
import pandas as pd
from datetime import datetime

# nohup python -u main.py 2>&1 | grep --line-buffered -vE "it/s|\[.*\]|^----" | grep --line-buffered "." > myoutfile &

NB_ITERATIONS = 30
GRAPH_NAMES = "artificial_graph_sbmv_6" # Name of graphs (root) stored in graph_library
#"artificial_graph_simple_somme" # Name of graphs (root) stored in graph_library

if __name__ == "__main__":

    execution_stats = []
    
    for nbiter in range(1,NB_ITERATIONS + 1) : 
        for sbm_ratio in np.arange(0.00, 1.10, 0.10):

            G_name = f"{GRAPH_NAMES}_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            G_name_bis = f"{GRAPH_NAMES}_AllFeatures_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')
            #G_name_bis = f"{GRAPH_NAMES}_TestsMethodesCombinees_{sbm_ratio:.2f}_pos_{1-sbm_ratio:.2f}_{nbiter}".replace('.', '_')

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
            compute_features(G, G_name_bis, "GT_pos")      
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
                
            print(f"Terminé en {duration:.2f} secondes.")
    
    df = pd.DataFrame(execution_stats)
    print("\n" + "="*50)
    print("RÉSUMÉ DES STATISTIQUES D'EXÉCUTION")
    print("="*50)
    print(df.to_string(index=False))
    
    
    start_time = time.time()
    date_and_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    all_results = analyze_features(G_name_short = f"{GRAPH_NAMES}_AllFeatures", nb_iterations=NB_ITERATIONS, spatial_ref = "GT_pos", i_min = 0.00, i_max = 1.00, nb_i=11, name_export_results=date_and_time)
    #all_results = analyze_features(G_name_short = f"{GRAPH_NAMES}_TestsMethodesCombinees", nb_iterations=NB_ITERATIONS, spatial_ref = "GT_pos", i_min = 0.00, i_max = 1.00, nb_i=11, name_export_results=date_and_time)
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "="*50)
    print("TEMPS D'EXEC POUR ANALYSE:")
    print("="*50)
    print(f"{duration} secs")
    

    # Affichage des résultats : 
    #date_and_time = "03-07-2026_17-54"
    input_dir = os.path.join(os.getcwd(), "your_results", "data")
    filename =f"link_prediction_perfs_{GRAPH_NAMES}_AllFeatures_{NB_ITERATIONS}iter_{date_and_time}.csv"
    #filename =f"link_prediction_perfs_{GRAPH_NAMES}_TestsMethodesCombinees_{NB_ITERATIONS}iter_{date_and_time}.csv"
    results_path = os.path.join(input_dir, filename)
    generate_and_show_plot(results_path, name=f"link_prediction_perfs_{NB_ITERATIONS}iter_{date_and_time}")

    print("="*80)
    print("======= FINI !!!!!!!!======")
    print("="*80)
    