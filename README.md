# DUPUY-Guilhem-2026-paper

> **Citation Note:** If you use this code or repository for your research paper, please cite: *[Insert Citation Here Upon Publication]*.

---

## 0 - Installation

This repository requires specific handling for `graph-tool` due to its C++ dependencies. Follow these steps to build the correct environment:

1. **Create and activate a clean environment:**
   ``` bash
   conda create --name env_linkpred python=3.10 -y
   conda activate env_linkpred
   ```

2. **Install `graph-tool` via Conda (Required first):**
   ``` bash
   conda install -c conda-forge graph-tool -y
   ```

3. **Install the remaining dependencies via Pip, listed in the file "requirements.txt":**
   ``` bash
   pip install -r requirements.txt
   ```
---

## 1 - Functions available for reproductibility

### a. Compute decorelated Features & compare with Baselines
To launch the embedding generation, community detection, and Link Prediction evaluation pipeline on both decorelated metrics and their baselines, run:

    ```bash
    python main.py
    ```

* **Key parameters inside `main.py`:**
  * `NB_ITERATIONS = 2`: Defines the subset range of graphs to process.
  * `GRAPH_NAMES = "artificial_graph_sbmv_4"`: Root name matching your generated benchmark.

The data generated during execution (Graphs with communities & embeddings stored as atteributes, train & evaluation datasets) will be stored in 📁 **`your_results/data`** in case you wish to perform further study.
The resulting plots will be stored in 📁 **`your_results/plots`**

### b. Re-generate the Synthetic Benchmark
An artificial benchmark of 330 graphs is already available in graph_library (hybrids ratioin linearily spaced between 0 and 1 with a step of 0.1).
If you wish to re-generate a new benchmark with your own hybrid ratios and / or more example of each, run:
python generate_graph_benchmark.py

* **Key parameters inside the file:**
  * `BENCHMARK_SIZE = 30`: Number of graphs generated per hybridization ratio.
  * `HYBRID_RATIO_LIST = np.arange(1.00, -0.10, -0.10)`: Grid of SBM/Spatial mix values.

All generated graphs will be sotred in 📁 **`graph_library/`**

---

## 3 Repository Structure

* * **`LouvainDecorele.py`** – Implementation of our original Spatial decorrelation method for spatial Louvain community detection.
* **`SiNEcustom.py`** – Implementation of our original custom signed decorrelated embedding model.
* **`ResidualDeterrenceEmbedding.py`** – Implementation of our original spatial residual deterrence embedding variants.
* **`MetaLouvain.py`** – Used for the computation of custom Null-Models Louvain communities.
* **`main.py`** – Main entry point to run the pipeline experiments and evaluations.
* **`generate_graph_benchmark.py`** – Generates the synthetic spatial/SBM hybrid graph datasets.
* **`pipeline_exec.py`** – Choreographs the execution of the feature computations, training/evaluation cycles, etc ...
* **`pipeline_utils.py`** – Core utility functions for features computations
* **`models.py`** – machine learning model (XGBoost)
* **`requirements.txt`** – List of Python package requirements (not including the graph-tools dependance)

