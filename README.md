# customer-analytics-and-segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](#)

Segment 3,000 customers of a national convenience retailer into actionable groups and profile each segment for Marketing and the CDO. This notebook engineers behaviour features, reduces dimensionality, evaluates clustering options, and delivers a 5-segment solution with pen portraits and an attractiveness ranking—plus a CSV mapping of `customer_number → segment`.

---

## Data

The notebook expects the coursework bundle (auto-downloaded in the first cell) with:

- `customers_sample.csv` — 3,000 rows; customer totals/averages (baskets, quantities, spend).  
- `category_spends_sample.csv` — 3,000 rows; spend across 20 categories.  
- `baskets_sample.csv` — ~195k rows; all visits (`purchase_time`, quantities, spend).  
- `lineitems_sample.csv` — ~1.46M rows; basket→product lines (used for richer features).

> Shapes printed in-notebook: Customers (3000×6), Category Spends (3000×21), Baskets (195,547×5), Line Items (1,461,315×6).

---

## Feature Engineering (final set used for clustering)

Scaled on the customer level, log1p applied to skewed variables. 
Final 20 features:

- **RFM & cadence:** `baskets`, `log_recency`, `log_tenure`, `log_avg_days_between`  
- **Basket economics & mix:** `avg_basket_spend`, `log_avg_item_count`, `total_spend`, `average_spend`  
- **Variety:** `unique_category_count`  
- **Category spend (key groups):**  
  `spend_dairy`, `log_spend_fruit_veg`, `log_spend_drinks`,  
  `log_spend_grocery_health_pets`, `log_spend_grocery_food`,  
  `log_spend_confectionary`, `log_spend_meat`, `spend_bakery`  
- **Category proportions (signals):** `log_prop_tobacco` (discouraging), `prop_dairy`, `prop_fruit_veg`

> Exclusions based on correlation/interpretability: e.g., `category_diversity`, `avg_basket_categories`, `avg_spend_per_item`, etc.

---

## Methodology

1. **Preprocessing:** `StandardScaler` on selected features, log transforms for |skew|>1.  
2. **Dimensionality reduction:** PCA to **≥80% variance** → **8 components** (shape: 3000×8).  
3. **Model selection:**  
   - **K-Means** evaluated for *k = 2…10* on PCA space using SSE (elbow), Silhouette, Calinski-Harabasz, Davies-Bouldin.  
   - **DBSCAN** grid over `eps ∈ [0.1, 2.0]`, `min_samples=5`, silhouette computed excluding noise.
4. **Chosen model:** **K-Means (k = 5)** — balances elbow/interpretability and satisfies coursework’s **5–7 clusters** requirement.  
   - K-Means (k=5) silhouette: **0.1594** on PCA space.  
   - DBSCAN alt: best `eps≈1.00`, silhouette (valid clusters only) **0.4261**, but yields **fewer clusters** and less actionable coverage, retained for comparison only.
5. **Explainability & business translation:**  
   - Cluster summaries (mean/std/median by feature).  
   - **Pen portraits** generated from high/low feature flags.  
   - **Attractiveness score** (hand-tuned weights that emphasise spend/frequency, penalise tobacco share) ranks segments for targeting.

---

## Results & Artefacts

- **Final solution:** **5 customer segments** (K-Means on PCA).  
- **Top target segments:** **Cluster 2** (score ≈ **0.928**) and **Cluster 1** (score ≈ **0.660**) by the attractiveness index.  
- **Deliverable CSV:** `kmeans_segmentation_results.csv` with columns:
  ```
  customer_number,kmeans_cluster
  ```
- Visuals produced in-notebook: correlation heatmap, elbow/SSE plot, method metrics table, PCA 2D scatter with centroids, cluster summary tables, RADAR chart, pen portraits.

---

## How to Run

### Option A — Jupyter/Colab (recommended)
1. Open `Customer_Analytics_and_Segmentation.ipynb`.  
2. **Run all cells**. The first cell downloads/unzips the coursework data bundle into `asa_cw1_data/`.  
3. Outputs include the printed analyses and the CSV mapping file.

### Option B — Local data
1. Place the four CSVs under `asa_cw1_data/` at repo root.  
2. Run the notebook top-to-bottom in a Python 3.10+ environment.

### Environment
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Repository Structure
```
.
├─ Customer_Analytics_and_Segmentation.ipynb   # end-to-end pipeline
├─ output/
   └─ kmeans_segmentation_results.csv          # output (customer → segment)
└─ README.md
```

---

## Notes & Assumptions

- Clustering run on PCA space (8 PCs capturing ≥80% variance) for stability and noise reduction.   
- Pen portraits and ranking are **data-driven** with light, transparent weighting to surface commercially attractive segments.

---

**Author:** Karthik Rajesh  
**Environment:** Google Colab
