# Hybrid Recommendation Engine with Cold-Start Handling

Hybrid recommender on MovieLens 1M: SVD collaborative filtering plus content-based
features, blended with a weight that shifts toward content and popularity when a
user has little history.

## Setup
Easiest: open `project.ipynb` in Google Colab and run all cells
(Runtime -> Run all). The notebook downloads the dataset itself.

Local: run `pip install -r requirements.txt`, download
https://files.grouplens.org/datasets/movielens/ml-1m.zip, unzip it next to the
notebook, then run the notebook.

## Reproducing results
The random seed is fixed (42). Run all cells in order. The results tables are produced
by the evaluation cells (Precision@10, Recall@10, NDCG@10). To re-run only the
evaluation, re-run the evaluation cells after the models are built.

## Method
- Split: each user's latest 20% of ratings are test; relevant = rating >= 4.
- Cold-start: 20% of users keep only 1-4 training ratings; cold items have <5 training ratings.
- Models: popularity, SVD (k=50), content (genre + decade TF-IDF), and hybrids.
- Blend: CF weight = n / (n + 80), where n is the user's number of ratings; the cold side is 0.1 content + 0.9 popularity.

## Key results (NDCG@10)
| Model | Cold users | Warm users |
|---|---|---|
| Popularity | 0.0402 | 0.0919 |
| CF only | 0.0357 | 0.1174 |
| Content only (v2) | 0.0075 | 0.0174 |
| Hybrid 50/50 | 0.0307 | 0.1060 |
| Hybrid adaptive (final) | 0.0449 | 0.1184 |

Cold items (584 items with <5 ratings): hybrid NDCG@10 0.053, CF 0.052, content-only 0.036, popularity 0.028.

## Notes
The adaptive hybrid beats popularity modestly on cold users and matches CF on warm users. Content features (genre and decade) were weak on their own, so most of the gain comes from CF plus a popularity prior. Hyperparameters were tuned on the evaluation split, and the cold-item slice is small (148 users), so small differences are not significant. See the notebook for the full write-up and failure analysis.
