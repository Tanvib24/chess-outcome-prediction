# Predicting Online Chess Game Outcomes with Machine Learning (KDD Process)

**Project title:** To Predict the Outcome of Online Chess Games from Player Ratings, Time Controls, and Openings Using Machine Learning Following the KDD Process

---

## 1. Problem Statement
Can we predict whether an online chess game ends in a **white win, black win, or draw** using only what is known *before the first move*: both players' ratings, the time control, whether the game is rated, and the opening that gets played?

The practical value is in fair matchmaking, player coaching, and opening advice. To keep the model honest, every result-revealing column (`turns`, `victory_status`, `moves`, `last_move_at`) is excluded.

## 2. Dataset
**Chess Game Dataset (Lichess)**, Kaggle: https://www.kaggle.com/datasets/datasnaek/chess (file: `data/games.csv`)

| Item | Value |
|---|---|
| Raw rows / columns | 20,058 / 16 |
| Duplicate game IDs removed | 945 |
| Rows after cleaning | 19,113 |
| Target (`winner`) | white 9,545 (49.9%), black 8,680 (45.4%), draw 888 (4.6%) |
| Missing values | none |

**Used as inputs:** `white_rating`, `black_rating`, `rated`, `increment_code`, `opening_eco`, `opening_ply`
**Excluded (leakage or no information):** `turns`, `victory_status`, `moves`, `last_move_at`, `created_at` (truncated timestamps), `id`, `white_id`, `black_id`

## 3. Methodology (KDD stages)
1. **Selection and understanding:** profile the data, define the three-class target, audit leakage.
2. **Preprocessing:** drop duplicate IDs, check missing values, inspect rating outliers with the IQR rule (genuine players, so kept).
3. **Transformation / feature engineering:**
   - `rating_diff`, `abs_rating_diff`, `avg_rating`
   - `elo_expected_white`: the classical Elo win expectancy, 1 / (1 + 10^(-diff/400))
   - `base_time`, `increment`, `time_category` (Bullet / Blitz / Rapid / Classical)
   - `eco_family` (A to E) and `eco_group` (top 30 ECO codes, rest "Other")
   - standard scaling for numbers, one-hot encoding for categories
4. **EDA and feature selection:** outcome by rating gap, opening family, time category; correlation heatmap; chi-square tests; permutation importance.
5. **Data mining:** stratified 80/20 split; Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (plus XGBoost if installed). Hyper-parameters tuned with 5-fold CV on the training set only, optimising log-loss.
6. **Evaluation and interpretation:** accuracy, balanced accuracy, macro precision/recall/F1, log-loss, ROC-AUC, confusion matrices, 5-fold CV on all data, two baselines, and a data-leakage demonstration.

## 4. Results (held-out test set, 3,823 games)

| Model | Accuracy | Macro F1 | Log-loss | ROC-AUC |
|---|---|---|---|---|
| Majority class (baseline) | 0.499 | 0.222 | 18.05 | 0.500 |
| Rule: higher rating wins (baseline) | 0.622 | 0.424 | n/a | n/a |
| **Logistic Regression** (best log-loss) | 0.624 | 0.424 | **0.7627** | **0.688** |
| Decision Tree | 0.620 | 0.417 | 0.7691 | 0.659 |
| Random Forest | 0.623 | 0.423 | 0.7627 | 0.685 |
| Gradient Boosting | **0.628** | **0.427** | 0.7632 | 0.684 |

5-fold cross-validated accuracy sits between 0.619 and 0.622 for all four models (standard deviation about 0.01), so the differences between them are within noise.

### Key findings
- **Rating gap dominates.** White's win rate climbs from 17.2% when white is 300+ points weaker to 82.1% when white is 300+ points stronger.
- **Openings and time control matter only slightly.** Chi-square tests show a statistically significant link (opening group p < 0.001, time category p < 0.001, rated p = 0.005), but permutation importance shows they add very little predictive power on top of ratings.
- **Draws are essentially unpredictable pre-game** (4.6% of games; F1 for draws is 0.00 for the best model).
- **Models agree.** Simple and complex models land within half a percentage point of each other, which tells us the ceiling is set by the information, not the algorithm.

### Why the model is not "100% accurate"
Chess results depend on the moves, which are unknown before the game. Section 8 of the notebook adds `turns` and `victory_status` as a demonstration and accuracy jumps to **90.3%**, but that gain is leakage (for example, an odd number of turns means white made the last move). A model that uses it cannot be used before a game starts, so the honest ~62-63% is the real result.

## 5. Repository Structure
```
.
├── chess_outcome_prediction.ipynb   # full implementation, executed with outputs
├── data/games.csv                   # dataset
├── figures/                         # charts used in the report and slides
├── report/Project_Report.pdf        # final report
├── report/Chess_Outcome_Prediction_Presentation.pptx
├── requirements.txt
└── README.md
```

## 6. How to Run
```bash
pip install -r requirements.txt
jupyter notebook chess_outcome_prediction.ipynb     # Run All
```
The notebook reads `data/games.csv`, saves charts to `figures/`, and writes `metrics.json`. Runtime is a few minutes on a laptop. XGBoost is optional; if installed it is added to the comparison automatically.

## 7. Limitations and Future Work
- Only pre-game attributes are used, so accuracy is capped by chess itself.
- One platform, one period, about 19,000 games; many players appear only once, so per-player history is not usable here.
- Future work: larger Lichess dumps with player history, engine evaluation of opening positions, and per-player form and head-to-head records.

## 8. References
1. Elo, A. (1978). *The Rating of Chessplayers, Past and Present.* Arco.
2. Glickman, M. E. (2001). Dynamic paired comparison models with stochastic variances. *Journal of Applied Statistics, 28*(6), 673-689.
3. Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine, 17*(3), 37-54.
4. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining. *ACM TKDD, 6*(4).
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825-2830.
6. Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32.
7. Mitchell, J. (2017). Chess Game Dataset (Lichess). Kaggle.
