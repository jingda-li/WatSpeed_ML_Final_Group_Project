# Predicting Forest Cover Type — Group 6 Final Project

Final group project for the **Introduction to Machine Learning** course offered by **WatSpeed**. The project develops, tunes, and compares five supervised machine learning models to predict `Cover_Type` — the dominant tree species of a forest patch — from environmental and topographic features.

**Team:** Jane Sanchez, Julie Huynh, Jingda Li, Jessy Donelle, Jeremy May

## Motivation

Traditional forest inventories rely on costly, time-intensive field surveys to collect vegetation information across large or remote landscapes. As forests face growing pressure from climate change, wildfires, invasive species, and land-use change, there is increasing demand for faster, more scalable ways to monitor forest conditions. Machine learning offers a way to predict forest cover type from readily available topographic and environmental variables — without extensive field sampling.

**Business problem:** support scalable, low-cost monitoring for forest inventory management, biodiversity conservation, wildfire risk assessment, habitat monitoring, and environmental planning.

## Objective

Compare five supervised classification algorithms — Logistic Regression, Linear SVC, Decision Tree, Random Forest, and Neural Network — trained on the same dataset and splits, to identify which best predicts forest cover type, and to understand:
- the strengths/limitations of linear vs. nonlinear approaches on this problem,
- the effect of class imbalance on model performance, and
- which environmental variables most influence forest cover type.

## Dataset

- **Source:** [Forest Cover Type dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset), UCI Machine Learning Repository (Blackard, 1998), originally compiled by the US Forest Service Region 2 Resource Information System for the Roosevelt National Forest, Colorado.
- **Size:** 581,012 observations (30×30m spatial cells) across four wilderness areas, 54 attributes — continuous variables (elevation, slope, aspect, hillshade indices, distances to hydrology/roadways/fire points) and binary indicators (4 wilderness areas, 40 soil types).
- **Data quality:** complete records, no missing values, no imputation required.
- **Preprocessing:** continuous features standardized with `StandardScaler` for scale-sensitive models (Logistic Regression, SVMs, Neural Network); binary indicators left as-is; tree-based models used unscaled features. 80/20 stratified train-test split (`random_state=42`) to preserve class proportions.

**Key challenges:**
- **Class imbalance** — Spruce/Fir and Lodgepole Pine dominate the dataset; Cottonwood/Willow and Aspen are heavily underrepresented. Addressed with `class_weight='balanced'` (and manually computed class weights for the Keras neural network).
- **Class overlap & non-linearity** — t-SNE and LDA projections show classes are only partially separable, with environmental features interacting in complex, non-linear ways.
- **Computational scale** — 581K+ records increased training cost; the RBF SVC was trained on a 50,000-record subsample, and model outputs were persisted to avoid recomputation.

## Approach

1. **Exploratory Data Analysis** — class distribution, correlation and continuous-feature signal, tree-based feature importance, binary feature engineering, soil/wilderness overlap analysis, and LDA/t-SNE projections for visualizing class separability.
2. **Model Evaluation Framework** — consistent 80/20 stratified train-test pipeline; performance compared via Accuracy, Macro Recall, Macro F1, and Weighted F1, with **Macro F1** as the primary metric since it weighs all seven cover types equally regardless of frequency.
3. **Models trained** (see `Table A.1` in the report for full hyperparameter details):
   - **Logistic Regression** — linear baseline (`lbfgs`, L2, `max_iter=500`)
   - **Linear SVC** — linear baseline
   - **Decision Tree** — tuned via `RandomizedSearchCV` (`max_depth`, `criterion`, `min_samples_split`, `min_samples_leaf`)
   - **Random Forest** — tuned via `RandomizedSearchCV` (`n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`)
   - **RBF SVC** — non-linear kernel, tuned via `GridSearchCV` on a 50k-record subsample
   - **Residual Neural Network** — ReLU network with residual connections, batch normalization, dropout (0.2–0.3), Adam optimizer, early stopping, and `ReduceLROnPlateau`, tuned manually against validation Macro F1
4. **Learning curves** generated for the classical models (vs. training set size) and separately for the neural networks (vs. training epoch) to assess bias/variance trade-offs and convergence.

## Results

| Model | Accuracy | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Logistic Regression | 0.5990 | 0.7093 | 0.5069 | 0.6278 |
| Linear SVC | 0.6809 | 0.6223 | 0.5426 | 0.6883 |
| Decision Tree (tuned) | 0.9391 | 0.9039 | 0.9029 | 0.9391 |
| **Random Forest (tuned)** | **0.9567** | **0.9466** | **0.9328** | **0.9568** |
| RBF SVC (tuned) | 0.7905 | 0.8351 | 0.7251 | 0.7973 |
| Neural Net (residual, balanced weights) | 0.9268 | 0.9588 | 0.8921 | 0.9280 |

**Random Forest was the best-performing model overall**, achieving the strongest Accuracy, Macro Recall, Macro F1, and Weighted F1, and the best per-class F1 for all seven cover types. Nonlinear models (Random Forest, Decision Tree, RBF SVC, Residual Neural Network) consistently and substantially outperformed the linear baselines (Logistic Regression, Linear SVC), especially on minority classes like Aspen, Cottonwood/Willow, and Douglas-fir — indicating the relationships between environmental variables and forest cover type are highly non-linear.

Feature importance analysis (mean decrease in impurity) on the final Random Forest model showed **Elevation** as by far the strongest predictor, consistent with tree species occupying distinct altitudinal bands driven by temperature and climate gradients, followed by distance-based features (roadways, fire points, hydrology).

## Conclusion

The project's objective was met: all five algorithms could classify forest cover type to varying degrees, and the **Random Forest classifier** was identified as the best overall model. The dataset itself was clean and complete, but exhibited substantial class imbalance and partial class overlap, reinforcing the value of Macro F1/Macro Recall over raw accuracy for evaluation. Environmental variables — especially elevation, soil type, and wilderness area — were strong predictors, and nonlinear models were needed to capture their complex interactions.

**Future work** could explore additional ensemble methods (XGBoost, LightGBM, CatBoost), incorporate climate or satellite-derived predictors, and test generalization across other geographic regions via spatial cross-validation.

## Repository Contents

| File | Description |
|---|---|
| `group-project.ipynb` | Full notebook: EDA, model training, hyperparameter tuning, evaluation, and learning curves |
| `Predicting_Forest_Cover_Type.pdf` | Final written report with detailed methodology, results, and references |

## Tech Stack

Python (Anaconda 3.13.5), pandas, NumPy, scikit-learn, Matplotlib/Seaborn, TensorFlow/Keras (Residual Neural Network). Data was pulled via `kagglehub`. Models were trained on macOS (Apple ARM64, 8 CPU cores, 16 GB RAM); TensorFlow ran on CPU only (no GPU acceleration).

## References

- Blackard, J. (1998). *Covertype* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N
- Blackard, J. A., & Dean, D. J. (1999). Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables. *Computers and Electronics in Agriculture, 24*(3), 131–151.
- Bonham, C. D. (2013). *Measurements for terrestrial vegetation* (2nd ed.). Wiley-Blackwell.
- Food and Agriculture Organization of the United Nations. (2020). *Global Forest Resources Assessment 2020: Main report.*
- UCI Machine Learning. (n.d.). *Forest cover type dataset* [Data set]. Kaggle.
- USDA Forest Service. (2020). *Forest Inventory and Analysis National Program.*

## Authors

Group 6 — Jane Sanchez, Julie Huynh, Jingda Li, Jessy Donelle, Jeremy May
WatSpeed Introduction to Machine Learning
