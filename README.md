# Comment Category Prediction Challenge

> Kaggle Competition: [Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)

---

## 📌 Problem Statement

The goal of this competition is to classify online comments into one of **4 categories (labels: 0, 1, 2, 3)** based on comment text and associated metadata such as upvotes, downvotes, emoticons, demographic flags, and date information.

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `train.csv` | Training data with labels |
| `test.csv` | Test data (no labels) |
| `Sample.csv` | Submission format template |

**Key columns:**
- `comment` — Raw text of the comment
- `created_date` — Timestamp of the comment
- `upvote`, `downvote` — Engagement metrics
- `emoticon_1`, `emoticon_2`, `emoticon_3` — Emoticon counts
- `if_1`, `if_2` — Interaction flags
- `race`, `religion`, `gender` — Demographic context
- `disability` — Boolean flag
- `label` — Target variable (0–3)

---

## 🔧 Approach

### 1. Data Cleaning
- Filled missing `comment` values with empty strings
- Replaced `"none"` in `race`, `religion`, `gender` with `0` and coerced to numeric
- Filled remaining NaN values with `0`

### 2. Exploratory Data Analysis (EDA)
- **Label Distribution** — Class `0` is significantly more frequent (class imbalance observed)
- **Upvote vs Downvote** — Most comments have low engagement; outliers present
- **Comment Length Distribution** — Majority of comments are short; a few are very long

### 3. Feature Engineering
- Extracted `year`, `month`, `day` from `created_date`
- Computed `comment_length` (character count)
- Computed `word_count` (whitespace-split tokens)

### 4. Preprocessing Pipeline
Used `ColumnTransformer` with:
- **TF-IDF Vectorizer** (`max_features=20000`) applied to the `comment` column
- **Passthrough** for numeric features: upvote, downvote, emoticons, flags, demographics, engineered date and text features

### 5. Train/Validation Split
- 80% training / 20% validation
- `random_state=42`, `stratify=y`

---

## 🤖 Models Used

| Model | Library | Key Parameters |
|-------|---------|----------------|
| **Logistic Regression** | `sklearn` | `max_iter=300` |
| **K-Nearest Neighbors (KNN)** | `sklearn` | Default parameters |
| **Random Forest** | `sklearn` | `n_estimators=200` |
| **XGBoost** | `xgboost` | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |
| **LightGBM** *(Final Model)* | `lightgbm` | `n_estimators=300`, `learning_rate=0.05`, `num_leaves=50`, `subsample=0.8`, `colsample_bytree=0.8` |

### Additional Models Explored (Milestone Sections)
| Model | Purpose |
|-------|---------|
| **Multinomial Naive Bayes** | Baseline text classification (Milestones 3, 4, 5) |
| **SGD Classifier** | Imported for potential use |
| **SVM (SVC)** | Imported for potential use |
| **Gradient Boosting** | Imported for potential use |
| **AdaBoost** | Variance analysis (Milestone 4) |
| **MLP Classifier (Neural Network)** | Architecture and regularization experiments (Milestone 4) |
| **Decision Tree** | Imported for potential use |

---

## ⚙️ Hyperparameter Tuning

### XGBoost (RandomizedSearchCV)
```python
param_grid = {
    "model__n_estimators": [300, 400],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [4, 6, 8],
    "model__subsample": [0.7, 0.9, 1]
}
# n_iter=4, cv=3, random_state=42
```

### Random Forest (RandomizedSearchCV — Milestone 4)
```python
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15]
}
# n_iter=5, cv=3, random_state=2306
```

### Logistic Regression (GridSearchCV — Milestone 5)
```python
param_grid = {"C": [0.1, 1, 10]}
# solver="liblinear", cv=3
```

---

## 🏆 Final Model

**LightGBM** was selected as the final model based on validation accuracy comparison across all trained models.

```python
final_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=50,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

Predictions were generated on the test set and saved to `submission.csv`.

---

## 📦 Libraries & Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

---

## 📁 Project Structure

```
├── train.csv
├── test.csv
├── Sample.csv
├── notebook.ipynb        # Main solution notebook
├── submission.csv        # Final predictions
└── README.md
```

---

## 📊 Evaluation Metric

**Accuracy Score** — primary metric used throughout model training and comparison.
Macro F1-score was also tracked during milestone experiments.

---

## 📝 Notes

- All preprocessing objects (TF-IDF, scalers, encoders) are fit **only on training data** and then applied to validation/test data to prevent data leakage.
- The notebook also contains milestone quiz answers (commented out) covering topics like TF-IDF feature counts, tokenization, scaling, and model evaluation from the course curriculum.
