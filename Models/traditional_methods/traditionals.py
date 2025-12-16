import time
import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


# ----------------------------
## 1. Data Loading and Split (Train: 22,500 | Dev: 2,500 | Test: 25,000)
# ----------------------------
print("Loading and splitting IMDB dataset...")
dataset = load_dataset("imdb")

# Combine the original 'train' set for splitting
train_dev_texts = dataset["train"]["text"]
train_dev_labels = dataset["train"]["label"]

# Split the 25,000 original train samples into 22,500 TRAIN and 2,500 DEV
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    train_dev_texts, 
    train_dev_labels, 
    test_size=2500, 
    random_state=42, 
    stratify=train_dev_labels
)

# Test set remains the original 25,000 samples
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print(f"Data splits: Train={len(train_texts)}, Dev={len(dev_texts)}, Test={len(test_texts)}")

# ----------------------------
## 2. TF-IDF Vectorization
# ----------------------------
print("\nFitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=50000)

# Fit only on the TRAIN set and transform all sets
X_train = vectorizer.fit_transform(train_texts) 
X_test = vectorizer.transform(test_texts)

y_train = np.array(train_labels)
y_test = np.array(test_labels)


# ----------------------------
## 3. Hyperparameter Grids
# ----------------------------

param_grid_svm = {
    "C": [0.125, 0.5, 1, 2, 8, 32, 128, 512, 2048, 8192],
}

param_grid_lr = {
    "C": [0.001, 0.01, 0.1, 1],
}

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": [0.08, 0.15, 0.30]
}

nb_model = MultinomialNB()


# ----------------------------
## 4. Model Definitions
# ----------------------------
models = {
    # --- FIX: Increased max_iter for LinearSVC to prevent ConvergenceWarning ---
    "svm": (svm.LinearSVC(dual=False, max_iter=10000), param_grid_svm), 
    # -------------------------------------------------------------------------
    "lr": (LogisticRegression(max_iter=1000, n_jobs=-1), param_grid_lr),
    "rf": (RandomForestClassifier(n_jobs=-1), param_grid_rf),
    "nb": (nb_model, None)
}

results = []


# ----------------------------
## 5. Train + Tune + Evaluate
# ----------------------------
for model_name, (model, param_grid) in models.items():

    print(f"\n===== {model_name.upper()} =====")

    if param_grid is not None:
        print("Running GridSearchCV (5-fold CV on 22,500 train samples)...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1
        )

        start_train = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - start_train

        best_model = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")

    else:
        # Naive Bayes (no grid)
        print("Training Naive Bayes...")
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        best_model = model

    # Predict and Evaluate on the 25,000 TEST set
    start_pred = time.time()
    y_pred = best_model.predict(X_test)
    pred_time = time.time() - start_pred

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "model": model_name,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "train_time": train_time,
        "predict_time": pred_time
    })


results_df = pd.DataFrame(results)
print("\n===== FINAL RESULTS (Evaluated on 25,000 TEST samples) =====")
print(results_df)

results_df.to_csv("traditional_results.csv", index=False)