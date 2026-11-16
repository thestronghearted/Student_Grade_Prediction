"""Train the final at-risk student model and save it to model.bin.

Usage:
    python train.py
"""

import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_PATH = "data/student-por.csv"
MODEL_PATH = "model.bin"
SEED = 1

# Final parameters chosen in notebook.ipynb (5-fold CV on the full train set)
XGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=-1,
)

# Lower than 0.5 because missing an at-risk student is worse than a false alarm
THRESHOLD = 0.2


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    # Target: student fails the final grade (G3 < 10 out of 20)
    df["at_risk"] = (df["G3"] < 10).astype(int)
    # G1/G2 are mid-year grades: drop them so the model works at the START of the year
    df = df.drop(columns=["G1", "G2", "G3"])
    return df


def train(df_train):
    y = df_train["at_risk"].values
    records = df_train.drop(columns=["at_risk"]).to_dict(orient="records")
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        xgb.XGBClassifier(**XGB_PARAMS),
    )
    pipeline.fit(records, y)
    return pipeline


def main():
    df = load_data()
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["at_risk"]
    )

    pipeline = train(df_full_train)

    y_test = df_test["at_risk"].values
    test_records = df_test.drop(columns=["at_risk"]).to_dict(orient="records")
    y_pred = pipeline.predict_proba(test_records)[:, 1]
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred):.3f}")

    # Retrain on all data for the deployed model
    pipeline = train(df)

    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump({"pipeline": pipeline, "threshold": THRESHOLD}, f_out)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
