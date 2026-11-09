"""Titanic survival prediction with logistic regression (Python / scikit-learn).

Reads train.csv and test.csv from DATA_DIR (default: ../data), engineers a few
features, fits a logistic regression, and prints every step to the terminal.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def add_features(df, ref):
    """Create new columns and fill missing values using stats from the training set."""
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    title = title.replace(TITLE_MAP)
    df["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Age"] = df["Age"].fillna(df["Title"].map(ref["age_by_title"])).fillna(ref["age_all"])
    df["Fare"] = df["Fare"].fillna(ref["fare"])
    df["Embarked"] = df["Embarked"].fillna(ref["embarked"])
    return df


def to_matrix(df, columns=None):
    """One-hot encode and keep a fixed set of columns so train/test line up."""
    X = pd.get_dummies(
        df[["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone"]],
        columns=["Pclass", "Sex", "Embarked", "Title"],
        dtype=float,
    )
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def main():
    banner("STEP 1: Load data")
    print(f"Looking for data in: {DATA_DIR}")
    try:
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")
    except FileNotFoundError as err:
        sys.exit(f"ERROR: {err}\nDownload train.csv and test.csv from Kaggle into the data folder (see README).")
    print(f"train.csv: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"test.csv:  {test.shape[0]} rows, {test.shape[1]} columns")
    print("\nFirst 5 rows of train.csv:")
    print(train.head().to_string())

    banner("STEP 2: Explore the training data")
    print(f"Overall survival rate: {train['Survived'].mean():.1%}")
    print("\nSurvival rate by sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by passenger class:")
    print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())
    print("\nMissing values in train.csv (columns with any):")
    missing = train.isna().sum()
    print(missing[missing > 0].to_string())

    banner("STEP 3: Clean and engineer features")
    train["Title"] = (
        train["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip().replace(TITLE_MAP)
    )
    train["Title"] = train["Title"].where(train["Title"].isin(COMMON_TITLES), "Rare")
    ref = {
        "age_by_title": train.groupby("Title")["Age"].median(),
        "age_all": train["Age"].median(),
        "fare": train["Fare"].median(),
        "embarked": train["Embarked"].mode()[0],
    }
    print("Added Title (extracted from Name; rare titles grouped as 'Rare')")
    print("Added FamilySize = SibSp + Parch + 1 and IsAlone")
    print(f"Filled missing Age with the median age of the passenger's Title (overall median {ref['age_all']:.1f})")
    print(f"Filled missing Fare with the train median ({ref['fare']:.2f})")
    print(f"Filled missing Embarked with the most common port ({ref['embarked']})")
    train = add_features(train, ref)
    print(f"Remaining missing values in model columns: {int(train[['Age', 'Fare', 'Embarked']].isna().sum().sum())}")
    print("Dropped Cabin, Ticket, Name and PassengerId (not used by the model)")
    print("One-hot encoded Pclass, Sex, Embarked and Title")

    X = to_matrix(train)
    y = train["Survived"]
    print(f"Model matrix: {X.shape[0]} rows x {X.shape[1]} features")

    banner("STEP 4: Fit logistic regression")
    X_fit, X_hold, y_fit, y_hold = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split train.csv 80/20 -> {len(X_fit)} fit rows, {len(X_hold)} hold-out rows (random_state=42)")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_fit, y_fit)
    coefs = pd.Series(model[-1].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("\nLargest coefficients (standardized features):")
    print(coefs.head(6).round(3).to_string())

    banner("STEP 5: Accuracy")
    print(f"Training accuracy:          {accuracy_score(y_fit, model.predict(X_fit)):.3f}")
    print(f"Hold-out accuracy:          {accuracy_score(y_hold, model.predict(X_hold)):.3f}  (20% of train.csv, never used to fit)")
    # Refit on all labeled data for the final test predictions
    model.fit(X, y)
    print(f"Training accuracy (refit on all {len(X)} rows): {accuracy_score(y, model.predict(X)):.3f}")

    banner("STEP 6: Predict on test.csv")
    test_feat = add_features(test, ref)
    X_test = to_matrix(test_feat, columns=X.columns)
    preds = model.predict(X_test)
    print(f"Predicted {len(preds)} passengers: {int(preds.sum())} survive, {int(len(preds) - preds.sum())} do not")
    out = DATA_DIR / "predictions_python.csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(out, index=False)
    print(f"Saved predictions to {out}")

    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy_score(test['Survived'], preds):.3f}")
    else:
        print("Note: Kaggle's test.csv has no Survived column, so test accuracy cannot be computed here.")
        print("The hold-out accuracy above is the honest estimate of performance on unseen passengers.")
        print("Upload predictions_python.csv to Kaggle for the official test score.")


if __name__ == "__main__":
    main()
