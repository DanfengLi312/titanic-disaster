# Titanic Disaster – Survival Prediction (Python & R, Dockerized)

Logistic regression models that predict which passengers survived the Titanic, built twice:
once in **Python** (scikit-learn) and once in **R** (`glm`). Each runs in its own Docker container
and prints every step (loading, exploration, cleaning, modeling, accuracy, test predictions) to the terminal.

## Repository layout

```
.
├── README.md
├── CODEOWNERS
├── .gitignore
└── src
    ├── data/              <- you put train.csv and test.csv here (git-ignored)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed **and running**
- Git
- A free [Kaggle](https://www.kaggle.com) account (only needed to download the data)

You do **not** need Python or R installed locally.

## Step 1 – Clone the repository

```bash
git clone https://github.com/DanfengLi312/titanic-disaster.git
cd titanic-disaster
```

## Step 2 – Download the data

The dataset is not stored in this repo. 

1. Go to <https://www.kaggle.com/competitions/titanic/data> and log in (you may need to click **Join Competition** / accept the rules).
2. Click **Download All** and unzip the file.
3. Copy **`train.csv`** and **`test.csv`** into the `src/data/` folder.

Check that it worked (from the repo root):

```bash
ls src/data
# expected: train.csv  test.csv
```

## Step 3 – Run the Python container

Run all commands from the **repo root**.

```bash
docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/data" titanic-python
```

On Windows PowerShell use `${PWD}` instead of `$(pwd)`:
`docker run --rm -v "${PWD}/src/data:/data" titanic-python`

## Step 4 – Run the R container

```bash
docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/data" titanic-r
```

The first R build takes a few minutes (it downloads the R base image and installs packages); later builds are cached.

## What you will see

Both scripts print the same six sections:

1. **Load data** – file locations, shapes, first rows
2. **Explore** – survival rate overall, by sex and by class; missing values
3. **Clean and engineer features** – `Title` extracted from `Name`, `FamilySize`, `IsAlone`; missing `Age` filled by title median, `Fare`/`Embarked` filled with median/mode; unused columns dropped
4. **Fit logistic regression** – 80/20 train/test split of `train.csv`, largest coefficients
5. **Accuracy** – training accuracy and test accuracy
6. **Predict on `test.csv`** – prediction counts, and a predictions file saved to `src/data/predictions_python.csv` / `predictions_r.csv`

## How test accuracy is measured

Kaggle's `test.csv` **does not include the `Survived` column**, so its accuracy cannot be computed locally.
To still measure accuracy on unseen data, each script splits `train.csv` 80/20 (fixed random seed):

- **Training set (80%)** – used to fit the model; the script prints **training accuracy**.
- **Test set (20%)** – never used for fitting; the script prints **TEST accuracy**.

The final model is then refit on all of `train.csv` and used to predict Kaggle's `test.csv`; the predictions are saved
to `src/data/predictions_*.csv` and can be uploaded to Kaggle (**Submit Predictions**) for the official score.
If you supply a `test.csv` that does contain `Survived`, the scripts also print accuracy on it automatically.

## Model details

| | Python | R |
|---|---|---|
| Library | scikit-learn 1.5.2, pandas 2.2.3 | base `glm` (binomial), dplyr |
| Base image | `python:3.12-slim` | `rocker/r-ver:4.4.1` |
| Features | Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone | same, minus Sex (redundant with Title) |

## Troubleshooting

- **`Cannot connect to the Docker daemon`** – start Docker Desktop and retry.
- **`ERROR: ... train.csv not found`** – the `src/data/` folder must contain `train.csv` and `test.csv`, and you must run `docker run` from the repo root so `$(pwd)/src/data` resolves correctly.
- **Predictions file missing** – it is written to `src/data/` on your machine (the folder is mounted into the container) and is git-ignored.
