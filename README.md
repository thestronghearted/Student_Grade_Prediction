# Student Grade Prediction: early warning for students at risk of failing

ML Zoomcamp 2026 midterm project.

## Problem description

Schools usually find out that a student is failing only after the mid-year grades come in. By then there is little time left to help. This project predicts **at the start of the school year** whether a secondary-school student is likely to **fail the final Portuguese exam** (final grade `G3` below 10 out of 20).

The model uses only information a school already has at the start of the year: demographics, family background, study habits, past failures and absences. The mid-year grades `G1` and `G2` are **deliberately excluded**. They would make the task almost trivial, and they aren't available when early intervention matters most.

**How it is used:** a teacher or counsellor sends a student's profile to a web service and gets back:

```json
{"fail_probability": 0.653, "at_risk": true}
```

Flagged students can be offered tutoring, extra school support or a meeting with their parents early in the year.

## Dataset

[UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (P. Cortez and A. Silva, 2008). The project uses the Portuguese language course file: 649 students from two Portuguese schools, 30 features plus three grades.

The data is committed to the repo at `data/student-por.csv` (`;`-separated), so nothing needs to be downloaded.

- Target: `at_risk = 1` if `G3 < 10` (about 15% of students, so the classes are imbalanced)
- Metric: ROC AUC

## Approach and results

The full analysis is in [`notebook.ipynb`](notebook.ipynb):

1. **Data preparation**: no missing values or duplicates. The binary target is created, and `G1`, `G2` and `G3` are dropped. Categorical features are one-hot encoded with `DictVectorizer`.
2. **EDA**: ranges and distributions of every feature, analysis of the target, correlations, risk rate per category and mutual information. The strongest signals are past `failures`, wanting higher education (`higher`), `school`, study time and alcohol use.
3. **Model training and tuning**: 5-fold stratified CV on the training set (80% of the data).

| Model | Tuned parameters | CV ROC AUC |
|---|---|---|
| Logistic regression | `C` | 0.837 |
| Decision tree | `max_depth`, `min_samples_leaf` | 0.823 |
| Random forest | `max_depth`, `min_samples_leaf`, `n_estimators` | 0.870 |
| **XGBoost** | `learning_rate`, `max_depth`, `min_child_weight` | **0.872** |

The final model is **XGBoost** (`learning_rate=0.01, max_depth=3, min_child_weight=1, n_estimators=200`). On the held-out test set it scores **ROC AUC ≈ 0.80**, against 0.76 for logistic regression.

**Decision threshold = 0.2.** Missing an at-risk student costs more than a false alarm. At 0.2 the model finds about 60% of at-risk students while flagging about a quarter of the class. At the default of 0.5 it would find only about 15%.

## Repository structure

| File | Purpose |
|---|---|
| `notebook.ipynb` | Data preparation, EDA, feature importance, model selection and tuning |
| `train.py` | Trains the final model and saves it to `model.bin` |
| `model.bin` | Trained pipeline (`DictVectorizer` + XGBoost) and threshold |
| `predict.py` | Flask web service that serves the model |
| `test_predict.py` | Sends a sample student to the service |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies, managed with [uv](https://docs.astral.sh/uv/) |
| `Dockerfile` | Container for the web service |
| `data/student-por.csv` | Dataset |

## How to run it

### 1. Set up the environment (uv)

Install uv if you don't have it:

```bash
pip install uv
```

Then create the virtual environment and install the exact locked dependencies:

```bash
git clone https://github.com/thestronghearted/Student_Grade_Prediction.git
cd Student_Grade_Prediction
uv sync
```

This creates a `.venv` folder. Either prefix commands with `uv run` (as below), or activate the environment with `source .venv/bin/activate` (`.venv\Scripts\activate` on Windows).

### 2. Train the model

```bash
uv run python train.py
```

This prints the test ROC AUC and writes `model.bin`. A trained `model.bin` is already committed, so this step is optional.

To re-run the notebook:

```bash
uv run jupyter notebook notebook.ipynb
```

### 3. Run the web service locally

```bash
uv run gunicorn --bind 0.0.0.0:9696 predict:app
```

In a second terminal:

```bash
uv run python test_predict.py
# {'at_risk': True, 'fail_probability': 0.653}
```

On Windows, gunicorn doesn't work. Use `uv run waitress-serve --listen=0.0.0.0:9696 predict:app` (after `uv add waitress`), or use Docker.

### 4. Run with Docker

```bash
docker build -t student-risk .
docker run -it --rm -p 9696:9696 student-risk
```

Then, in another terminal:

```bash
uv run python test_predict.py
```

Or without Python:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"school":"GP","sex":"F","age":16,"address":"U","famsize":"GT3","Pstatus":"T","Medu":4,"Fedu":4,"Mjob":"teacher","Fjob":"services","reason":"reputation","guardian":"mother","traveltime":1,"studytime":3,"failures":0,"schoolsup":"no","famsup":"yes","paid":"no","activities":"yes","nursery":"yes","higher":"yes","internet":"yes","romantic":"no","famrel":5,"freetime":3,"goout":2,"Dalc":1,"Walc":1,"health":5,"absences":2}'
# {"at_risk":false,"fail_probability":0.049}
```

## API

`POST /predict` takes one student as a JSON object with the 30 input features below and returns `fail_probability` and `at_risk`. `GET /ping` is a health check.

| Feature | Description |
|---|---|
| `school` | `GP` (Gabriel Pereira) or `MS` (Mousinho da Silveira) |
| `sex`, `age` | `F`/`M`; age 15–22 |
| `address` | `U` urban, `R` rural |
| `famsize`, `Pstatus` | `LE3`/`GT3` family size; parents living `T`ogether or `A`part |
| `Medu`, `Fedu` | Mother's and father's education (0 none … 4 higher) |
| `Mjob`, `Fjob` | `teacher`, `health`, `services`, `at_home`, `other` |
| `reason`, `guardian` | Reason for choosing the school; student's guardian |
| `traveltime`, `studytime` | 1–4 scales |
| `failures` | Number of past class failures (0–3) |
| `schoolsup`, `famsup`, `paid` | Extra school support, family support, paid classes (`yes`/`no`) |
| `activities`, `nursery`, `higher`, `internet`, `romantic` | `yes`/`no` |
| `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health` | 1–5 scales (`Dalc`/`Walc` = workday/weekend alcohol use) |
| `absences` | Number of school absences |
