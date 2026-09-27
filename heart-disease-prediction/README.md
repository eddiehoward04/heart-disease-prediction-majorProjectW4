# Heart Disease Prediction — Capstone Project

**Focus:** Full ML pipeline + documentation + presentation
**Goal:** Predict whether a patient is likely to have heart disease, using classic tabular ML.

## 1. Dataset

Source: [Kaggle — Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

1. Download the CSV from the link above.
2. Save it as `data/heart.csv` in this project folder.

Typical columns include `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure),
`chol` (cholesterol), `fbs` (fasting blood sugar), `restecg`, `thalach` (max heart rate),
`exang` (exercise-induced angina), `oldpeak`, `slope`, `ca`, `thal`, and a target/label column.
`main.py` auto-detects the label column name and converts it to a clean binary target
(0 = no disease, 1 = disease).

## 2. Project structure

```
heart-disease-prediction/
├── data/               # put heart.csv here (not committed to git)
├── models/             # trained models saved here after running
├── outputs/            # plots + metrics saved here after running
├── main.py             # the full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. Setup

```bash
# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 4. Run the pipeline

```bash
python main.py
```

This will:
1. **Load** `data/heart.csv`
2. **Clean** the data — standardize column names, fix/binarize the target, fill missing values
3. **Scale** features with `StandardScaler` and split into train/test (80/20, stratified)
4. **Train** both a Logistic Regression model and a Decision Tree
5. **Evaluate** each model on accuracy, precision, recall, F1, and ROC-AUC
6. **Save**:
   - `outputs/confusion_matrix_<model>.png`
   - `outputs/roc_curve_<model>.png`
   - `outputs/model_comparison.csv` (side-by-side metrics table)
   - `models/<model>.joblib` and `models/scaler.joblib` (for reuse/inference)

## 5. Interpreting the results

- **Accuracy** — overall % correct, but can be misleading if classes are imbalanced.
- **Precision** — of patients predicted "has disease," how many actually did (matters if false
  alarms are costly, e.g. unnecessary further testing).
- **Recall** — of patients who actually have disease, how many did we catch (matters most in a
  medical screening context, where missing a case is worse than a false alarm).
- **ROC curve / AUC** — how well the model separates the two classes across all thresholds; closer
  to 1.0 is better, 0.5 is random guessing.
- **Confusion matrix** — the raw breakdown of true/false positives and negatives.

In practice, Logistic Regression tends to be a strong, well-calibrated baseline for this dataset,
while the Decision Tree can capture non-linear patterns but is more prone to overfitting — compare
their metrics side by side in `outputs/model_comparison.csv`.

## 6. For the presentation

Suggested structure:
1. Problem statement — why early heart disease prediction matters
2. Dataset overview (features, size, target distribution)
3. Pipeline walkthrough (cleaning → scaling → training → evaluation)
4. Results: confusion matrices, ROC curves, metric comparison table
5. Limitations (dataset size, features are proxies not diagnoses) and next steps
   (cross-validation, more models, hyperparameter tuning, feature importance)

## 7. Next steps / extensions

- Add k-fold cross-validation for more robust metrics
- Try Random Forest / Gradient Boosting for comparison
- Add feature importance / coefficient plots for interpretability
- Wrap the trained model in a small Flask/Streamlit demo app
