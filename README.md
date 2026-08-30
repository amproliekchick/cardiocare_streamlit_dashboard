# CardioCare Heart Disease Risk Dashboard

Streamlit deployment prototype for the BMDS2003 Data Science assignment
(Section 7.0 of the report). Built on top of the models developed and tuned
in `DataScience_Assignment.ipynb` (Sections 4.0–5.0 of the report).

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard itself |
| `heart_disease.csv` | Default dataset (bundled so the app works out of the box) |
| `rf_model.joblib` | The tuned Random Forest model + scaler + selected feature list, saved with `joblib` |
| `model_metrics.json` | Pre-computed test-set metrics for all 4 models (accuracy, AUC, log loss, precision/recall/F1, confusion matrices, feature importances, coefficients) — used to power the "Model Comparison" and "Feature Importance" tabs without re-running the full hyperparameter search inside the app |
| `requirements.txt` | Python dependencies |

## Running it locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Streamlit will open the dashboard in your browser at `http://localhost:8501`.

## Running it on Streamlit Community Cloud

1. Push this folder to a GitHub repository (keep `app.py`, `heart_disease.csv`,
   `rf_model.joblib`, `model_metrics.json` and `requirements.txt` together in
   the same folder).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click
   **New app**.
3. Point it at the repository and set the main file path to `app.py`.
4. Deploy — no other configuration is required.

## Dashboard tabs

1. **Data Explorer** – filter the dataset (age range, gender, smoking status,
   diagnosis) and sort/preview/download the result; live histograms and pie
   charts update as you filter.
2. **EDA** – interactive box plots and the full correlation heatmap
   (encoded features vs. target), matching Section 3.4 of the report.
3. **Feature Importance** – Random Forest importances and Logistic
   Regression coefficients, with an interpretation caution consistent with
   Section 6.2 of the report.
4. **Model Comparison** – switch between accuracy / AUC / log loss / CV
   score, and inspect precision, recall, F1 and confusion matrices for each
   of the four tuned models (Section 5.0).
5. **Risk Prediction** – enter a patient's characteristics through sliders
   and dropdowns and get a live prediction + probability gauge from the
   deployed Random Forest model, with a clinical-use disclaimer.
6. **Live Monitoring** – simulates a real-time patient feed by streaming
   rows from the dataset in batches and updating a running risk chart, to
   demonstrate how the dashboard would behave against a live data source.

## Note on model performance

As documented in Section 6.0 (Results and Discussion) of the report, none of
the four tuned models exceed random-guessing-level test AUC (~0.49–0.52).
The dashboard is built to demonstrate the *deployment pipeline* required by
the assignment; the underlying predictions should not be treated as
clinically validated.
