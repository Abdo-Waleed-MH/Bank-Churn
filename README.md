# Bank Customer Churn Predictor

An end-to-end ML project: a tuned Random Forest predicts whether a bank
customer is about to churn, served through a small Flask API and a single-page
web front end.

**Model performance (held-out test set):** Accuracy 0.839 · Precision 0.592 ·
Recall 0.666 · **F1-Score 0.627**

## Project structure

```
.
├── Bank_Customer_Churn_Prediction_ML_Pipeline.ipynb   # full analysis + training notebook
├── Churn_Modelling.csv                                # dataset
├── train_model.py                                     # regenerates model/bank_churn_model.pkl
├── model/
│   └── bank_churn_model.pkl                            # trained preprocessor + Random Forest
├── app.py                                              # Flask app (/ and /predict)
├── templates/index.html                                # web page
├── static/style.css, static/app.js
├── requirements.txt
└── Procfile                                            # for Render / Railway / Heroku-style hosts
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# only needed once, or whenever you want to retrain:
python train_model.py

python app.py
```

Then open **http://127.0.0.1:5000**.

## Deploying (no server management needed)

This is a small Flask app, so any free-tier PaaS works. **Render** is the
simplest:

1. Push this repo to GitHub (see below).
2. On [render.com](https://render.com) → **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `gunicorn app:app`
4. Deploy — Render gives you a public URL.

Railway, Fly.io, or PythonAnywhere work the same way (any host that runs
`gunicorn app:app` from this repo).

## Push to GitHub

```bash
cd churn-project
git init
git add .
git commit -m "Bank customer churn prediction: pipeline + Flask deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

(Create the empty repo on GitHub first, or use `gh repo create` if you have
the GitHub CLI installed.)

## Retraining / improving the model

`train_model.py` reproduces the notebook's pipeline (cleaning → feature
engineering → preprocessing → tuned Random Forest) end to end. See the
notebook's final section for ideas on pushing the F1-score higher — SMOTE,
threshold tuning, LightGBM/CatBoost, and a wider hyperparameter search are
the highest-leverage next steps.
