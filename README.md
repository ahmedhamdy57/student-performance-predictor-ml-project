# Student Performance Predictor 🎓

A small practice ML project that predicts a student's **Performance Index**
(10–100) using multiple linear regression, with a Streamlit app for
interactive predictions.

## Dataset

[Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
on Kaggle — 10,000 synthetic student records.

**Features**
- `Hours Studied`
- `Previous Scores`
- `Extracurricular Activities` (Yes/No)
- `Sleep Hours`
- `Sample Question Papers Practiced`

**Target**
- `Performance Index` (10–100)

> Note: the dataset is synthetic and created for illustrative purposes;
> relationships between variables may not reflect real-world scenarios.

## Project structure

```
.
├── train.py              # Downloads data, trains & evaluates the model, saves model.pkl
├── app.py                 # Streamlit app for interactive predictions
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Kaggle authentication: `kagglehub` needs a Kaggle API token. Either:
   - Place `kaggle.json` (from your Kaggle account settings) at
     `~/.kaggle/kaggle.json`, or
   - Set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

3. Train the model (downloads the dataset automatically and saves
   `model.pkl` + `model_metadata.json`):

   ```bash
   python train.py
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Model

A simple `LinearRegression` (scikit-learn) trained on an 80/20 train/test
split. `train.py` prints R², MAE, and RMSE on the test set and stores them
in `model_metadata.json`, which the Streamlit app displays under
"Model details".

## Disclaimer

This is a practice/portfolio project using a synthetic dataset — predictions
are for demonstration purposes only.
