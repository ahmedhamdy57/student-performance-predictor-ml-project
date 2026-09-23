# Student Performance Predictor 🎓

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-predictor-ml-project-dar3ff6dnfzyl3yu4etbt.streamlit.app/)
 
A small practice ML project that predicts a student's **Performance Index**
(10–100) using multiple linear regression, with a Streamlit app for
interactive predictions.
 
🔗 **[Try the live demo](https://student-performance-predictor-ml-project-dar3ff6dnfzyl3yu4etbt.streamlit.app/)**


 ML project that predicts a student's **Performance Index**
(10–100) using multiple linear regression, with a Streamlit app for
interactive predictions.

**Features**
- `Hours Studied`
- `Previous Scores`
- `Extracurricular Activities` (Yes/No)
- `Sleep Hours`
- `Sample Question Papers Practiced`

## Project structure

```
.
├── train_student_performance_model.py   # Downloads data, trains & evaluates the model, saves model.pkl
├── student_performance_app.py           # Streamlit app for interactive predictions
├── model.pkl                            # Pre-trained model (used by the live demo)
├── model_metadata.json                  # Feature names, metrics & coefficients
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


