# Instagram Fake Account Detector

A small Flask web app that predicts whether an Instagram account is likely fake using a neural network trained on account features.

## Project structure

- **app.py** – Flask app: model load/train, prediction routes
- **model/** – Saved model (`fake_account_model.keras`) and scaler (`scaler.joblib`), created when you train
- **data/train.csv** – Training data (required for training)
- **data/test.csv** – Optional, for reference
- **templates/** – `index.html` (form), `result.html` (prediction result)
- **static/style.css** – Styles

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   # or: venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   # If you see SSL certificate errors, use:
   # pip install --trusted-host pypi.org --trusted-host files.pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

2. Training data is in `data/train.csv`. The app supports two CSV formats: legacy (`profile pic`, `#posts`, `#followers`, `#follows`, `description length`, `external URL`, `private`, `fake`) or app columns (see below). To (re)train, run the app and visit **http://localhost:5001/train** or use the link on the home page.

3. Run the app:

   - **macOS/Linux:** `./run_macos.sh` (or `python app.py`)
   - **Windows:** `run_windows.bat` (or `python app.py`)

4. Open **http://localhost:5001** in your browser. Fill the form and click **Analyze Account**. Train the model first if you see “Model not loaded”.

## Features (form / CSV columns)

- `account_age_days` – Account age in days  
- `num_posts` – Number of posts  
- `num_followers` – Follower count  
- `num_following` – Following count  
- `has_profile_pic` – 0 or 1  
- `username_length` – Length of username  
- `is_private` – 0 or 1  
- `bio_length` – Bio length in characters  
- `has_external_url` – 0 or 1  

Label in `train.csv`: `is_fake` (0 = real, 1 = fake).
