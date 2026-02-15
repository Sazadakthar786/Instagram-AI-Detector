"""
Instagram Fake Account Detection - Modern AI Dashboard
Stable, Fast, Production-Ready Version
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

app = Flask(__name__)

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_account_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.joblib")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

FEATURES = [
    "account_age_days",
    "num_posts",
    "num_followers",
    "num_following",
    "has_profile_pic",
    "username_length",
    "is_private",
    "bio_length",
    "has_external_url",
]
TARGET = "is_fake"

_LEGACY_COLS = {
    "profile pic",
    "nums/length username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
}

model = None
scaler = None


# -------------------------------------------------------------------
# NORMALIZE DATASET
# -------------------------------------------------------------------
def _normalize_train_df(df):
    cols = list(df.columns)

    target_col = "is_fake" if "is_fake" in cols else "fake"
    if target_col not in cols:
        raise ValueError("train.csv must contain 'fake' or 'is_fake' column.")

    # Canonical schema
    if all(f in cols for f in FEATURES):
        out = df[FEATURES + [target_col]].copy()
        if target_col == "fake":
            out = out.rename(columns={"fake": TARGET})
        return out

    # Legacy schema
    if not _LEGACY_COLS.issubset(set(cols)):
        raise ValueError(
            "train.csv must have canonical columns or legacy schema columns."
        )

    out = pd.DataFrame()
    out["has_profile_pic"] = df["profile pic"].astype(float)
    out["num_posts"] = df["#posts"].astype(float)
    out["num_followers"] = df["#followers"].astype(float)
    out["num_following"] = df["#follows"].astype(float)
    out["bio_length"] = df["description length"].astype(float)
    out["has_external_url"] = df["external URL"].astype(float)
    out["is_private"] = df["private"].astype(float)

    ratio = df["nums/length username"].astype(float)
    out["username_length"] = (ratio * 25).clip(3, 30).astype(int)
    out["account_age_days"] = (df["#posts"].astype(float) * 1.5).clip(1, 400).astype(int)
    out[TARGET] = df[target_col].astype(int)

    return out[FEATURES + [TARGET]]


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def safe_float(val, default=0.0):
    try:
        if val is None or str(val).strip() == "":
            return default
        return float(val)
    except:
        return default


def yes_no_to_int(val):
    if not val:
        return 0
    return 1 if str(val).strip().lower() in ["yes", "1", "true"] else 0


def _build_explanation(features, probability, is_fake):
    pct = probability * 100
    following = features["num_following"]
    followers = features["num_followers"]
    posts = features["num_posts"]

    ratio = following / (followers + 1)

    if is_fake:
        reasons = []
        if ratio > 2:
            reasons.append("follows significantly more accounts than followers")
        if posts < 10:
            reasons.append("very low post count")
        if features["has_profile_pic"] == 0:
            reasons.append("no profile picture")
        if features["bio_length"] == 0:
            reasons.append("empty bio")

        reason_text = ", ".join(reasons[:4]) if reasons else "overall suspicious pattern"

        return f"This account was classified as fake with {pct:.0f}% confidence. Signals detected: {reason_text}."

    reasons = []
    if ratio <= 1.5:
        reasons.append("balanced follower/following ratio")
    if posts >= 20:
        reasons.append("consistent post activity")
    if features["has_profile_pic"] == 1:
        reasons.append("profile picture present")
    if features["bio_length"] > 0:
        reasons.append("bio provided")

    reason_text = ", ".join(reasons[:4]) if reasons else "overall natural profile pattern"

    return f"This account was classified as real. The profile shows: {reason_text}."


# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------
def load_model_and_scaler():
    global model, scaler

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return False

    try:
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return True
    except Exception as e:
        print("Load error:", e)
        return False


def load_metrics():
    if os.path.exists(METRICS_PATH):
        return joblib.load(METRICS_PATH)
    return None


# -------------------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------------------
def train_model():
    global model, scaler

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError("train.csv not found in data folder")

    from tensorflow import keras
    from tensorflow.keras import layers

    df = pd.read_csv(TRAIN_CSV)
    df = _normalize_train_df(df)

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    if len(y) < 10:
        raise ValueError("train.csv must have at least 10 rows.")

    scaler_obj = StandardScaler()
    X_scaled = scaler_obj.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = keras.Sequential([
        layers.Input(shape=(len(FEATURES),)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=0
    )

    # Performance metrics
    y_pred = (model.predict(X_val, verbose=0) >= 0.5).astype(int).flatten()

    metrics = {
        "accuracy": round(accuracy_score(y_val, y_pred) * 100, 2),
        "precision": round(precision_score(y_val, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_val, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_val, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler_obj, SCALER_PATH)
    joblib.dump(metrics, METRICS_PATH)

    scaler = scaler_obj


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train():
    try:
        train_model()
        load_model_and_scaler()
        return render_template("result.html", result={"trained": True})
    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler

    if model is None or scaler is None:
        if not load_model_and_scaler():
            train_model()
            load_model_and_scaler()

    try:
        username = (request.form.get("username") or "").strip()
        bio = (request.form.get("bio") or "").strip()

        if not request.form.get("num_posts") or not request.form.get("num_followers") or not request.form.get("num_following"):
            return render_template("result.html", error="Please fill in posts, followers, and following.")

        features = {
            "account_age_days": 180,
            "num_posts": safe_float(request.form.get("num_posts")),
            "num_followers": safe_float(request.form.get("num_followers")),
            "num_following": safe_float(request.form.get("num_following")),
            "has_profile_pic": yes_no_to_int(request.form.get("has_profile_pic")),
            "username_length": max(1, len(username)),
            "is_private": yes_no_to_int(request.form.get("is_private")),
            "bio_length": len(bio),
            "has_external_url": yes_no_to_int(request.form.get("has_external_url")),
        }

        X = np.array([[features[col] for col in FEATURES]], dtype=float)
        X_scaled = scaler.transform(X)

        probability = float(model.predict(X_scaled, verbose=0)[0][0])
        is_fake = 1 if probability >= 0.5 else 0
        prob_pct = round(probability * 100, 2)

        risk_level = "Low" if prob_pct < 30 else "Medium" if prob_pct <= 70 else "High"

        explanation = _build_explanation(features, probability, is_fake)
        model_metrics = load_metrics()

        return render_template(
            "result.html",
            result={
                "is_fake": is_fake,
                "probability": prob_pct,
                "risk_level": risk_level,
                "followers": int(features["num_followers"]),
                "following": int(features["num_following"]),
                "posts": int(features["num_posts"]),
                "username_length": features["username_length"],
                "bio_length": features["bio_length"],
                "explanation": explanation,
                "model_metrics": model_metrics,
            },
        )

    except Exception as e:
        return render_template("result.html", error=f"Prediction Error: {str(e)}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    load_model_and_scaler()
    app.run(host="0.0.0.0", port=5001, debug=True)
