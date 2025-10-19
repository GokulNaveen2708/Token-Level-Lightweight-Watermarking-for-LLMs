import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

MODEL_DIR = "model"
CLF_PATH = os.path.join(MODEL_DIR, "clf_full_features.joblib")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "classification_data", filename)


def get_raw_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, 'data', filename)


def load_prompts(path=None):
    if path is None:
        path = get_data_path("prompts.json")
    with open(path, "r") as f:
        return json.load(f)


def train_classifier(data_path):
    print("🔧 Training classifier with full features (text + z_score + perplexity)...")
    data = load_jsonl(get_data_path(data_path))
    df = pd.DataFrame(data)

    # Features for training
    train_features = ["text"]
    X = df[train_features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Define transformers
    text_vectorizer = TfidfVectorizer(max_features=1000)
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "text")
        ]
    )

    # Build pipeline
    clf = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Fit and evaluate
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("Model trained")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model & vectorizer separately for inference
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, CLF_PATH)
    joblib.dump(text_vectorizer, VEC_PATH)


def predict_on_new_data(unwatermarked_path):
    print(f"\nPredicting on new unwatermarked data: {unwatermarked_path}")
    clf_pipeline = joblib.load(CLF_PATH)  # full pipeline

    data = load_jsonl(unwatermarked_path)
    df = pd.DataFrame(data)

    if "label" not in df.columns:
        df["label"] = 1  # assume unwatermarked

    # Supply placeholder values for missing features
    df["z_score"] = 0.0
    df["perplexity"] = 50.0  # arbitrary neutral value

    # Match training column structure
    X = df[["text"]].rename(columns={"text": "text"})
    y_pred = clf_pipeline.predict(X)
    y_prob = clf_pipeline.predict_proba(X)[:, 1]

    df["predicted_label"] = y_pred
    df["predicted_prob"] = y_prob

    false_positives = df[df["predicted_label"] == 0]
    print(f"\nFalse Positives (wrongly predicted as WM): {len(false_positives)} out of {len(df)}")
    print(false_positives[["prompt_id", "text"]].head(5))

    return df

if __name__ == "__main__":
    # train_classifier("human_vs_model_dataset_gamma_0.5.jsonl")
    predict_on_new_data(get_data_path("human_vs_model_dataset_gamma_0.5.jsonl"))
