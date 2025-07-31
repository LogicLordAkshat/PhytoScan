# === IMPORTS ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from google.colab import files

# === LOAD ALL 4 CSVs ===
disease_char = pd.read_csv("Crop_Disease_Characteristics.csv")
insect_char = pd.read_csv("Crop_Insect_Characteristics.csv")
disease_train = pd.read_csv("Crop_Disease_TrainingData.csv")
insect_train = pd.read_csv("Crop_Insect_TrainingData.csv")

# === ENCODE YES/NO AS 1/0 ===
def encode_boolean(df):
    return df.replace({'Yes': 1, 'No': 0})

disease_train = encode_boolean(disease_train)
insect_train = encode_boolean(insect_train)

# === PREPARE X, y ===
def prepare_xy(df, label_col="disease_present"):
    X = df.drop(columns=[label_col])
    y = df[label_col].replace({'Yes': 1, 'No': 0}) if df[label_col].dtype == object else df[label_col]
    return X.values, y.values

# === TRAIN TABNET CLASSIFIER & SAVE ===
def train_tabnet(X, y, model_name):
    print(f"\n🔧 Training {model_name} model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = TabNetClassifier(verbose=0, seed=42, device_name='cuda' if torch.cuda.is_available() else 'cpu')

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=100,
        patience=10,
        batch_size=128,
        virtual_batch_size=64
    )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ {model_name} Accuracy: {acc:.4f}")
    print("🧾 Classification Report:\n", classification_report(y_test, y_pred))

    # Sample inference
    print(f"\n🔍 {model_name} Sample Inference:")
    sample_input = X_test[0].reshape(1, -1)
    prediction = clf.predict(sample_input)[0]
    print("→ Prediction:", "PRESENT" if prediction == 1 else "NOT PRESENT")

    # === SAVE MODEL ===
    model_filename = model_name.replace(" ", "_").lower()  # e.g., "crop_disease"
    clf.save_model(model_filename)
    print(f"💾 Model saved to {model_filename}.zip")

    return clf

# === RUN BOTH MODELS ===
X_disease, y_disease = prepare_xy(disease_train, label_col="disease_present")
X_insect, y_insect = prepare_xy(insect_train, label_col="insect_present")

disease_model = train_tabnet(X_disease, y_disease, model_name="Crop Disease")
insect_model = train_tabnet(X_insect, y_insect, model_name="Crop Insect")
