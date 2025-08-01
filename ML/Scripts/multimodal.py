# === IMPORTS ===
from ultralytics import YOLO
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import os

# === FILE PATHS ===
image_path = "ML/Sample_Input/testleaf.jpg"
disease_csv_path = "ML/Sample_Input/disease_sample.csv"
insect_csv_path = "ML/Sample_Input/insect_sample.csv"
yolo_disease_model_path = "ML/Trained_Model/Disease/best.pt"
yolo_insect_model_path = "ML/Trained_Model/Insect/best.pt"
tabnet_disease_model_path = "ML/Trained_Model/Tabnet-Disease/crop_disease.zip"
tabnet_insect_model_path = "ML/Trained_Model/Tabnet-Insect/crop_insect.zip"

# === CHECK FILES EXIST ===
required_files = [
    image_path, disease_csv_path, insect_csv_path,
    yolo_disease_model_path, yolo_insect_model_path,
    tabnet_disease_model_path, tabnet_insect_model_path
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Required file not found: {file}")

# === YOLO INFERENCE ===
model_insect = YOLO(yolo_insect_model_path)
model_disease = YOLO(yolo_disease_model_path)

result_insect = model_insect.predict(source=image_path, verbose=False)[0]
result_disease = model_disease.predict(source=image_path, verbose=False)[0]

insect_present_yolo = result_insect.boxes is not None and len(result_insect.boxes.data) > 0
disease_present_yolo = result_disease.masks is not None and len(result_disease.masks.data) > 0

# === LOAD TABNET MODELS ===
clf_disease = TabNetClassifier()
clf_disease.load_model(tabnet_disease_model_path)

clf_insect = TabNetClassifier()
clf_insect.load_model(tabnet_insect_model_path)

import numpy as np

# === FILE PATHS ===
model_path = "ML/Trained_Model/Tabnet-Disease/crop_disease.zip"
sample_csv_path = "ML/Sample_Input/disease_sample.csv"
df_sample = pd.read_csv(sample_csv_path)
df_sample = df_sample.replace({"Yes": 1, "No": 0})
X_sample = df_sample.fillna(0).astype(np.float32).values
clf = TabNetClassifier()
clf.load_model(model_path)
pred = clf.predict(X_sample)
is_diseased = bool(pred[0])  # True if 1, False if 0

# === INSECT CSV INFERENCE ===
model_path = "ML/Trained_Model/Tabnet-Insect/crop_insect.zip"
sample_csv_path = "ML/Sample_Input/insect_sample.csv"
df_sample = pd.read_csv(sample_csv_path)
df_sample = df_sample.replace({"Yes": 1, "No": 0})
X_sample = df_sample.fillna(0).astype(np.float32).values
clf = TabNetClassifier()
clf.load_model(model_path)
pred = clf.predict(X_sample)
is_insect = bool(pred[0])  # True if 1, False if 0


# === FINAL MULTIMODAL OUTPUT ===
final_output = {
    "Disease_Present_YOLO": disease_present_yolo,
    "Insect_Present_YOLO": insect_present_yolo,
    "Disease_Present_TabNet": is_diseased,
    "Insect_Present_TabNet": is_insect
}

final_decision = {
    "Disease Present": final_output["Disease_Present_YOLO"] or final_output["Disease_Present_TabNet"],
    "Insect Present": final_output["Insect_Present_YOLO"] or final_output["Insect_Present_TabNet"]
}

# === DISPLAY RESULTS ===
print("✅ Final Decision:")
print("Disease Present" if final_decision["Disease Present"] else "Disease Not Present")
print("Insect Present" if final_decision["Insect Present"] else "Insect Not Present")