# === YOLO INFERENCE ===
from ultralytics import YOLO

# Load trained YOLO models
model_insect = YOLO('ML/Trained_Model/Insect/best.pt')
model_disease = YOLO('ML/Trained_Model/Disease/best.pt')

# Predict on the uploaded image
result_insect = model_insect.predict(source='test_leaf.jpg')[0]
result_disease = model_disease.predict(source='test_leaf.jpg')[0]

# Convert to boolean
insect_present_yolo = result_insect.boxes is not None and len(result_insect.boxes.data) > 0
disease_present_yolo = result_disease.masks is not None and len(result_disease.masks.data) > 0

# === LOAD SAVED TABNET MODELS ===
from pytorch_tabnet.tab_model import TabNetClassifier

clf_disease = TabNetClassifier()
clf_disease.load_model("crop_disease.zip")

clf_insect = TabNetClassifier()
clf_insect.load_model("crop_insect.zip")

# === CSV INFERENCE ===
import pandas as pd

# Load CSV input and prepare input arrays
df_disease = pd.read_csv("disease_answers.csv")
X_disease = df_disease.drop(columns=["Label", "leaf_id"], errors="ignore").values
tabnet_disease_pred = clf_disease.predict(X_disease)
disease_present_tabnet = bool(tabnet_disease_pred[0])

df_insect = pd.read_csv("insect_answers.csv")
X_insect = df_insect.drop(columns=["Label", "leaf_id"], errors="ignore").values
tabnet_insect_pred = clf_insect.predict(X_insect)
insect_present_tabnet = bool(tabnet_insect_pred[0])

# === FINAL MULTIMODAL OUTPUT ===
final_output = {
    "Disease_Present_YOLO": disease_present_yolo,
    "Insect_Present_YOLO": insect_present_yolo,
    "Disease_Present_TabNet": disease_present_tabnet,
    "Insect_Present_TabNet": insect_present_tabnet
}

final_decision = {
    "Disease Present": final_output["Disease_Present_YOLO"] or final_output["Disease_Present_TabNet"],
    "Insect Present": final_output["Insect_Present_YOLO"] or final_output["Insect_Present_TabNet"]
}

# === DISPLAY RESULTS ===
print("✅ Final Decision:")
if final_decision["Disease Present"]:
    print("Disease Present")
else:
    print("Disease Not Present")

if final_decision["Insect Present"]:
    print("Insect Present")
else:
    print("Insect Not Present")
