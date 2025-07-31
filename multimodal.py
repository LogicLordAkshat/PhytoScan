# === IMPORTS ===
from ultralytics import YOLO
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import os

# === FILE PATHS ===
image_path = "test_leaf.jpg"
disease_csv_path = "disease_answers.csv"
insect_csv_path = "insect_answers.csv"
yolo_disease_model_path = "runs/segment/disease-segmentation/weights/best.pt"
yolo_insect_model_path = "runs/detect/insect-detection/weights/best.pt"
tabnet_disease_model_path = "crop_disease.zip"
tabnet_insect_model_path = "crop_insect.zip"

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

# === CSV INFERENCE ===
df_disease = pd.read_csv(disease_csv_path)
X_disease = df_disease.drop(columns=["Label", "leaf_id"], errors="ignore").values
tabnet_disease_pred = clf_disease.predict(X_disease)
disease_present_tabnet = bool(tabnet_disease_pred[0])

df_insect = pd.read_csv(insect_csv_path)
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
print("Disease Present" if final_decision["Disease Present"] else "Disease Not Present")
print("Insect Present" if final_decision["Insect Present"] else "Insect Not Present")
