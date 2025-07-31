from ultralytics import YOLO

disease_yaml_path = "ML/CVAT_Annotations/Disease_Annotated_YOLO_Segmentation/data.yaml"


# Load YOLOv8 segmentation model
disease_model = YOLO('yolov8s-seg.pt')

# Train with augmentation and validation
disease_model.train(
    data=disease_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name='disease-segmentation',
    augment=True,
    degrees=10,       # random rotation
    scale=0.5,        # scale up/down
    shear=5,          # shearing
    flipud=0.2,       # vertical flip
    fliplr=0.5,       # horizontal flip
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.2
)


insect_yaml_path = "ML/CVAT_Annotations/Insect_Annotated_YOLO_Detection/data.yaml"

# Load YOLOv8 detection model
insect_model = YOLO('yolov8s.pt')

# Train with augmentation
insect_model.train(
    data=insect_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name='insect-detection',
    augment=True,
    degrees=15,
    translate=0.1,
    scale=0.7,
    shear=2,
    flipud=0.2,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.2
)


# Load the trained model (make sure to load the best weights)
disease_model = YOLO('ML/Trained_Model/Disease/best.pt')

# Validate using the test set (defined in data.yaml as val path)
disease_metrics = disease_model.val(data=disease_yaml_path)


# Load the trained model (best weights)
insect_model = YOLO('ML/Trained_Model/Insect/best.pt')

# Validate using the test set (defined in data.yaml as val path)
insect_metrics = insect_model.val(data=insect_yaml_path)


def print_metrics(results, model_name):
    print(f"\n🔍 Evaluation Results for {model_name}")
    print("-" * 40)

    # Check if predictions exist
    if results.box.p is None or len(results.box.p) == 0:
        print("⚠️  No predictions were made on the test set.")
        print("🔎 Check your test images and annotations.")
    else:
        precision = float(results.box.p.mean())
        recall = float(results.box.r.mean())
        map50 = float(results.box.map50.mean())
        map5095 = float(results.box.map.mean())

        print(f"Precision:        {precision:.3f}")
        print(f"Recall:           {recall:.3f}")
        print(f"mAP@0.5:          {map50:.3f}")
        print(f"mAP@0.5:0.95:     {map5095:.3f}")
    print("-" * 40)

# Print results
print_metrics(disease_metrics, "Disease Segmentation")
print_metrics(insect_metrics, "Insect Detection")
