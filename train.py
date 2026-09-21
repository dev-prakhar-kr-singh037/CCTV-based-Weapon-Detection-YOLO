from ultralytics import YOLO

if __name__ == "__main__":
    # Load base model
    model = YOLO("yolov8n.pt")

    # YOLO reads data.yaml -> data.yaml points to your external folder
    model.train(
        data=r"C:\Users\prera\OneDrive\Desktop\Combined ML Dataset\dataset_combined\data.yaml",
        epochs=50,
        imgsz=640
    )