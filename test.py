import os
import cv2
from ultralytics import YOLO


def run_weapon_detection():
    # 1. Path to your fine-tuned weights file
    # Replace 'runs/detect/train/weights/best.pt' with your exact weights path if different
    model_path = "runs/detect/train/weights/best.pt"

    # 2. Path to the image/video you want to test
    # Replace with the path to a test image on your computer
    source_path = "dataset/test/images/test_sample.jpg"

    # Check if weights exist before running
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at '{model_path}'. Make sure training completed successfully.")
        return

    # Load the custom-trained YOLO model
    model = YOLO(model_path)

    # Run inference (predict bounding boxes)
    # conf=0.5 means only show predictions with at least 50% confidence
    results = model.predict(source=source_path, conf=0.5, save=True)

    # Process and display result frame
    for result in results:
        # Plot draws bounding boxes and confidence scores directly onto the frame
        annotated_frame = result.plot()

        # Display output in a window (Press any key to close)
        cv2.imshow("Weapon Detection Test", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_weapon_detection()