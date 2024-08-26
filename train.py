from ultralytics import YOLO

def train_model():
    # Load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(data="config.yaml", epochs=100)

if __name__ == "__main__":
    train_model()
