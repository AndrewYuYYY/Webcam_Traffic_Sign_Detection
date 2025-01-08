import torch
from ultralytics import YOLO
import argparse

data_path = r'/Users/andrewyuyy/Documents/GitHub/Webcam_Detection/dataset_split/dataset.yaml'

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO model on custom dataset")
    parser.add_argument('--model', type=str, default='yolov8l.pt',
                        help="Pretrained YOLO model (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training and validation")
    parser.add_argument('--device', type=str, default='mps', help="Device to train on: 'cpu', 'cuda', or 'mps'")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Verify the device
    device = torch.device(args.device if torch.backends.mps.is_available() or args.device == 'cpu' else 'cuda')
    print(f"Using device: {device}")

    # Initialize YOLO model
    model = YOLO(args.model)

    # Train the model
    print("Starting training...")
    model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=device
    )
    print("Training completed!")


if __name__ == "__main__":
    main()