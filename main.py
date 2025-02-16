import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument(
        '--webcam-resolution',
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO(r'/Users/andrewyuyy/Documents/GitHub/Webcam_Detection/runs/detect/train4/weights/best.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1,
    )

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.names[class_id]}{confidence:0.2f}"
            for bbox, confidence, class_id
            in zip(detections.xyxy, detections.confidence, detections.class_id)
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow('Traffic_sign_detection', frame)

        if (cv2.waitKey(30) == 27):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()