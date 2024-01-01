import cv2
import supervision as sv
import inference

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.9]
    print(detections)
    cv2.imshow(
        "Prediction",
        annotator.annotate(
            scene=image,
            detections=detections,
            labels=labels
        )
    )
    cv2.waitKey(1)

inference.Stream(
    source="webcam",  # or rtsp stream or camera id
    model="currency-detection-u7uav/1",  # from Universe
    output_channel_order="BGR",
    use_main_thread=True,  # for OpenCV display
    on_prediction=on_prediction,
)
