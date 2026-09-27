from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def analyze_image(image_path):

    results = model(image_path, verbose=False)

    objects = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            objects.append({
                "object": object_name,
                "confidence": round(confidence, 4)
            })

    return objects