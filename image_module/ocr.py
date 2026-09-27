import cv2
import easyocr

reader = easyocr.Reader(['en'])


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def extract_text(image_path):

    source_image = cv2.imread(image_path)

    if source_image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    processed = preprocess_image(image_path)

    source_height, source_width = source_image.shape[:2]
    processed_height, processed_width = processed.shape[:2]

    scale_x = source_width / processed_width
    scale_y = source_height / processed_height

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        mag_ratio=1.5
    )

    results_data = []

    for result in results:

        box = result[0]
        detected_text = result[1]
        confidence = result[2]

        if confidence >= 0.40:

            original_box = [
                [
                    float(point[0]) * scale_x,
                    float(point[1]) * scale_y
                ]
                for point in box
            ]

            results_data.append({
                "text": detected_text,
                "confidence": round(float(confidence), 4),
                "box": original_box
            })

    return results_data