import cv2

_READER = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(['en'])
    return _READER


class _LazyReaderProxy:
    def __getattr__(self, name):
        return getattr(get_reader(), name)


reader = _LazyReaderProxy()


def preprocess_image(image_path, max_upscale_dimension=1280):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = image.shape[:2]
    # If the image is already high resolution (e.g. 640x360 or greater), a 3x upscale
    # creates a 9x pixel explosion (1920x1080) that slows down EasyOCR CRAFT detector to 36+ sec/frame.
    # Adaptive scaling: only upscale tiny images where min dimension < 300px.
    if max(h, w) < 400:
        scale = min(3.0, max_upscale_dimension / max(h, w))
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    elif max(h, w) > 1920:
        scale = 1920 / max(h, w)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return image, gray


def extract_text(image_path, fast_mode=False):

    source_image = cv2.imread(image_path)

    if source_image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    processed_color, processed = preprocess_image(image_path)

    source_height, source_width = source_image.shape[:2]
    processed_height, processed_width = processed.shape[:2]

    scale_x = source_width / processed_width
    scale_y = source_height / processed_height

    mag_ratio = 1.0 if (fast_mode or max(source_height, source_width) >= 640) else 1.5

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        mag_ratio=mag_ratio
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