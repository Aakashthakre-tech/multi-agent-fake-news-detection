def get_box_info(box):

    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]

    left = min(x_values)
    right = max(x_values)
    top = min(y_values)
    bottom = max(y_values)

    center_y = (top + bottom) / 2
    height = bottom - top

    return left, right, top, bottom, center_y, height


def group_ocr_phrases(ocr_results):

    if not ocr_results:
        return []

    items = []

    for result in ocr_results:

        left, right, top, bottom, center_y, height = \
            get_box_info(result["box"])

        items.append({
            "text": result["text"],
            "confidence": result["confidence"],
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "center_y": center_y,
            "height": height,
            "width": right - left,
            "box": result["box"]
        })

    lines = []

    for item in sorted(
        items,
        key=lambda value: (value["center_y"], value["left"])
    ):

        best_line = None
        best_distance = float("inf")

        for line in lines:

            line_top = min(word["top"] for word in line)
            line_bottom = max(word["bottom"] for word in line)
            line_left = min(word["left"] for word in line)
            line_right = max(word["right"] for word in line)
            line_height = line_bottom - line_top

            overlap = min(item["bottom"], line_bottom) - max(
                item["top"],
                line_top
            )
            minimum_height = max(
                1.0,
                min(item["height"], line_height)
            )

            if overlap < minimum_height * 0.35:
                continue

            if item["right"] <= line_left:
                horizontal_gap = line_left - item["right"]
            elif item["left"] >= line_right:
                horizontal_gap = item["left"] - line_right
            else:
                horizontal_gap = 0

            average_height = sum(
                word["height"] for word in line
            ) / len(line)

            if horizontal_gap > average_height * 2.5:
                continue

            average_center_y = sum(
                word["center_y"] for word in line
            ) / len(line)
            distance = abs(item["center_y"] - average_center_y)

            if distance < best_distance:
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append([item])
        else:
            best_line.append(item)

    lines.sort(
        key=lambda line: min(word["top"] for word in line)
    )

    phrases = []

    for line in lines:

        line.sort(key=lambda value: value["left"])

        words = []
        confidences = []
        boxes = []

        for item in line:

            words.append(item["text"])
            confidences.append(item["confidence"])
            boxes.append(item["box"])

        all_x = []
        all_y = []

        for box in boxes:
            for point in box:
                all_x.append(point[0])
                all_y.append(point[1])

        combined_box = [
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)]
        ]

        phrases.append({
            "text": " ".join(words),
            "confidence": round(
                sum(confidences) / len(confidences),
                4
            ),
            "box": combined_box
        })

    return phrases