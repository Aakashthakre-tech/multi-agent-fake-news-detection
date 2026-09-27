from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


def calculate_region_text_similarity(
    image_path,
    box,
    text,
    padding=1.5
):

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    # Get OCR bounding-box coordinates
    x_values = [float(point[0]) for point in box]
    y_values = [float(point[1]) for point in box]

    x1 = min(x_values)
    x2 = max(x_values)
    y1 = min(y_values)
    y2 = max(y_values)

    # Check original bounding box
    if x2 <= x1 or y2 <= y1:
        print(f"Skipping invalid box for: {text}")
        return 0.0

    # Calculate region dimensions
    box_width = x2 - x1
    box_height = y2 - y1

    # Add surrounding visual context
    x1 = x1 - (box_width * padding)
    y1 = y1 - (box_height * padding)

    x2 = x2 + (box_width * padding)
    y2 = y2 + (box_height * padding)

    # Keep coordinates inside image
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    # Final safety check
    if x2 <= x1 or y2 <= y1:
        print(f"Skipping invalid crop for: {text}")
        return 0.0

    # Crop region
    cropped_image = image.crop(
        (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )
    )

    # Prepare CLIP input
    inputs = processor(
        text=[text],
        images=cropped_image,
        return_tensors="pt",
        padding=True
    )

    # Calculate embeddings
    with torch.no_grad():

        outputs = model(**inputs)

        image_embedding = outputs.image_embeds
        text_embedding = outputs.text_embeds

        # Normalize image embedding
        image_embedding = (
            image_embedding /
            image_embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

        # Normalize text embedding
        text_embedding = (
            text_embedding /
            text_embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

        # Cosine similarity
        similarity = (
            image_embedding @ text_embedding.T
        ).item()

    return round(float(similarity), 4)