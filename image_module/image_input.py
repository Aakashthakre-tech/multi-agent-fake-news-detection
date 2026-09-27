from PIL import Image


def load_image(image_path):

    image = Image.open(image_path)

    print("Image loaded successfully")
    print("Image size:", image.size)
    print("Image mode:", image.mode)

    return image