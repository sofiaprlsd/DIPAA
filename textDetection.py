import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO
import pytesseract

def download_image(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status() # check if the request was successful
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as err:
        print(err)
        return None

def preproccess_imgage(img):
    try:
        img = img.convert('L')

        img = ImageEnhance.Contrast(img).enhance(3)

        img = ImageEnhance.Contrast(img).enhance(6)

        return img
    except Exception as err:
        print(err)
        return img # return original image if process fails

def extract_text_from_image(img):
    try:
        # Perform OCR using Tesseract
        return pytesseract.image_to_string(img) # return text
    except Exception as err:
        print(err)
        return ""

def extract_text_from_url(img_url):
    img = download_image(img_url)
    if img is None:
        return "Failed to download and process the image"
    img = preproccess_imgage(img)
    return extract_text_from_image(img)


def main():
    img_url = input("Enter image URL: ")
    print(extract_text_from_url(img_url))

if __name__ == "__main__":
    main()
