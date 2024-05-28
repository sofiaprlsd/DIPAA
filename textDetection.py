import requests
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import pytesseract
import numpy as np
import cv2

def download_image(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status()  # check if the request was successful
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as err:
        print(err)
        return None

def preprocess_image(img):
    try:
        # Convert image to grayscale
        img = img.convert('L')
        
        img = ImageEnhance.Contrast(img).enhance(2)

        img = img.filter(ImageFilter.SHARPEN)

        # Convert to numpy array for processing with OpenCV
        img_np = np.array(img)

        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        img_np = cv2.medianBlur(img_np, 3)
        
        # Convert back to PIL Image
        img = Image.fromarray(img_np)

        return img
    except Exception as err:
        print(err)
        return img  # return original image if process fails

def extract_text_from_image(img):
    try:
        # Perform OCR using Tesseract
        return pytesseract.image_to_string(img)  # return text
    except Exception as err:
        print(err)
        return ""

def extract_text_from_url(img_url):
    img = download_image(img_url)
    if img is None:
        return "Failed to download and process the image"
    img = preprocess_image(img)
    return extract_text_from_image(img)

def extract_text_from_file(file_path):
    try:
        img = Image.open(file_path)
        img = preprocess_image(img)
        return extract_text_from_image(img)
    except Exception as err:
        print(err)
        return "Failed to process the image file"

def main():
    choice = input("Press 1 to process an image from a URL or 2 to process a local image file: ").strip().lower()
    if choice == '1':
        img_url = input("Enter image URL: ").strip()
        print(extract_text_from_url(img_url))
    elif choice == '2':
        file_path = input("Enter the local image file path: ").strip()
        print(extract_text_from_file(file_path))
    else:
        print("Invalid choice. Please enter 'url' or 'file'.")

if __name__ == "__main__":
    main()
