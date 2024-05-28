import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps  
from io import BytesIO
import pytesseract
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def download_image(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status()  # check if the request was successful
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as err:
        print(err)
        return None

def preprocess_image(img, local_file=False):
    try:
        # Convert image to grayscale
        img = img.convert('RGB')
        
        # Increase contrast
        img = ImageEnhance.Contrast(img).enhance(2)

        img = img.filter(ImageFilter.SHARPEN)
        # img =  ImageOps.equalize(img, mask = None)

        # Convert to numpy array for processing with OpenCV
        img_np = np.array(img)

        # Resize for better result quality - assure minimum font height to 64 px
        image_data = pytesseract.image_to_data(img_np, output_type=pytesseract.Output.DICT)
        min_h = 64
        for i, word in enumerate(image_data['text']):
            if word != "":
                h = image_data['height'][i]
                if h < min_h:
                    min_h = h  
        resize_factor = 64 // min_h
        img_np = cv2.resize(img_np, (img_np.shape[1] * resize_factor, img_np.shape[0] * resize_factor), interpolation = cv2.INTER_LINEAR) 

        # Threshold every channel
        _, img_np[0] = cv2.threshold(img_np[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, img_np[1] = cv2.threshold(img_np[1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, img_np[2] = cv2.threshold(img_np[2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert to grayscale image
        # h = len(img_np)
        # w = len(img_np[0])
        # img_tmp = np.zeros((h, w))
        # for i in range(h):
        #     for j in range(w):
        #         img_tmp[i, j] = np.average([img_np[i, j, 0], img_np[i, j, 1], img_np[i, j, 2]])
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # img_np = img_tmp.astype('uint8')
        # Image.fromarray(img_np).show()

        # Remove noise
        img_np = cv2.medianBlur(img_np, 3)

        # contours, hierarchy = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # im2 = img_np.copy()
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
            
        #     # Drawing a rectangle on copied image
        #     cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow("window", im2)
        # cv2.waitKey(0)

        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Enhancing edges
        kernel = np.ones((1,1), np.uint8)
        img_np = cv2.dilate(img_np, kernel, iterations=1)

        if local_file == True:
            # Calculate rotation angle
            angle = calculate_rotation_angle(img_np)

            # Rotate image to correct the orientation
            (h, w) = img_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Convert back to PIL Image
        img = Image.fromarray(img_np)
        img.show()

        return img
    except Exception as err:
        print(err)
        return img  # return original image if process fails

def calculate_rotation_angle(img_np):
    try:
        edges = cv2.Canny(img_np, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            angles = []
            for _, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)

            # Calculate median angle
            median_angle = np.median(angles)
            return median_angle
        return 0
    except Exception as err:
        print(err)
        return 0

def extract_text_from_image(img):
    try:
        # Perform OCR using Tesseract
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(img, config=custom_config)  # return text
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
        img = preprocess_image(img, True)
        return extract_text_from_image(img)
    except Exception as err:
        print(err)
        return "Failed to process the image file"

def main():
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # pytesseract.pytesseract.tesseract_cmd = r'D:\Aplikacje\Tesseract\tesseract.exe'

    if len(sys.argv) > 1:
        try:
            f = open("data.txt", "r")
            data = []

            for _ in range(int(sys.argv[1])+1):
                data = f.readline().split(";")
            img_url = data[1].strip()
            print(extract_text_from_url(img_url))

            # for _ in range(14):
            #     data = f.readline().split(";")
            #     img_url = data[1].strip()
            #     print(extract_text_from_url(img_url))

        except:
            print("wrong parameter")
    else:
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
