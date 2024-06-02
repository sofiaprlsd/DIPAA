import cv2
import numpy as np
import requests
from scipy.ndimage import label
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO
import pytesseract
from scipy.ndimage import interpolation as inter


def download_image(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status() # check if the request was successful
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as err:
        print(err)
        return None


def preprocess_image1(img):
    try:

        img_np = np.array(img)

        # Calculate rotation angle and rotate image
        angle, corrected = correct_skew(img_np)
        img = Image.fromarray(corrected)

        img_gray = img.convert('L')

        # binary thresholding to make more distinguishable from background
        binary_img = img_gray.point(lambda p: 255 if p >= 127 else 0)

        # Invert colors (works better with white background and black text)
        binary_img = ImageOps.invert(binary_img)

        img = ImageEnhance.Contrast(binary_img).enhance(6)
        img.show()
        return img
    except Exception as err:
        print(err)
        return img  # return original image if process fails

def preprocess_image2(img):
    img = img.convert('L')

    # adding just the perfect amount of both contrast and Sharpness
    img = ImageEnhance.Contrast(img).enhance(9)
    img = ImageEnhance.Sharpness(img).enhance(4)

    img_array = np.array(img)
    inverted = 255 - img_array
    flat = inverted.flatten()
    text_color = np.bincount(flat).argmax()  # is going to be 0 in pretty much all cases (black)

    # use binary thresholding to make the text more distinguishable
    img = img.point(lambda p: 255 if p > text_color else 0)
    img.show()
    return img


def preprocess_noisy(img):
    return img
    img = img.convert('L')

    # binary thresholding
    img = img.point(lambda p: 255 if p > 128 else 0)

    inverted = ImageOps.invert(img)
    img_array = np.array(inverted)

    components, num_components = label(img_array)
    component_sizes = np.bincount(components.ravel())

    # threshold for noise removal
    min_size = 25

    filtered_components = np.zeros_like(img_array)

    # discard components with size smaller than min_Size
    for i in range(1, num_components + 1):
        if component_sizes[i] >= min_size:
            # components == i creates boolean mask
            filtered_components[components == i] = 255

    img = np.invert(filtered_components)

    img = Image.fromarray(img)
    img.show()
    return img


def preprocess_image(img):
    # choose one enhancement function from below
    # return preprocess_image1(img)
    # return  preprocess_image2(img)
    return preprocess_noisy(img)


# function below is not ours. It has not been written by any member of our team
# function taken from Stackoverflow https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def extract_text_from_image(img):
    try:
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(img, config=custom_config)  # return text
    except Exception as err:
        print(err)
        return ""

def extract_text_from_url(img_path):
    try:
        img = Image.open(img_path)
        img = preprocess_image(img)
        return extract_text_from_image(img)
    except Exception as err:
        print(err)
        return "Failed to process the image file"

def main():
    img_url = input("Enter image path: ")
    print(extract_text_from_url(img_url))

if __name__ == "__main__":
    main()