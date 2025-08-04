import base64
import cv2
import numpy as np
from PIL import Image
import io
from io import BytesIO
import json

def base64_to_cv_gray(base64_img):
    image_data = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def base64_to_cv_rgb(base64_img):
    image_data = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def cv_to_base64(cv_img):
    img = Image.fromarray(cv_img)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img

def save_image(image, path):
    """
    Save a PIL Image to a file.

    :param image: PIL Image object
    :param path: Path to save the image
    """
    image.save(path)

def get_window_cuts(cv_img, coords, window_size=64):
    cuts = []
    half_size = int(window_size // 2)
    for coord in coords:
        left = max(coord[0] - half_size, 0)
        top = max(coord[1] - half_size, 0)
        right = min(coord[0] + half_size, cv_img.shape[1])
        bottom = min(coord[1] + half_size, cv_img.shape[0])
        cut = cv_img[top:bottom, left:right]
        cuts.append(cut)

    return cuts

def action_grounding(img, action):
    ui = action['ui_image']
    cv2.matchTemplate(img, ui, cv2.TM_CCOEFF_NORMED)

    h, w = ui.shape[:2]

    result = cv2.matchTemplate(img, ui, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    threshold = 0.4
    if max_val >= threshold:
        x, y = max_loc
        center_x, center_y = x + w // 2, y + h // 2  # 计算中心点
        return {"name": action['operation'], "coordinates": [[int(center_x), int(center_y)]]}
    else:
        print("No action found : ", action['action_name'])
        print(max_val)
        return None
    
def image_grounding(img1, img2, threshold=0.65):

    h, w = img2.shape[:2]

    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
        center_x, center_y = x + w // 2, y + h // 2  # 计算中心点
        return [int(center_x), int(center_y)], max_val
    else:
        return None, max_val
    
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def image_grounding_v2(src_img, template_img, threshold=0.8):
    gray_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    h, w = template_img.shape[:2]
    best_score = -1
    best_center = None
    angles = [-8, -4, 0, 4, 8]
    scales = [0.75, 1.0]

    for scale in scales:
        scaled = cv2.resize(template_img, (int(w * scale), int(h * scale)))
        

        for angle in angles:
            rotated = rotate_image(scaled, angle)
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            th, tw = gray.shape[:2]

            if gray_src.shape[0] < th or gray_src.shape[1] < tw:
                continue

            result = cv2.matchTemplate(gray_src, gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_center = (max_loc[0] + tw // 2, max_loc[1] + th // 2)

    if best_score >= threshold:
        x, y = best_center
        return [int(x), int(y)], best_score
    else:
        return None, best_score

def image_grounding_v3(src_img, template_img, threshold=0.8):

    h, w = template_img.shape[:2]
    best_score = -1
    best_center = None
    angles = [-8, -4, 0, 4, 8]
    scales = [1.0]

    for scale in scales:
        scaled = cv2.resize(template_img, (int(w * scale), int(h * scale)))
        

        for angle in angles:
            rotated = rotate_image(scaled, angle)
            th, tw = rotated.shape[:2]

            if src_img.shape[0] < th or src_img.shape[1] < tw:
                continue

            result = cv2.matchTemplate(src_img, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_center = (max_loc[0] + tw // 2, max_loc[1] + th // 2)

    if best_score >= threshold:
        x, y = best_center
        return [int(x), int(y)], best_score
    else:
        return None, best_score
    
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)  
    return [(v - min_val) / (max_val - min_val) for v in values]

def operations_to_str(operations):
    result = ''
    for op in operations:
        if op["operate"] == "Click":
            result +=  f"Click({op['object_id']}) \n"
        elif op["operate"] == "RightSingle":
            result +=  f"RightSingle({op['object_id']}) \n"

    return result

