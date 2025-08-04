from BottomUpAgent.Eye import Eye
from BottomUpAgent.LongMemory import LongMemory
from BottomUpAgent.Detector import Detector
from utils.utils import image_grounding, image_grounding_v2, image_grounding_v3

import yaml
import cv2
import matplotlib.pyplot as plt
import argparse


def test_action_grounding(config):
    long_memory = LongMemory(config)
    if not long_memory.is_initialized():
        long_memory.initialize()

    detector = Detector(config)

    img1 = "scripts/images/defend.png"
    img1_cv = cv2.imread(img1)
    img1_cv = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)

    state_feature = detector.encode_image(img1_cv)


    potential_actions = long_memory.get_actions(state_feature)

    coordinates = []

    for action in potential_actions:
        ui = action['ui_image']
        cv2.matchTemplate(img1_cv, ui, cv2.TM_CCOEFF_NORMED)

        h, w = ui.shape[:2]

        result = cv2.matchTemplate(img1_cv, ui, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        threshold = 0.55
        if max_val >= threshold:
            x, y = max_loc
            center_x, center_y = x + w // 2, y + h // 2  # 计算中心点
            coordinates.append((center_x, center_y))
        else:
            print(action['action_name'])
            print(max_val)
            print("not found")


    print(coordinates)


    plt.figure(figsize=(20,20))
    plt.imshow(img1_cv)
    for action in coordinates:
        plt.scatter(action[0], action[1], color='red', s=100)

    plt.show()

def test_image_grounding(config, image_path):
    long_memory = LongMemory(config)

    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    for id, object in long_memory.objects.items():
        grounding_result,score = image_grounding_v3(img_cv, object['image'])
        print(score, object['name'], grounding_result)

def test_objects_grounding(config):
    long_memory = LongMemory(config)
    eye = Eye(config)
    if not long_memory.is_initialized():
        long_memory.initialize()

    enemy = 0

    for i in range(1, 5):
        print(i)

        img_cv = eye.get_screenshot_cv()

        for id, object in long_memory.objects.items():
            grounding_result,score = image_grounding_v3(img_cv, object['image'])
            print(score, object['name'], grounding_result)
            if grounding_result: 
                if object['name'] == 'Enemy' or object['name'] == 'Enemy2':
                    print("enemy found")
                    enemy += 1
                    break

    print(enemy)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, help='path to the config file')
    parser.add_argument('--img_file', required=True, help='path to the image file')
    opt = parser.parse_args()

    with open(opt.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    test_objects_grounding(config)