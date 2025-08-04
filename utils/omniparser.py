from utils.omni import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
from typing import Dict

# save debug image
import time
import os
class Omniparser(object):
    def __init__(
        self, 
        som_model_path,
        caption_model_name,
        caption_model_path,
        BOX_TRESHOLD,
        iou_threshold,
        text_overlap_threshold
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.som_model = get_yolo_model(model_path=som_model_path).to(device)
        self.caption_model_processor = get_caption_model_processor(model_name=caption_model_name, model_name_or_path=caption_model_path, device=device)
        print('Omniparser initialized!!!')
        print(self.som_model.device, type(self.som_model))
        self.BOX_TRESHOLD = BOX_TRESHOLD
        self.iou_threshold = iou_threshold
        self.text_overlap_threshold = text_overlap_threshold

    def parse(self, image_base64: str, **kwargs):
        """
        Parse image with optional text preservation settings
        
        Args:
            image_base64: Base64 encoded image string
            preserve_text_priority: If True, prioritize preserving OCR text boxes over icon boxes
            text_overlap_threshold: Threshold for determining if text should be preserved (lower = more preservation)
            **kwargs: Additional arguments passed to get_som_labeled_img
        """
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (ocr_text, ocr_bbox), _ = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', # must use with int_box_area
            goal_filtering=None,
            text_threshold=0.9,
            use_paddleocr=True
        )
        
        # Merge default parameters with kwargs
        default_params = {
            'BOX_TRESHOLD': self.BOX_TRESHOLD,
            'output_coord_in_ratio': False, 
            'ocr_bbox': ocr_bbox, 
            'draw_bbox_config': draw_bbox_config, # for display labeled image
            'caption_model_processor': self.caption_model_processor,
            'ocr_text': ocr_text,
            'use_local_semantics': True,
            'iou_threshold': self.iou_threshold,
            'scale_img': False,
            'batch_size': 128,
            'preserve_text_priority': True,
            'text_overlap_threshold': self.text_overlap_threshold
        }
        default_params.update(kwargs)
        
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, 
            self.som_model, 
            **default_params
        )
        
        os.makedirs('debug', exist_ok=True)
        result_img = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        # save to debug folder
        now = time.strftime("%Y-%m-%d-%H-%M-%S")
        result_img.save(f'debug/debug_{now}.png')

        return dino_labled_img, label_coordinates, parsed_content_list