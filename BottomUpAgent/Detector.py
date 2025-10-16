import torch
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from utils.omniparser import Omniparser
from utils.utils import cv_to_base64
import clip
from PIL import Image
from typing import List, Dict
import os
import pathlib
from .Eye import Eye
from sklearn.metrics.pairwise import cosine_similarity

class CLIP:
    def __init__(self, model_name: str = "ViT-B/32", use_gpu: bool = True):

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
    
    def encode_text(self, text_query: str, max_length: int = 77) -> np.ndarray:
        text_query = text_query[:max_length]

        with torch.no_grad():
            text_tokens = clip.tokenize([text_query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features_np = text_features.cpu().numpy()
        return text_features_np
    
    def encode_image(self, img_cv) -> np.ndarray:
        pil_img = Image.fromarray(img_cv.astype('uint8'))
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
        
        features_np = features.cpu().numpy()
        return features_np
    
class Detector:
    def __init__(self, config):
        self.detector_type = config['detector']['type']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.detector_type == 'sam':
            sam_config = config['detector']['sam']
            sam_type = config['detector']['sam']['sam_type']
            sam = sam_model_registry[sam_type](checkpoint=sam_config['sam_weights'])
            sam = sam.to(self.device)
            
            self.sam_predictor = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.92,
                min_mask_region_area=100,
            )
        elif self.detector_type == 'omni':
            omni_config = config['detector']['omni']
            self.omniparser = Omniparser(
                som_model_path=omni_config['som_model_path'],
                caption_model_name=omni_config['caption_model_name'],
                caption_model_path=omni_config['caption_model_path'],
                BOX_TRESHOLD=omni_config['BOX_TRESHOLD'],
                iou_threshold=omni_config['iou_threshold'],
                text_overlap_threshold=omni_config['text_overlap_threshold'],
            )
        
        self.area_threshold = 0.03

        self.clip = CLIP(model_name=config['detector']['clip_model'], use_gpu=True)
    
    def encode_image(self, img_cv):
        return self.clip.encode_image(img_cv)

    def encode_text(self, text_query: str):
        return self.clip.encode_text(text_query)
    
    # TODO: merge sam and omni
    def extract_objects_sam(self, image: np.ndarray) -> List[Dict]:
        height, width = image.shape[:2]
        image_area = height * width

        masks = self.sam_predictor.generate(image)
        objects = []

        for mask in masks:
            bbox = mask['bbox']
            x, y, w, h = bbox
            x0, y0 = int(x), int(y)
            x1, y1 = int(x + w), int(y + h)
            area = w * h
            rel_area = area / image_area

            if rel_area > self.area_threshold or w <= 5 or h <= 5:
                continue

            cropped = image[y0:y1, x0:x1]
            resized = cv2.resize(cropped, (32, 32))

            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            if np.std(gray) < 10: 
                continue

            hash_val = self._average_hash(resized)
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2

            if center_y < 25:
                continue

            is_duplicate = False
            # TODO: Check the O(n^2) efficiency
            for prev_object in objects:
                px, py = prev_object['center']
                if abs(px - center_x) <= 6 and abs(py - center_y) <= 6:
                    is_duplicate = True
                    break

                hash_dist = self._hamming_distance(prev_object['hash'], hash_val)
                if hash_dist <= 8:  
                    is_duplicate = True
                    break

            if is_duplicate:
                continue
            
            object_meta = {
                'id': None,
                'content': '', # SAM doesn't provide content, set to empty string
                'bbox': [x0, y0, w, h],
                'area': area,
                'hash': hash_val,
                'center': (center_x, center_y),
                'image': cropped
            }
            # print(object_meta)
            objects.append(object_meta)

        return objects
    
    def get_detected_objects(self, image: np.ndarray, text_prompt=None) -> List[Dict]:
        """Get detected objects from image - unified interface for MCP mode"""
        if self.detector_type == 'sam':
            return self.extract_objects_sam(image)
        elif self.detector_type == 'omni':
            return self.extract_objects_omni(image)
        else:
            print(f"Unknown detector type: {self.detector_type}")
            return []
   
    def _calculate_similarity(self, img1, img2):
        """Calculate similarity between two images using simple and effective metrics."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Simple grayscale difference similarity (similar to Eye.py approach)
        diff = cv2.absdiff(gray1, gray2)
        diff_score = 1.0 - (np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255))
        
        # Template matching
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_score = np.max(result)
        
        # Histogram comparison
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Weighted combination (adjusted weights since we removed SSIM)
        combined_score = (diff_score * 0.4 + template_score * 0.4 + hist_score * 0.2)
        return max(0.0, combined_score)
   
    def get_detected_objects_with_context(self, image: np.ndarray, historical_objects: List[Dict] = None) -> List[Dict]:
        """Get detected objects including both new and historical objects for comprehensive MCP context"""
        # Get newly detected objects
        new_objects = self.get_detected_objects(image)
        
        if not historical_objects:
            return new_objects
        
        # CRITICAL FIX: Use objects_rematch to assign IDs to new objects based on historical objects
        self.objects_rematch(new_objects, historical_objects)
        print(f"Applied object ID matching: {len(new_objects)} new objects matched against {len(historical_objects)} historical objects")
        
        # Create a comprehensive object list that includes both new and relevant historical objects
        comprehensive_objects = []
        
        # Add all new objects (now with proper IDs from matching)
        for obj in new_objects:
            comprehensive_objects.append(obj)
        
        # Add ALL historical objects to provide comprehensive context for MCP
        # Even if they match with new objects, they provide valuable historical context
        for hist_obj in historical_objects:
            hist_center = hist_obj.get('center', (0, 0))
            matched_new_obj = None
            
            # Check if this historical object matches any new object
            for new_obj in new_objects:
                new_center = new_obj.get('center', (0, 0))
                # Use position-based matching
                if (abs(hist_center[0] - new_center[0]) <= 10 and 
                    abs(hist_center[1] - new_center[1]) <= 10):
                    matched_new_obj = new_obj
                    break
                
                # Use hash-based matching if available
                if (hist_obj.get('hash') and new_obj.get('hash') and 
                    self._hamming_distance(hist_obj['hash'], new_obj['hash']) <= 8):
                    matched_new_obj = new_obj
                    break
            
            # Always add historical objects, but mark their relationship to new objects
            hist_obj_copy = hist_obj.copy()
            if matched_new_obj:
                hist_obj_copy['source'] = 'historical_matched'
                hist_obj_copy['matched_new_id'] = matched_new_obj.get('id')
                # Log successful ID assignment
                if matched_new_obj.get('id'):
                    print(f"Historical object {hist_obj.get('id')} matched with new object ID: {matched_new_obj.get('id')}")
            else:
                hist_obj_copy['source'] = 'historical_unique'
            comprehensive_objects.append(hist_obj_copy)
        
        # Count objects with valid IDs for debugging
        objects_with_ids = sum(1 for obj in comprehensive_objects if obj.get('id') is not None)
        print(f"Comprehensive objects: {len(new_objects)} new + {len(comprehensive_objects) - len(new_objects)} historical = {len(comprehensive_objects)} total, {objects_with_ids} with valid IDs")
        return comprehensive_objects

    def extract_objects_omni(self, image: np.ndarray, get_parsed_img = False) -> List[Dict]:
        height, width = image.shape[:2]
        image_area = height * width
        base64_encoded_img = cv_to_base64(image)
        if get_parsed_img: # avoid returning a tuple
            labeled_img, coods_xywh_list, contents_list = self.omniparser.parse(base64_encoded_img)
        else:
            _, coods_xywh_list, contents_list = self.omniparser.parse(base64_encoded_img)

        objects = []

        box_cnt = 0
        for box_cnt in range(len(coods_xywh_list)):
            bbox = coods_xywh_list[str(box_cnt)]
            content_text = contents_list[box_cnt].get('content','').strip()
            obj_type = contents_list[box_cnt].get('type','')
            x, y, w, h = bbox
            x0, y0 = int(x), int(y)
            w, h = int(w), int(h)
            x1, y1 = x0 + w, y0 + h
            area = w * h
            rel_area = area / image_area

            if rel_area > self.area_threshold or w <= 5 or h <= 5:
                continue

            cropped = image[y0:y1, x0:x1]
            resized = cv2.resize(cropped, (32, 32))

            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            if np.std(gray) < 10: 
                continue

            hash_val = self._average_hash(resized)
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2

            if center_y < 20:
                continue

            is_duplicate = False
            for prev_object in objects:
                px, py = prev_object['center']
                if abs(px - center_x) <= 6 and abs(py - center_y) <= 6:
                    is_duplicate = True
                    break

                hash_dist = self._hamming_distance(prev_object['hash'], hash_val)
                if hash_dist <= 5:  
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            object_meta = {
                'id': None,
                'content':  content_text, 
                'bbox': [x0, y0, w, h],
                'area': area,
                'hash': hash_val,
                'type': obj_type,
                'center': (center_x, center_y),
                'image': cropped
            }
            objects.append(object_meta)
        
        if get_parsed_img:
            return objects, labeled_img
        else:
            return objects

    def _average_hash(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (8, 8))
        avg = resized.mean()
        return ''.join(['1' if pixel > avg else '0' for row in resized for pixel in row])

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        return sum(ch1 != ch2 for ch1, ch2 in zip(hash1, hash2))

    def objects_rematch(self, objects: List[Dict], existed_objects: List[Dict], area_tol=0.1, hash_threshold=15) -> List[Dict]:
        for object in objects:
            for existed_object in existed_objects:
                if abs(object['area'] - existed_object['area']) / existed_object['area'] > area_tol:
                    continue
                if self._hamming_distance(object['hash'], existed_object['hash']) <= hash_threshold:
                    object['id'] = existed_object['id']
                    break
                # TODO: semantic matching

    def update_objects(self, img, existed_objects):
        if self.detector_type == 'sam':
            objects = self.extract_objects_sam(img)
        elif self.detector_type == 'omni':
            objects = self.extract_objects_omni(img)
        self.objects_rematch(objects, existed_objects)
    
        return objects