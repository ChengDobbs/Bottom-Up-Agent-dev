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
from skimage.metrics import structural_similarity as ssim

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
        elif self.detector_type == 'template':
            # Initialize template matching detector
            template_config = config['detector']['template']
            self.assets_path = template_config.get('assets_path', '')
            self.grid_size = template_config.get('grid_size', [9, 9])
            self.template_threshold = template_config.get('threshold', 0.8)
            self.templates = self._load_templates()
        elif self.detector_type == 'crafter_api':
            # Initialize Crafter API-based detector
            crafter_config = config['detector']['crafter_api']
            self.grid_size = crafter_config.get('grid_size', [7, 9])  # 7x9 main view
            self.inventory_size = crafter_config.get('inventory_size', [9, 2])  # 9x2 inventory
            self.unit_size = crafter_config.get('unit_size', 64)  # Size of each grid cell
            self.crafter_env = None  # Will be set by the adapter
            self.enable_semantic_view = crafter_config.get('enable_semantic_view', True)
            
            # Detection method configuration
            self.detection_method = crafter_config.get('detection_method', 'hybrid')
            
            # Similarity matching configuration
            similarity_config = crafter_config.get('similarity_matching', {})
            self.similarity_enabled = similarity_config.get('enabled', True)
            self.similarity_threshold = similarity_config.get('similarity_threshold', 0.8)
            self.use_color_analysis = similarity_config.get('use_color_analysis', True)
            self.cache_templates = similarity_config.get('cache_templates', True)
            
            # Direct API configuration
            api_config = crafter_config.get('direct_api', {})
            self.direct_api_enabled = api_config.get('enabled', True)
            self.use_world_state = api_config.get('use_world_state', True)
            self.use_player_state = api_config.get('use_player_state', True)
            self.include_objects = api_config.get('include_objects', True)
            self.fallback_to_similarity = api_config.get('fallback_to_similarity', True)
            
            # Initialize template cache for similarity matching
            self.template_cache = {}
            self.assets_path = 'c:\\Users\\angus\\Documents\\Bottom-Up-Agent-dev\\crafter\\crafter\\assets'
            
            self.eye = Eye(config)
            self.grid_extractor_enabled = True
            print(f"✅ Eye module initialized with grid extraction and detection method: {self.detection_method}")
        
        self.area_threshold = 0.03

        # Initialize CLIP only if needed (not for crafter_api with direct_api enabled)
        self.clip = None
        if self.detector_type == 'crafter_api':
            # For crafter_api, only initialize CLIP if similarity matching is enabled and direct API is disabled
            if self.similarity_enabled and (not self.direct_api_enabled or self.fallback_to_similarity):
                print("🔧 Initializing CLIP for similarity matching fallback...")
                self.clip = CLIP(model_name=config['detector']['clip_model'], use_gpu=True)
            else:
                print("✅ Skipping CLIP initialization - using direct API only")
        else:
            # For other detector types, always initialize CLIP
            self.clip = CLIP(model_name=config['detector']['clip_model'], use_gpu=True)
    
    def start_parallel_crafter_launcher(self, max_steps: int = None) -> bool:
        """
        Start parallel crafter_interactive_launcher for grid extraction.
        Only available when detector_type is 'crafter_api'.
        
        Args:
            max_steps: Maximum steps for the launcher (None to use config default)
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.detector_type != 'crafter_api':
            print("❌ Parallel launcher only available for crafter_api detector type")
            return False
            
        if not hasattr(self, 'eye'):
            print("❌ Eye module not initialized")
            return False
        
        # Use config default if max_steps not provided
        if max_steps is None:
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 1000)
            print(f"📋 Using max_steps from config: {max_steps}")
            
        return self.eye.start_parallel_launcher(max_steps)
    
    def stop_parallel_crafter_launcher(self):
        """
        Stop the parallel crafter_interactive_launcher.
        """
        if hasattr(self, 'eye'):
            # Eye module doesn't have parallel launcher functionality
            print("ℹ️ Eye module doesn't support parallel launcher")
    
    def is_parallel_launcher_running(self) -> bool:
        """
        Check if parallel launcher is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        if hasattr(self, 'eye'):
            # Eye module doesn't have parallel launcher functionality
            return False
        return False
    
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
                'image': cropped,
                'interactivity': 'unknown'
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
        elif self.detector_type == 'crafter_api':
            return self.extract_objects_crafter_api(image, text_prompt)
        else:
            print(f"Unknown detector type: {self.detector_type}")
            return []
    
    def extract_objects_crafter_api(self, image, text_prompt=None):
        """
        Extract objects using improved Crafter detection methods.
        
        Args:
            image: Input image from screen capture
            text_prompt: Optional text prompt for filtering objects
            
        Returns:
            List of detected objects with accurate type identification
        """
        objects = []
        
        # Choose detection method based on configuration
        if self.detection_method == 'similarity_matching':
            objects = self._extract_objects_similarity_matching(image)
        elif self.detection_method == 'direct_api':
            objects = self._extract_objects_direct_api(image)
        elif self.detection_method == 'hybrid':
            objects = self._extract_objects_hybrid(image)
        else:
            # Fallback to original grid extractor method
            objects = self._extract_objects_grid_extractor(image)
        
        print(f"📊 Total objects extracted: {len(objects)} using {self.detection_method} method")
        return objects
    
    def _extract_objects_grid_extractor(self, image):
        """Grid extractor method using Eye module."""
        objects = []
        if hasattr(self, 'eye') and self.grid_extractor_enabled:
            try:
                grid_cells = self.eye.extract_grid_cells(image)
                for cell in grid_cells:
                    grid_pos = cell.get('grid_position', [0, 0])
                    obj = {
                        'id': f"grid_cell_{grid_pos[0]}_{grid_pos[1]}",
                        'type': 'grid_cell',
                        'bbox': cell['bbox'],
                        'center': cell['center'],
                        'image': cell['image'],
                        'grid_position': grid_pos,
                        'cell_size': cell.get('size', [44, 57]),
                        'confidence': 1.0,
                        'source': 'eye_grid_extractor'
                    }
                    objects.append(obj)
            except Exception as e:
                print(f"❌ Error in Eye grid extractor: {e}")
        return objects
    
    def _extract_objects_similarity_matching(self, image):
        """Extract objects using template similarity matching."""
        objects = []
        if not self.similarity_enabled:
            return objects
            
        try:
            # Get grid cells from Eye module
            grid_cells = self.eye.extract_grid_cells(image)
            
            for cell in grid_cells:
                cell_image = cell['image']
                grid_pos = cell.get('grid_position', [0, 0])
                
                # Perform similarity matching
                detected_type, confidence = self._match_cell_to_template(cell_image)
                
                obj = {
                    'id': f"cell_{grid_pos[0]}_{grid_pos[1]}",
                    'type': detected_type,
                    'bbox': cell['bbox'],
                    'center': cell['center'],
                    'image': cell_image,
                    'grid_position': grid_pos,
                    'confidence': confidence,
                    'source': 'similarity_matching'
                }
                objects.append(obj)
                
        except Exception as e:
            print(f"❌ Error in similarity matching: {e}")
            
        return objects
    
    def _extract_objects_direct_api(self, image):
        """Extract objects using direct Crafter API access."""
        objects = []
        # If direct API is not available, fall back to similarity matching
        if not self.direct_api_enabled or self.crafter_env is None:
            if self.fallback_to_similarity:
                return self._extract_objects_similarity_matching(image)
            return objects
            
        try:
            world = self.crafter_env._world
            player = self.crafter_env._player
            
            # Get grid cells for positioning
            grid_cells = self.eye.extract_grid_cells(image)
            
            for cell in grid_cells:
                grid_pos = cell.get('grid_position', [0, 0])  # [row, col] format
                
                # Calculate world position - convert [row, col] to [x, y] coordinates
                offset = np.array([self.grid_size[1] // 2, self.grid_size[0] // 2])  # [cols//2, rows//2]
                world_pos = player.pos + np.array([grid_pos[1], grid_pos[0]]) - offset  # [col, row] -> [x, y]
                
                # Get material and object from API
                if (0 <= world_pos[0] < world.area[0] and 0 <= world_pos[1] < world.area[1]):
                    material, obj = world[world_pos]
                    
                    detected_type = 'empty'
                    # Prioritize objects (creatures/items) over materials (terrain)
                    if obj:
                        detected_type = obj.__class__.__name__.lower()
                        # Handle directional objects like arrows
                        if hasattr(obj, 'facing') and obj.facing is not None:
                            direction_map = {
                                (-1, 0): 'left',
                                (1, 0): 'right', 
                                (0, -1): 'up',
                                (0, 1): 'down'
                            }
                            direction = direction_map.get(tuple(obj.facing), '')
                            if direction:
                                detected_type = f"{detected_type}-{direction}"
                    elif material:
                        detected_type = material
                else:
                    # Handle out of bounds positions
                    detected_type = 'out_of_bounds'
                
                cell_obj = {
                    'id': f"api_cell_{grid_pos[0]}_{grid_pos[1]}",
                    'type': detected_type,
                    'bbox': cell['bbox'],
                    'center': cell['center'],
                    'image': cell['image'],
                    'grid_position': grid_pos,
                    'world_position': world_pos.tolist(),
                    'confidence': 1.0,
                    'source': 'direct_api'
                }
                objects.append(cell_obj)
                    
        except Exception as e:
            print(f"❌ Error in direct API: {e}")
            if self.fallback_to_similarity:
                return self._extract_objects_similarity_matching(image)
                
        return objects
    
    def _extract_objects_hybrid(self, image):
        """Extract objects using hybrid approach (API + similarity matching)."""
        objects = []
        
        # Try direct API first
        api_objects = self._extract_objects_direct_api(image)
        
        # Use similarity matching for unknown/empty cells
        similarity_objects = self._extract_objects_similarity_matching(image)
        
        # Merge results, prioritizing API results
        api_positions = {tuple(obj['grid_position']): obj for obj in api_objects}
        
        for sim_obj in similarity_objects:
            pos = tuple(sim_obj['grid_position'])
            if pos in api_positions:
                api_obj = api_positions[pos]
                # Use API result if it's not empty, otherwise use similarity result
                if api_obj['type'] in ['empty', 'unknown']:
                    if sim_obj['type'] not in ['empty', 'unknown'] and sim_obj['confidence'] > 0.6:
                        api_obj['type'] = sim_obj['type']
                        api_obj['confidence'] = sim_obj['confidence']
                        api_obj['source'] = 'hybrid'
                objects.append(api_obj)
            else:
                objects.append(sim_obj)
                
        return objects
    
    def _match_cell_to_template(self, cell_image):
        """Match a cell image to the best template."""
        if cell_image is None or cell_image.size == 0:
            return 'empty', 0.0
            
        # Load templates if not cached
        if not self.template_cache:
            self._load_crafter_templates()
            
        best_match = 'obj'
        best_score = 0.0
        
        # Resize cell image to standard size
        cell_resized = cv2.resize(cell_image, (64, 64))
        
        # Try template matching
        for template_name, template_img in self.template_cache.items():
            try:
                # Calculate similarity using multiple methods
                score = self._calculate_similarity(cell_resized, template_img)
                
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    best_match = template_name
                    
            except Exception as e:
                continue
                
        # Fallback to color analysis if no good template match
        if best_score < self.similarity_threshold and self.use_color_analysis:
            color_type = self._analyze_cell_color(cell_resized)
            if color_type != 'obj':
                return color_type, 0.7
                
        return best_match, best_score
    
    def _load_crafter_templates(self):
        """Load Crafter asset templates for similarity matching."""
        if not os.path.exists(self.assets_path):
            print(f"❌ Assets path not found: {self.assets_path}")
            return
            
        template_files = {
            'grass': 'grass.png',
            'stone': 'stone.png', 
            'tree': 'tree.png',
            'water': 'water.png',
            'player': 'player.png',
            'coal': 'coal.png',
            'iron': 'iron.png',
            'diamond': 'diamond.png',
            'lava': 'lava.png',
            'sand': 'sand.png',
            'path': 'path.png'
        }
        
        for name, filename in template_files.items():
            filepath = os.path.join(self.assets_path, filename)
            if os.path.exists(filepath):
                template = cv2.imread(filepath)
                if template is not None:
                    template_resized = cv2.resize(template, (64, 64))
                    self.template_cache[name] = template_resized
                    
        print(f"✅ Loaded {len(self.template_cache)} templates for similarity matching")
    
    def _calculate_similarity(self, img1, img2):
        """Calculate similarity between two images using multiple metrics."""
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Structural similarity
        ssim_score = ssim(gray1, gray2)
        
        # Template matching
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_score = np.max(result)
        
        # Histogram comparison
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Weighted combination
        combined_score = (ssim_score * 0.4 + template_score * 0.4 + hist_score * 0.2)
        return max(0.0, combined_score)
    
    def _analyze_cell_color(self, cell_image):
        """Analyze cell color to determine type."""
        if cell_image is None or cell_image.size == 0:
            return 'empty'
            
        # Calculate average color
        avg_color = np.mean(cell_image, axis=(0, 1))
        b, g, r = avg_color
        
        # Calculate brightness
        brightness = np.mean(avg_color)
        
        # Color-based classification
        if brightness < 30:
            return 'empty'
        elif brightness > 200:
            return 'empty'
        elif g > r and g > b and g > 100:  # Green dominant
            return 'grass'
        elif b > r and b > g and b > 80:   # Blue dominant
            return 'water'
        elif r > g and r > b and r > 100:  # Red dominant
            return 'player'
        elif r > 100 and g > 80 and b < 60:  # Brown/yellow
            return 'tree'
        elif brightness < 80:  # Dark
            return 'stone'
        else:
            return 'obj'
    
    def _extract_grid_objects(self, world, player):
        """
        Extract objects from the grid-based local view around the player.
        
        Args:
            world: Crafter world object
            player: Player object
            
        Returns:
            List of grid-based objects
        """
        objects = []
        grid_width, grid_height = self.grid_size
        offset = np.array([grid_width // 2, grid_height // 2])
        
        for x in range(grid_width):
            for y in range(grid_height):
                # Calculate world position
                world_pos = player.pos + np.array([x, y]) - offset
                
                # Check if position is within world bounds
                if not (0 <= world_pos[0] < world.area[0] and 0 <= world_pos[1] < world.area[1]):
                    continue
                    
                # Get material and object at this position
                material, obj = world[world_pos]
                
                # Calculate screen position (grid coordinates)
                screen_x = x * self.unit_size
                screen_y = y * self.unit_size
                
                # Add material as object if it exists
                if material:
                    objects.append({
                        'type': 'material',
                        'name': material,
                        'grid_pos': [x, y],
                        'world_pos': world_pos.tolist(),
                        'bbox': [screen_x, screen_y, screen_x + self.unit_size, screen_y + self.unit_size],
                        'center': [screen_x + self.unit_size // 2, screen_y + self.unit_size // 2],
                        'confidence': 1.0,
                        'source': 'crafter_api_grid'
                    })
                
                # Add object if it exists
                if obj:
                    objects.append({
                        'type': 'object',
                        'name': obj.__class__.__name__.lower(),
                        'texture': getattr(obj, 'texture', 'unknown'),
                        'grid_pos': [x, y],
                        'world_pos': world_pos.tolist(),
                        'bbox': [screen_x, screen_y, screen_x + self.unit_size, screen_y + self.unit_size],
                        'center': [screen_x + self.unit_size // 2, screen_y + self.unit_size // 2],
                        'confidence': 1.0,
                        'source': 'crafter_api_grid'
                    })
        
        return objects
    
    def _extract_inventory_objects(self, inventory):
        """
        Extract objects from the player's inventory.
        
        Args:
            inventory: Player inventory object
            
        Returns:
            List of inventory objects
        """
        objects = []
        inv_width, inv_height = self.inventory_size
        
        for index, (item, amount) in enumerate(inventory.items()):
            if amount < 1:
                continue
                
            # Calculate grid position in inventory
            inv_x = index % inv_width
            inv_y = index // inv_width
            
            if inv_y >= inv_height:
                break  # Inventory full
                
            # Calculate screen position (assuming inventory is below main grid)
            screen_x = inv_x * self.unit_size
            screen_y = (self.grid_size[1] + inv_y) * self.unit_size
            
            objects.append({
                'type': 'inventory_item',
                'name': item,
                'amount': amount,
                'inventory_pos': [inv_x, inv_y],
                'inventory_index': index,
                'bbox': [screen_x, screen_y, screen_x + self.unit_size, screen_y + self.unit_size],
                'center': [screen_x + self.unit_size // 2, screen_y + self.unit_size // 2],
                'confidence': 1.0,
                'source': 'crafter_api_inventory'
            })
        
        return objects
    
    def _extract_semantic_objects(self, world):
        """
        Extract semantic objects from the world (all objects with their types).
        
        Args:
            world: Crafter world object
            
        Returns:
            List of semantic objects
        """
        objects = []
        
        # Get all objects in the world
        for obj in world.objects:
            if obj and hasattr(obj, 'pos'):
                objects.append({
                    'type': 'semantic_object',
                    'name': obj.__class__.__name__.lower(),
                    'texture': getattr(obj, 'texture', 'unknown'),
                    'world_pos': obj.pos.tolist(),
                    'properties': {
                        'removed': getattr(obj, 'removed', False),
                        'health': getattr(obj, 'health', None),
                        'age': getattr(obj, 'age', None)
                    },
                    'confidence': 1.0,
                    'source': 'crafter_api_semantic'
                })
        
        return objects
    
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
                'image': cropped,
                'interactivity': 'unknown'  # 初始化为未知状态
            }
            # print(object_meta)
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