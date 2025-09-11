import pyautogui
import cv2
import numpy as np
import platform

# Windows specific imports
try:
    import win32gui
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

# Linux specific imports
try:
    import Xlib
    from Xlib import display
    XLIB_AVAILABLE = True
except ImportError:
    XLIB_AVAILABLE = False

class Eye:
    def __init__(self, config):
        self.window_name = config['game_name']
        self.left = None
        self.top = None
        self.width = config['eye']['width']
        self.height = config['eye']['height']
        self.platform = platform.system().lower()
        self.game_name = config['game_name']  # Store for game-specific logic

        # Initialize platform-specific components
        if self.platform == 'linux' and XLIB_AVAILABLE:
            self.display = display.Display()
        else:
            self.display = None
            
        # Grid extraction configuration
        self.grid_config = config.get('grid', {})
        self.grid_rows = self.grid_config.get('rows', 9)
        self.grid_cols = self.grid_config.get('cols', 9)
        self.detection_size = self.grid_config.get('detection_size', 64)
        self.resolution = self.grid_config.get('resolution', 64)
        
        # Calculate cell dimensions based on screen size
        self.cell_width = self.width // self.grid_cols
        self.cell_height = self.height // self.grid_rows

    def _find_window_linux(self, window_name):

        """Find window on Linux using multiple methods"""
        try:
            if self.display:
                return self._find_window_xlib(window_name)
        except FileNotFoundError:
            pass
        return None

    def _find_window_xlib(self, window_name):
        """Find window using Xlib"""

        try:
            root = self.display.screen().root
            window_ids = root.get_full_property(self.display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value

            for window_id in window_ids:
                window = self.display.create_resource_object('window', window_id)
                try:
                    window_title = window.get_full_property(self.display.intern_atom('_NET_WM_NAME'), Xlib.X.AnyPropertyType)
                    if window_name.lower() in window_title.value.decode('utf-8', errors='ignore').lower():
                        geometry = window.get_geometry()
                        # Get absolute position
                        coords = window.translate_coords(root, 0, 0)
                        return { # TODO: temporarily use abs() to adjust the correct screen location
                            'left': abs(coords.x),
                            'top': abs(coords.y),
                            'width': geometry.width,
                            'height': geometry.height
                        }
                except Exception:
                    continue
        except Exception as e:
            print(f"Error finding window with Xlib: {e}")
        return None

    def _find_window_windows(self, window_name):
        """Find window on Windows"""
        if not WINDOWS_AVAILABLE:
            return None

        hwnd = win32gui.FindWindow(None, window_name)
        if not hwnd:
            return None

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return {
            'left': left,
            'top': top,
            'width': right - left,
            'height': bottom - top
        }

    def find_window_cross_platform(self, window_name):
        """Find window across different platforms"""
        if self.platform == 'windows':
            return self._find_window_windows(window_name)
        elif self.platform == 'linux':
            return self._find_window_linux(window_name)
        else:
            print(f"Unsupported platform: {self.platform}")
            return None
    
    def list_all_windows(self):
        """List all visible windows for debugging purposes"""
        if self.platform == 'windows' and WINDOWS_AVAILABLE:
            return self._list_windows_windows()
        elif self.platform == 'linux' and XLIB_AVAILABLE:
            return self._list_windows_linux()
        else:
            return []
    
    def _list_windows_windows(self):
        """List all windows on Windows"""
        windows = []
        def enum_windows_proc(hwnd, lParam):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text:
                    windows.append(window_text)
            return True
        
        try:
            win32gui.EnumWindows(enum_windows_proc, 0)
        except Exception as e:
            print(f"Error listing windows: {e}")
        return windows
    
    def _list_windows_linux(self):
        """List all windows on Linux"""
        windows = []
        try:
            if self.display:
                root = self.display.screen().root
                window_ids = root.get_full_property(self.display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value
                
                for window_id in window_ids:
                    window = self.display.create_resource_object('window', window_id)
                    try:
                        window_title = window.get_full_property(self.display.intern_atom('_NET_WM_NAME'), Xlib.X.AnyPropertyType)
                        if window_title and window_title.value:
                            title = window_title.value.decode('utf-8', errors='ignore')
                            if title:
                                windows.append(title)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error listing windows: {e}")
        return windows

    def get_screenshot_cv(self):
        if not self.window_name:
            print("Window name not set")
            return None

        # Find window based on platform
        window_info = None
        if self.platform == 'windows':
            window_info = self._find_window_windows(self.window_name)
        elif self.platform == 'linux':
            window_info = self._find_window_linux(self.window_name)
        else:
            print(f"Unsupported platform: {self.platform}")
            return None

        if not window_info:
            print(f"Window '{self.window_name}' not found on {self.platform}, please launch the game before starting the run.")
            return None

        left = window_info['left']
        top = window_info['top']
        width = window_info['width']
        height = window_info['height']

        # Take screenshot using pyautogui
        try:
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            img = np.array(screenshot) # to RGB directly, no more need to convert
            print('screenshot mode:', screenshot.mode)
            # Save screenshot for debugging
            # now = time.strftime("%Y-%m-%d-%H-%M-%S")
            # logger.log_img_cv(img, f"{now}.png")

            # Update window position
            self.left = left
            self.top = top
            return img
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None

    def detect_acted_cv(self, last_screenshot_cv, current_screenshot_cv)->float:
        if last_screenshot_cv is None:
            return 1.0
        last_gray = cv2.cvtColor(last_screenshot_cv, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_screenshot_cv, cv2.COLOR_RGB2GRAY)

        diff = cv2.absdiff(last_gray, current_gray)
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        change_ratio = np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255)
        print(f"Change ratio: {change_ratio}")
        # TODO: focus near where Click and Touch operation happens, 
        # should weigh higher for that salient surrounding than others, 
        # e.g., the button area, the text area, the input area, etc.
        # i.e., saliency and heatmap method
        return change_ratio
    
    def extract_grid_cells(self, screen: np.ndarray = None) -> list:
        """
        Extract individual grid cells from the screen as objects
        
        Args:
            screen: Screen image to extract from (if None, captures new screenshot)
            
        Returns:
            List of grid cell objects with images and metadata
        """
        if screen is None:
            screen = self.get_screenshot_cv()
            if screen is None:
                return []
        
        grid_objects = []
        
        try:
            # Extract each grid cell
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # Calculate cell boundaries
                    x1 = col * self.cell_width
                    y1 = row * self.cell_height
                    x2 = min(x1 + self.cell_width, screen.shape[1])
                    y2 = min(y1 + self.cell_height, screen.shape[0])
                    
                    # Extract cell image from screen
                    cell_image = screen[y1:y2, x1:x2]
                    
                    # Resize cell image to detection size for consistent processing
                    if cell_image.size > 0:
                        cell_image_resized = cv2.resize(cell_image, (self.detection_size, self.detection_size))
                    else:
                        cell_image_resized = cell_image
                    
                    # Create grid object
                    grid_obj = {
                        'type': 'grid_cell',
                        'detector_type': 'eye_grid_extractor',
                        'grid_position': [row, col],
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'size': [x2 - x1, y2 - y1],
                        'image': cell_image_resized,  # Use resized image for detection
                        'original_image': cell_image,  # Keep original for reference
                        'confidence': 1.0,
                        'metadata': {
                            'cell_id': f"cell_{row}_{col}",
                            'resolution': self.resolution,
                            'screen_cell_size': [x2 - x1, y2 - y1],  # Actual screen size
                            'detection_size': [self.detection_size, self.detection_size]  # Detection size
                        }
                    }
                    
                    grid_objects.append(grid_obj)
            
            print(f"🎯 Extracted {len(grid_objects)} grid cells from {self.grid_rows}x{self.grid_cols} grid")
            return grid_objects
            
        except Exception as e:
             print(f"❌ Error extracting grid cells: {e}")
             return []
    
    def get_cell_at_position(self, x: int, y: int, screen: np.ndarray = None):
        """
        Get the grid cell at a specific screen position
        
        Args:
            x, y: Screen coordinates
            screen: Screen image (if None, captures new screenshot)
            
        Returns:
            Grid cell object at the position, or None if out of bounds
        """
        # Calculate grid position
        col = x // self.cell_width
        row = y // self.cell_height
        
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            grid_cells = self.extract_grid_cells(screen)
            cell_index = row * self.grid_cols + col
            if cell_index < len(grid_cells):
                return grid_cells[cell_index]
        
        return None
    
    def analyze_cell_content(self, cell_obj: dict) -> dict:
        """
        Analyze the content of a grid cell using basic image processing
        
        Args:
            cell_obj: Grid cell object with image data
            
        Returns:
            Analysis results with detected features
        """
        try:
            cell_image = cell_obj['image']
            
            # Basic image analysis
            gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
            
            # Calculate basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Color analysis
            dominant_color = np.mean(cell_image, axis=(0, 1))
            
            analysis = {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'edge_density': float(edge_density),
                'dominant_color': dominant_color.tolist(),
                'has_content': mean_intensity > 50 and std_intensity > 10,
                'is_uniform': std_intensity < 5
            }
            
            # Update cell object with analysis
            cell_obj['analysis'] = analysis
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing cell content: {e}")
            return {}
    
    def extract_grid_objects(self, screen: np.ndarray = None) -> list:
        """
        Extract grid cell objects from the screen for BottomUpAgent framework
        
        Args:
            screen: Screen image to extract from (if None, captures new screenshot)
            
        Returns:
            List of grid objects in BottomUpAgent format
        """
        try:
            grid_cells = self.extract_grid_cells(screen)
            
            # Convert grid cells to BottomUpAgent object format
            grid_objects = []
            for cell in grid_cells:
                row, col = cell['grid_position']
                grid_obj = {
                    'id': f"grid_{row}_{col}",
                    'type': 'grid_cell',
                    'content': f"Grid[{row},{col}]",
                    'center': cell['center'],
                    'bbox': cell['bbox'],
                    'interactivity': 'clickable',
                    'source': 'grid_slice',
                    'grid_position': [row, col],
                    'cell_image': cell['image'],
                    'metadata': cell.get('analysis', {})
                }
                grid_objects.append(grid_obj)
            
            return grid_objects
            
        except Exception as e:
            print(f"❌ Error in extract_grid_objects: {e}")
            return []

    # ==================== Scene Analysis Module ====================
    
    def get_game_context(self, gym_adapter, observation=None):
        """
        Unified function to get inventory and game state for any game environment.
        
        Args:
            gym_adapter: The gym environment adapter
            observation: Optional observation dict (fallback source)
            
        Returns:
            dict: Contains 'inventory' and 'game_state'
        """
        inventory = {}
        game_state = {}
        
        try:
            # Try to get inventory from Crafter environment (most detailed)
            if hasattr(gym_adapter, 'env') and hasattr(gym_adapter.env, '_player'):
                player_inventory = gym_adapter.env._player.inventory
                # Only include non-zero items
                inventory = {item: count for item, count in player_inventory.items() if count > 0}
            elif hasattr(gym_adapter, 'env') and hasattr(gym_adapter.env, 'unwrapped'):
                if hasattr(gym_adapter.env.unwrapped, '_player'):
                    player_inventory = gym_adapter.env.unwrapped._player.inventory
                    inventory = {item: count for item, count in player_inventory.items() if count > 0}
            elif observation and 'inventory' in observation:
                # Fallback to observation inventory
                inventory = observation['inventory']
        except Exception as e:
            print(f"⚠️ Could not get inventory: {e}")
            if observation and 'inventory' in observation:
                inventory = observation['inventory']
        
        # Get game state info
        try:
            if hasattr(gym_adapter, 'get_info'):
                game_state = gym_adapter.get_info() or {}
            elif observation and 'game_state' in observation:
                game_state = observation['game_state']
        except Exception as e:
            print(f"⚠️ Could not get game state: {e}")
        
        return {
            'inventory': inventory,
            'game_state': game_state
        }
    
    def get_detected_objects_with_logging(self, detector, screen, context_name="Detection"):
        """
        Unified function to get detected objects with consistent logging.
        
        Args:
            detector: The detector instance
            screen: Screen/image data
            context_name: Context name for logging (e.g., "Step", "MCP")
            
        Returns:
            list: Detected objects
        """
        detected_objects = []
        
        try:
            if hasattr(detector, 'get_detected_objects'):
                detected_objects = detector.get_detected_objects(screen)
            elif hasattr(detector, 'extract_objects_crafter_api'):
                detected_objects = detector.extract_objects_crafter_api(screen)
            
            # Consistent logging
            if detected_objects:
                print(f"🔍 [{context_name}] Extracted {len(detected_objects)} objects from {self.game_name} environment")
                
                # Show detection method if available
                if hasattr(detector, 'detection_method'):
                    print(f"📊 [{context_name}] Total objects extracted: {len(detected_objects)} using {detector.detection_method} method")
                else:
                    print(f"📊 [{context_name}] Total objects extracted: {len(detected_objects)}")
                
                # Log object distribution
                object_types = {}
                for obj in detected_objects:
                    obj_type = obj.get('type', 'unknown')
                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
                
                if len(object_types) <= 10:  # Only show if not too many types
                    print(f"📋 [{context_name}] Object distribution: {dict(sorted(object_types.items()))}")
            else:
                print(f"⚠️ [{context_name}] No objects detected")
                
        except Exception as e:
            print(f"❌ [{context_name}] Object detection failed: {e}")
        
        return detected_objects
    
    def generate_and_display_scene_summary(self, brain, detected_objects, inventory, game_state, 
                                         step_info="", context_name="SCENE"):
        """
        Unified function to generate and display scene summary with consistent formatting.
        
        Args:
            brain: Brain instance with _generate_scene_summary method
            detected_objects: List of detected objects
            inventory: Player inventory dict
            game_state: Game state dict
            step_info: Step number or identifier for display
            context_name: Context name for the summary header
            
        Returns:
            str: Generated scene summary
        """
        scene_summary = ""
        
        if len(detected_objects) > 0 and hasattr(brain, '_generate_scene_summary'):
            try:
                # Generate comprehensive scene summary
                scene_summary = brain._generate_scene_summary(detected_objects, inventory, game_state)
                
                # Display with consistent formatting
                header = f"=== 🎯 {context_name}"
                if step_info:
                    header += f" {step_info}"
                header += " SCENE SUMMARY ==="
                
                print(f"\n{header}")
                print(f"{scene_summary}")
                print("=== END SCENE SUMMARY ===\n")
                
            except Exception as e:
                print(f"❌ Error generating scene summary: {e}")
                scene_summary = f"Error: {e}"
        else:
            print(f"⚠️ Cannot generate scene summary: {'No objects detected' if not detected_objects else 'Brain method unavailable'}")
        
        return scene_summary