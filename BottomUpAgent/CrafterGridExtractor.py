#!/usr/bin/env python3
"""
Crafter Grid Extractor Module
This module provides functionality to run crafter_interactive_launcher in parallel
and extract grid cell images as objects for the BottomUpAgent framework.
"""

import threading
import subprocess
import time
import numpy as np
import cv2
import pygame
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys
import os
import yaml
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    WINDOWS_CAPTURE_AVAILABLE = True
except ImportError:
    WINDOWS_CAPTURE_AVAILABLE = False
    print("⚠️ Windows screen capture not available. Install pywin32 for full functionality.")

class CrafterGridExtractor:
    """
    Extracts grid cell images from Crafter interactive launcher
    and provides them as objects for BottomUpAgent detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gui_config = config.get('gui', {})
        
        # Parse resolution - handle both string presets and numeric values
        resolution_raw = self.gui_config.get('resolution', 'low')
        self.resolution = self._parse_resolution(resolution_raw)
        
        self.grid_size = config.get('detector', {}).get('crafter_api', {}).get('grid_size', [7, 9])
        self.unit_size = config.get('detector', {}).get('crafter_api', {}).get('unit_size', 64)
        
        # Grid extraction parameters
        self.grid_rows, self.grid_cols = self.grid_size
        # Cell dimensions on screen - calculated from actual screen resolution
        # These are the actual pixel dimensions of each cell on the screen
        self.cell_width = self.resolution // self.grid_cols
        self.cell_height = self.resolution // self.grid_rows
        # unit_size controls the internal resolution for detection (cells are resized to this)
        self.detection_size = self.unit_size
        
        # Parallel launcher state
        self.launcher_process = None
        self.launcher_thread = None
        self.is_running = False
        self.shared_screen = None
        self.screen_lock = threading.Lock()
        
        print(f"🔧 CrafterGridExtractor initialized:")
        print(f"  - Resolution: {self.resolution}x{self.resolution}")
        print(f"  - Grid size: {self.grid_rows}x{self.grid_cols}")
        print(f"  - Screen cell size: {self.cell_width}x{self.cell_height}")
        print(f"  - Detection size: {self.detection_size}x{self.detection_size}")
    
    def _parse_resolution(self, resolution_input) -> int:
        """
        Parse resolution input - handle both string presets, custom mode, and numeric values
        """
        # Resolution presets mapping (same as crafter_interactive_launcher)
        resolution_presets = {
            'tiny': 200,
            'small': 300,
            'low': 400,
            'medium': 600,
            'high': 800,
            'ultra': 1200
        }
        
        if isinstance(resolution_input, str):
            if resolution_input.lower() == 'custom':
                # Custom mode: use explicit width/height from GUI config
                # For custom mode, use the smaller dimension to maintain square aspect ratio
                width = self.gui_config.get('width', 400)
                height = self.gui_config.get('height', 400)
                resolution = min(width, height)  # Use smaller dimension for square rendering
                print(f"✅ Custom resolution mode: using {resolution}x{resolution} (from {width}x{height} window)")
                return resolution
            elif resolution_input.lower() in resolution_presets:
                return resolution_presets[resolution_input.lower()]
            else:
                # Try to parse as string number
                try:
                    return int(resolution_input)
                except ValueError:
                    print(f"⚠️ Unknown resolution preset '{resolution_input}', supported presets: {list(resolution_presets.keys())} + 'custom'")
                    print(f"   Using default 400x400 resolution")
                    return 400
        elif isinstance(resolution_input, (int, float)):
            return int(resolution_input)
        else:
            print(f"⚠️ Invalid resolution type {type(resolution_input)}, using default 400")
            return 400
    
    def start_parallel_launcher(self, max_steps: int = 1000) -> bool:
        """
        Start the parallel crafter_interactive_launcher process with keyboard integration
        
        Args:
            max_steps: Maximum steps for the launcher
            
        Returns:
            bool: True if launcher started successfully
        """
        if self.launcher_process and self.launcher_process.poll() is None:
            print(f"⚠️ Launcher already running (PID: {self.launcher_process.pid})")
            return True
        
        try:
            # Build command for crafter_interactive_launcher
            project_root = Path(__file__).parent.parent
            launcher_script = project_root / "demos" / "crafter_interactive_launcher.py"
            
            if not launcher_script.exists():
                print(f"❌ Launcher script not found: {launcher_script}")
                return False
            
            # Command to run launcher with specified resolution
            cmd = [
                sys.executable, "-m", "demos.crafter_interactive_launcher",
                "--resolution", str(self.resolution),
                "--max-steps", str(max_steps)
                # Note: GUI window will be created for screen capture
            ]
            
            print(f"🚀 Starting parallel launcher with keyboard integration: {' '.join(cmd)}")
            
            # Start launcher in separate process
            self.launcher_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Start monitoring thread
            self.is_running = True
            self.launcher_thread = threading.Thread(
                target=self._monitor_launcher,
                daemon=True
            )
            self.launcher_thread.start()
            
            # Wait a bit for launcher to initialize
            time.sleep(3)
            
            print(f"✅ Parallel launcher started successfully (PID: {self.launcher_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start parallel launcher: {e}")
            return False
    
    def _monitor_launcher(self):
        """
        Monitor the launcher process and capture screen updates
        """
        while self.is_running and self.launcher_process:
            try:
                # Check if process is still running
                if self.launcher_process.poll() is not None:
                    print("⚠️ Launcher process terminated")
                    break
                
                # Try to capture screen (this would need pygame screen sharing)
                # For now, we'll simulate screen capture
                self._simulate_screen_capture()
                
                time.sleep(0.1)  # 10 FPS monitoring
                
            except Exception as e:
                print(f"❌ Error in launcher monitoring: {e}")
                break
        
        self.is_running = False
    
    def update_screen_data(self, screen_data: np.ndarray):
        """
        Update shared screen data from GymAdapter
        
        Args:
            screen_data: Screen image data from GymAdapter's environment
        """
        if screen_data is not None:
            with self.screen_lock:
                # Resize screen data to match expected resolution if needed
                if screen_data.shape[:2] != (self.resolution, self.resolution):
                    self.shared_screen = cv2.resize(screen_data, (self.resolution, self.resolution))
                else:
                    self.shared_screen = screen_data.copy()
    
    def _capture_window_screen(self) -> Optional[np.ndarray]:
        """
        Capture screen content from Crafter GUI window
        
        Returns:
            Screen image as numpy array or None if capture fails
        """
        if not WINDOWS_CAPTURE_AVAILABLE:
            return None
            
        try:
            # Find Crafter window by title
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if "Crafter" in window_title or "pygame" in window_title.lower():
                        windows.append((hwnd, window_title))
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            if not windows:
                return None
                
            # Use the first matching window
            hwnd, title = windows[0]
            
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            # Capture window content
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy window content
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (height, width, 4)  # BGRA format
            img = img[:, :, :3]  # Remove alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Clean up
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            
            # Resize to expected resolution
            if img.shape[:2] != (self.resolution, self.resolution):
                img = cv2.resize(img, (self.resolution, self.resolution))
                
            return img
            
        except Exception as e:
            print(f"❌ Screen capture failed: {e}")
            return None
    
    def _simulate_screen_capture(self):
        """
        Capture screen from GUI window and update shared screen data
        """
        screen_data = self._capture_window_screen()
        if screen_data is not None:
            with self.screen_lock:
                self.shared_screen = screen_data.copy()
    
    def extract_grid_cells(self, screen: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Extract individual grid cells from the screen as objects
        
        Args:
            screen: Screen image to extract from (if None, uses shared screen)
            
        Returns:
            List of grid cell objects with images and metadata
        """
        if screen is None:
            with self.screen_lock:
                if self.shared_screen is None:
                    return []
                screen = self.shared_screen.copy()
        
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
                    import cv2
                    if cell_image.size > 0:
                        cell_image_resized = cv2.resize(cell_image, (self.detection_size, self.detection_size))
                    else:
                        cell_image_resized = cell_image
                    
                    # Create grid object
                    grid_obj = {
                        'type': 'grid_cell',
                        'detector_type': 'crafter_grid_extractor',
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
    
    def get_cell_at_position(self, x: int, y: int, screen: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Get the grid cell at a specific screen position
        
        Args:
            x, y: Screen coordinates
            screen: Screen image (if None, uses shared screen)
            
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
    
    def analyze_cell_content(self, cell_obj: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def extract_grid_objects(self, screen: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract grid cell objects from the screen for BottomUpAgent framework
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
    
    def stop_parallel_launcher(self):
        """
        Stop the parallel launcher process (alias for stop_launcher)
        """
        self.stop_launcher()
    
    def stop_launcher(self):
        """
        Stop the parallel launcher process
        """
        self.is_running = False
        
        if self.launcher_process:
            try:
                self.launcher_process.terminate()
                self.launcher_process.wait(timeout=5)
                print("✅ Launcher process terminated")
            except subprocess.TimeoutExpired:
                self.launcher_process.kill()
                print("⚠️ Launcher process killed (timeout)")
            except Exception as e:
                print(f"❌ Error stopping launcher: {e}")
        
        if self.launcher_thread and self.launcher_thread.is_alive():
            self.launcher_thread.join(timeout=2)
    
    def is_launcher_running(self) -> bool:
        """
        Check if the launcher is currently running
        
        Returns:
            bool: True if launcher is running
        """
        return self.is_running and self.launcher_process and self.launcher_process.poll() is None
    
    def get_launcher_status(self) -> Dict[str, Any]:
        """
        Get current status of the launcher
        
        Returns:
            Status information dictionary
        """
        return {
            'is_running': self.is_launcher_running(),
            'process_id': self.launcher_process.pid if self.launcher_process else None,
            'resolution': self.resolution,
            'grid_size': self.grid_size,
            'has_screen': self.shared_screen is not None
        }
    
    def __del__(self):
        """
        Cleanup when object is destroyed
        """
        self.stop_launcher()