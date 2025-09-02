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

        # Initialize platform-specific components
        if self.platform == 'linux' and XLIB_AVAILABLE:
            self.display = display.Display()
        else:
            self.display = None

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