from base_model import creat_base_model
import pyautogui
import re
import time

class Hand:
    def __init__(self, config):

        self.game_name = config['game_name']
        print(f"Hand initialized for game {self.game_name}")
        

    def move(self, x, y, duration=0.5):
        pyautogui.moveTo(x, y, duration)

    def left_single_click(self, x, y):
        pyautogui.moveTo(x, y, duration=0.05)
        pyautogui.mouseDown()
        pyautogui.mouseUp()

    def right_single_click(self, x, y):
        pyautogui.moveTo(x, y, duration=0.05)
        pyautogui.mouseDown(button='right')
        pyautogui.mouseUp(button='right')


    def do_operation(self, operation: dict, left: int = 0, top: int = 0):
        """
        Execute a unified operation with optional offset coordinates.
        
        Args:
            operation (dict): A unified operation dictionary from UnifiedOperation class
            left (int): X coordinate offset
            top (int): Y coordinate offset
        """
        operate = operation["operate"]
        params = operation["params"]
        
        if operate == "Click":
            x = params["x"] + left
            y = params["y"] + top
            self.left_single_click(x, y)
            print(f"Clicked at ({x}, {y})")
            
        elif operate == "Drag":
            x1 = params["x1"] + left
            y1 = params["y1"] + top
            x2 = params["x2"] + left
            y2 = params["y2"] + top
            pyautogui.moveTo(x1, y1, duration=0.5)
            pyautogui.dragTo(x2, y2, duration=1)
            print(f"Dragged from ({x1}, {y1}) to ({x2}, {y2})")
            
        elif operate == "Scroll":
            x = params["x"] + left
            y = params["y"] + top
            direction = params["direction"]
            if direction == "up":
                pyautogui.scroll(1)
            else:
                pyautogui.scroll(-1)
            print(f"Scrolled {direction} at ({x}, {y})")
            
        elif operate == "Type":
            text = params["content"]
            pyautogui.typewrite(text)
            print(f"Typed '{text}'")
            
        elif operate == "Wait":
            time.sleep(0.5)  # Default wait time
            print("Waited")
            
        elif operate == "Finished":
            print("Task finished")
            
        elif operate == "CallUser":
            print("Calling user")
            
        elif operate == "Hotkey":
            key = params["key"]
            pyautogui.hotkey(key)
            print(f"Pressed hotkey '{key}'")
            
        elif operate == "LeftDouble":
            x = params["x"] + left
            y = params["y"] + top
            pyautogui.doubleClick(x, y)
            print(f"Double clicked at ({x}, {y})")
            
        elif operate == "RightSingle":
            x = params["x"] + left
            y = params["y"] + top
            pyautogui.click(x, y, button='right')
            print(f"Right clicked at ({x}, {y})")
            
        elif operate == "LongPress":
            x = params["x"] + left
            y = params["y"] + top
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.mouseDown()
            time.sleep(1)  # Long press duration
            pyautogui.mouseUp()
            print(f"Long pressed at ({x}, {y})")
            
        elif operate == "PressBack":
            pyautogui.press('backspace')
            print("Pressed back")
            
        elif operate == "PressHome":
            pyautogui.press('home')
            print("Pressed home")
            
        elif operate == "PressEnter":
            pyautogui.press('enter')
            print("Pressed enter")
            
        else:
            print(f"Unknown operation: {operate}")

