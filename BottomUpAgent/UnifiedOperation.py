from dataclasses import dataclass
from typing import Optional, Tuple, Literal

@dataclass
class UnifiedOperation:
    # Shared Operations
    @staticmethod
    def Click(x: int, y: int) -> dict:
        return {"operate": "Click", "params": {"x": x, "y": y}}
    
    @staticmethod
    def Hover(x: int, y: int) -> dict:
        return {"operate": "Hover", "params": {"x": x, "y": y}}

    @staticmethod
    def Drag(x1: int, y1: int, x2: int, y2: int) -> dict:
        return {"operate": "Drag", "params": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}

    @staticmethod
    def Scroll(x: int, y: int, direction: str) -> dict:
        return {"operate": "Scroll", "params": {"x": x, "y": y, "direction": direction}}

    @staticmethod
    def Type(content: str) -> dict:
        return {"operate": "Type", "params": {"content": content}}

    @staticmethod
    def Wait() -> dict:
        return {"operate": "Wait", "params": {}}

    @staticmethod
    def Finished() -> dict:
        return {"operate": "Finished", "params": {}}

    @staticmethod
    def CallUser() -> dict:
        return {"operate": "CallUser", "params": {}}

    # Desktop Operations
    @staticmethod
    def Hotkey(key: str) -> dict:
        return {"operate": "Hotkey", "params": {"key": key}}

    @staticmethod
    def LeftDouble(x: int, y: int) -> dict:
        return {"operate": "LeftDouble", "params": {"x": x, "y": y}}

    @staticmethod
    def RightSingle(x: int, y: int) -> dict:
        return {"operate": "RightSingle", "params": {"x": x, "y": y}}

    # Mobile Operations
    @staticmethod
    def LongPress(x: int, y: int) -> dict:
        return {"operate": "LongPress", "params": {"x": x, "y": y}}

    @staticmethod
    def PressBack() -> dict:
        return {"operate": "PressBack", "params": {}}

    @staticmethod
    def PressHome() -> dict:
        return {"operate": "PressHome", "params": {}}

    @staticmethod
    def PressEnter() -> dict:
        return {"operate": "PressEnter", "params": {}}
    
