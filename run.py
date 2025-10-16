import yaml
import argparse
import sys
from pathlib import Path
from BottomUpAgent.BottomUpAgent import BottomUpAgent
from BottomUpAgent.Eye import Eye

def check_window_exists(config):
    """Check if the game window exists before initializing the agent"""
    try:
        # Create a temporary Eye instance to check window
        temp_eye = Eye(config)
        window_name = config['game_name']
        
        # Try to find the window
        if temp_eye.platform == 'windows':
            window_info = temp_eye._find_window_windows(window_name)
        elif temp_eye.platform == 'linux':
            window_info = temp_eye._find_window_linux(window_name)
        else:
            print(f"Unsupported platform: {temp_eye.platform}")
            return False
            
        if not window_info:
            print(f"Window '{window_name}' not found on {temp_eye.platform}, please launch the game before starting the run.")
            return False
            
        print(f"[INIT] Window '{window_name}' found, proceeding with agent initialization.")
        return True
        
    except Exception as e:
        print(f"[INIT] Error checking window: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    if not check_window_exists(config):
        print("[INIT] Cannot proceed without game window. Please launch the game first.")
        sys.exit(1)
        
    agent = BottomUpAgent(config)
    task = config.get('task', 'Play the game')
    print(f"[INIT] Task: {task}")
    agent.run(task, max_step=config.get('max_step', 1000))
