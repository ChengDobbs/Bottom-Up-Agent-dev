import numpy as np
import cv2
import pygame
import gymnasium as gym
import threading
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import json
from PIL import Image
from .Eye import Eye

class GymEnvironmentAdapter:
    """
    Universal adapter for Gym environments to work with Bottom-Up Agent framework.
    Renders any Gym environment as an independent GUI window for visual interaction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_name = config.get('game_name', 'CartPole-v1')
        
        # Read GUI configuration with resolution presets
        gui_config = config.get('gui', {})
        
        # Resolution presets (matching demo implementation)
        presets = {
            'tiny': [200, 200],
            'small': [300, 300], 
            'low': [400, 400],
            'medium': [600, 600],
            'high': [800, 800],
            'ultra': [1200, 1200]
        }
        
        # Determine resolution from preset or explicit values
        preset = gui_config.get('resolution', 'low')
        if preset in presets:
            gui_width, gui_height = presets[preset]
            resolution_display = f"{gui_width}x{gui_height}"
        elif preset == 'custom':
            # Custom mode: use explicit width/height values
            gui_width = gui_config.get('width', 800)
            gui_height = gui_config.get('height', 800)
            resolution_display = f"{gui_width}x{gui_height}"
        else:
            # Fallback for unknown presets
            gui_width = gui_config.get('width', 800)
            gui_height = gui_config.get('height', 800)
            resolution_display = f"{gui_width}x{gui_height}"
            print(f"⚠️ Unknown resolution preset '{preset}', using custom values")
            
        self.window_size = (gui_width, gui_height)
        self.fps = gui_config.get('fps', 60)
        print(f"GUI configured: {gui_width}x{gui_height} (preset: {preset})")
        
        # Create Gym environment based on configuration
        self.env = self._create_environment(config)
        
        # GUI window setup - always show actual pixel dimensions in title
        self.window_title = f"Gym Environment: {self.env_name} [{resolution_display}]"
        self.screen = None
        self.clock = None
        self.running = False
        self.gui_thread = None
        
        # Interactive mode settings
        self.interactive_mode = False
        self.waiting_for_user = False
        self.user_decision = None
        self.decision_lock = threading.Lock()
        
        # State management
        self.current_obs = None
        self.current_reward = 0.0
        self.current_done = False
        self.current_info = {}
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes = 0
        
        # Crafter interactive features
        self.is_crafter = self.env_name == 'Crafter' or (self.env_name and 'crafter' in self.env_name.lower())
        self.crafter_achievements_history = []
        self.crafter_total_reward = 0.0
        self.crafter_total_steps = 0
        
        # Crafter action mappings
        self.crafter_actions = {
            0: "noop", 1: "move_left", 2: "move_right", 3: "move_up", 4: "move_down",
            5: "do", 6: "sleep", 7: "place_stone", 8: "place_table", 9: "place_furnace",
            10: "place_plant", 11: "make_wood_pickaxe", 12: "make_stone_pickaxe", 
            13: "make_iron_pickaxe", 14: "make_wood_sword", 15: "make_stone_sword", 16: "make_iron_sword"
        }
        
        # Crafter achievements list
        self.crafter_achievements = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        # Crafter keyboard mapping
        self.crafter_keyboard_mapping = {
            pygame.K_w: 3,          # Move up
            pygame.K_a: 1,          # Move left  
            pygame.K_s: 4,          # Move down
            pygame.K_d: 2,          # Move right
            pygame.K_SPACE: 5,      # Collect/attack/interact (do)
            pygame.K_TAB: 6,        # Sleep
            pygame.K_t: 8,          # Place table
            pygame.K_r: 7,          # Place rock/stone
            pygame.K_f: 9,          # Place furnace
            pygame.K_p: 10,         # Place plant
            pygame.K_1: 11,         # Craft wood pickaxe
            pygame.K_2: 12,         # Craft stone pickaxe
            pygame.K_3: 13,         # Craft iron pickaxe
            pygame.K_4: 14,         # Craft wood sword
            pygame.K_5: 15,         # Craft stone sword
            pygame.K_6: 16,         # Craft iron sword
        }
        
        # Action mapping for different environment types
        self.action_mapping = self._create_action_mapping()
        self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # State feature extraction parameters
        self.state_feature_dim = config.get('state_feature_dim', 64)
        
        # Initialize pygame for rendering
        pygame.init()
        
        # Initialize detector if specified
        self.detector = None
        if 'detector' in config:
            from .Detector import Detector
            self.detector = Detector(config)
            
            # Set crafter environment reference for crafter_api detector
            if self.detector and self.detector.detector_type == 'crafter_api':
                # Check different ways to access crafter environment
                crafter_env = None
                if hasattr(self.env, '_world'):  # Direct crafter.Env
                    crafter_env = self.env
                elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, '_world'):  # Gymnasium wrapper
                    crafter_env = self.env.unwrapped
                elif hasattr(self.env, 'env') and hasattr(self.env.env, '_world'):  # Nested wrapper
                    crafter_env = self.env.env
                
                if crafter_env:
                    self.detector.crafter_env = crafter_env
                    print(f"✅ Set Crafter environment reference for crafter_api detector")
                else:
                    print(f"⚠️ Could not find Crafter environment with _world attribute")
                    print(f"   Environment type: {type(self.env)}")
                    print(f"   Available attributes: {[attr for attr in dir(self.env) if not attr.startswith('_')][:10]}...")
        
        # Note: Eye module is already initialized in Detector for crafter_api type
        # No need to create additional grid_extractor instance
        self.grid_extractor = None
        if (self.detector is not None and 
            self.detector.detector_type == 'crafter_api' and 
            hasattr(self.detector, 'eye')):
            # Use the Eye instance from Detector to avoid duplication
            self.grid_extractor = self.detector.eye
            print(f"✅ Using Eye module from Detector for grid extraction in {self.env_name}")
        elif self.env_name == 'Crafter' or 'crafter' in self.env_name.lower():
            try:
                self.grid_extractor = Eye(config)
                print(f"✅ Eye module initialized for grid extraction in {self.env_name}")
            except Exception as e:
                print(f"❌ Failed to initialize Eye module: {e}")
                self.grid_extractor = None
    
    def _create_environment(self, config: Dict[str, Any]):
        """Create environment based on configuration"""
        # Get gym configuration from config
        gym_config = config.get('gym_environment', config.get('gym', {}))
        # Get env_type from the main config first, then from gym_config
        env_type = config.get('env_type', gym_config.get('env_type', 'gymnasium'))
        
        # Environment configuration loaded
        
        if env_type == 'crafter_pypi':
            # Import crafter first to trigger environment registration
            try:
                import crafter
                print("✅ Crafter module imported successfully")
                
                # Get Crafter-specific configuration
                detector_config = config.get('detector', {})
                crafter_api_config = detector_config.get('crafter_api', {})
                unit_size = crafter_api_config.get('unit_size', 64)  # Default to 64 if not specified
                
                # Create Crafter environment with unit_size parameter for internal resolution
                # unit_size controls the pixel size of each grid cell, affecting internal rendering resolution
                env = crafter.Env(size=(unit_size, unit_size))
                print(f"✅ Created Crafter environment with unit_size={unit_size} (internal resolution control)")
                print(f"   Note: unit_size controls internal rendering resolution, not window size")
                
                # Set GUI mode if specified
                gui_mode = gym_config.get('gui', True)
                if hasattr(env, 'set_gui_mode'):
                    env.set_gui_mode(gui_mode)
                    print(f"GUI mode set to: {gui_mode}")
                else:
                    print(f"Note: GUI mode setting not available, using default behavior")
                    
                return env
            except ImportError:
                print("❌ Crafter not available, falling back to CartPole")
                # Final fallback to CartPole
                try:
                    env = gym.make('CartPole-v1')
                    print("🔄 Falling back to CartPole-v1")
                    return env
                except Exception as e:
                    raise e
        
        # Use gymnasium.make for standard environments
        try:
            # For GUI environments, use render_mode='human' to enable window display
            if self.env_name in ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']:
                env = gym.make(self.env_name, render_mode='human')
                print(f"✅ Created {self.env_name} environment with GUI mode via gymnasium")
            else:
                env = gym.make(self.env_name)
                print(f"✅ Created {self.env_name} environment via gymnasium")
            return env
        except Exception as e:
            print(f"❌ Failed to create environment {self.env_name}: {e}")
            # Try fallback to CartPole if the requested environment fails
            if self.env_name != 'CartPole-v1':
                print("🔄 Falling back to CartPole-v1")
                return gym.make('CartPole-v1', render_mode='human')
            else:
                raise e
        
    def _create_action_mapping(self) -> Dict[str, int]:
        """Create action mapping based on environment action space"""
        action_space = self.env.action_space
        
        if hasattr(action_space, 'n'):  # Discrete action space
            mapping = {}
            for i in range(action_space.n):
                mapping[f'action_{i}'] = i
            
            # Add common action names for popular environments
            if 'CartPole' in self.env_name:
                mapping.update({'left': 0, 'right': 1})
            elif 'LunarLander' in self.env_name:
                mapping.update({
                    'noop': 0, 'fire_left': 1, 'fire_main': 2, 'fire_right': 3
                })
            elif 'Breakout' in self.env_name or 'Pong' in self.env_name:
                mapping.update({
                    'noop': 0, 'fire': 1, 'right': 2, 'left': 3, 'right_fire': 4, 'left_fire': 5
                })
            
            return mapping
        else:
            # Continuous action space - create discrete approximations
            return {'action_0': 0}  # Simplified for continuous spaces
    
    def start_gui(self, interactive_mode=False):
        """Start the GUI window in a separate thread"""
        if not self.running:
            # Initialize pygame
            pygame.init()
            self.interactive_mode = interactive_mode
            self.running = True
            self.gui_thread = threading.Thread(target=self._run_gui_loop, daemon=True)
            self.gui_thread.start()
            mode_text = "Interactive" if interactive_mode else "Observation"
            print(f"GUI started in separate thread - {mode_text} mode")
            
            # Print Crafter controls if this is a Crafter environment
            if self.is_crafter and interactive_mode:
                self._print_crafter_controls()
            
            time.sleep(0.5)  # Give GUI time to initialize
    
    def start_crafter_interactive(self, max_steps=10000):
        """Start Crafter in interactive mode with full controls"""
        if not self.is_crafter:
            print("❌ This method is only available for Crafter environments")
            return False
        
        print("🎮 Starting Crafter Interactive Mode...")
        print("📋 Use keyboard controls to play Crafter manually")
        
        # Start GUI in interactive mode
        self.start_gui(interactive_mode=True)
        
        # Reset environment to start fresh
        self.reset()
        
        print(f"🚀 Crafter Interactive Mode started! Max steps: {max_steps}")
        print("   Press ESC to quit")
        
        return True
    
    def _print_crafter_controls(self):
        """Print Crafter keyboard controls"""
        print("\n🎮 Crafter Controls:")
        print("   WASD: Move (W=Up, A=Left, S=Down, D=Right)")
        print("   SPACE: Do (collect/attack/interact)")
        print("   TAB: Sleep")
        print("   T: Place Table, R: Place Stone, F: Place Furnace, P: Place Plant")
        print("   1-6: Craft items (1=Wood Pickaxe, 2=Stone Pickaxe, 3=Iron Pickaxe,")
        print("                    4=Wood Sword, 5=Stone Sword, 6=Iron Sword)")
        print("   ESC: Quit")
        print()
    
    def stop_gui(self):
        """Stop the GUI window"""
        self.running = False
        if hasattr(self, 'gui_thread') and self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=2.0)
        try:
            pygame.quit()
        except:
            pass  # Ignore pygame quit errors
    
    def _run_gui_loop(self):
        """Main GUI rendering loop"""
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(self.window_title)
        self.clock = pygame.time.Clock()
        
        while self.running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keyboard_input(event.key)
            
            # Render current environment state
            self._render_environment()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)
    
    def _handle_keyboard_input(self, key):
        """Handle keyboard input for manual control (optional)"""
        # Handle Crafter-specific keyboard mapping
        if self.is_crafter and key in self.crafter_keyboard_mapping:
            action = self.crafter_keyboard_mapping[key]
            # Execute action and print action name for feedback
            action_name = self.crafter_actions.get(action, f"action_{action}")
            print(f"🎮 Crafter Action: {action_name} (key: {pygame.key.name(key)})")
            self.step(action)
            return
        
        # Handle ESC key to quit
        if key == pygame.K_ESCAPE:
            print("🚪 ESC pressed - stopping GUI...")
            self.running = False
            return
            
        # Map common keys to actions for non-Crafter environments
        key_mapping = {
            pygame.K_LEFT: 'left',
            pygame.K_RIGHT: 'right',
            pygame.K_UP: 'up', 
            pygame.K_DOWN: 'down',
            pygame.K_SPACE: 'fire',
            pygame.K_RETURN: 'action_0'
        }
        
        if key in key_mapping:
            action_name = key_mapping[key]
            if action_name in self.action_mapping:
                action = self.action_mapping[action_name]
                # Execute action (for manual testing)
                self.step(action)
        
        # Handle interactive mode decisions
        if self.interactive_mode and self.waiting_for_user:
            self._handle_interactive_input(key)
    
    def _handle_interactive_input(self, key):
        """Handle user input in interactive decision mode"""
        with self.decision_lock:
            if key == pygame.K_1:
                self.user_decision = {"action_type": "click", "object_index": 0}
            elif key == pygame.K_2:
                self.user_decision = {"action_type": "click", "object_index": 1}
            elif key == pygame.K_3:
                self.user_decision = {"action_type": "click", "object_index": 2}
            elif key == pygame.K_4:
                self.user_decision = {"action_type": "click", "object_index": 3}
            elif key == pygame.K_5:
                self.user_decision = {"action_type": "click", "object_index": 4}
            elif key == pygame.K_r:
                self.user_decision = {"action_type": "random"}
            elif key == pygame.K_s:
                self.user_decision = {"action_type": "skip"}
            elif key == pygame.K_q:
                self.user_decision = {"action_type": "quit"}
            
            if self.user_decision:
                self.waiting_for_user = False
                print(f"User selected: {self.user_decision}")
    
    def wait_for_user_decision(self, available_objects, timeout=None):
        """Wait for user to make a decision through GUI"""
        print("\n=== Waiting for User Decision ===")
        print("Available actions:")
        if available_objects:
            for i, obj in enumerate(available_objects[:5]):  # Limit to 5 objects
                print(f"  Press {i+1}: Click on {obj.get('type', 'object')} - '{obj.get('content', '')[:30]}'")
        print("  Press R: Random action")
        print("  Press S: Skip this step")
        print("  Press Q: Quit")
        
        with self.decision_lock:
            self.waiting_for_user = True
            self.user_decision = None
        
        # Wait for user decision
        start_time = time.time()
        while self.waiting_for_user:
            if timeout and (time.time() - start_time) > timeout:
                print("Timeout reached, using random action")
                return {"action_type": "random"}
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Check if GUI is still running
            if not self.running:
                return {"action_type": "quit"}
        
        return self.user_decision
    
    def _render_environment(self):
        """Render the current environment state"""
        if self.current_obs is None:
            return
        
        try:
            # For environments with render_mode='human', just call render() to display
            if hasattr(self.env, 'render'):
                # Check if environment was created with render_mode='human'
                if (hasattr(self.env, 'render_mode') and self.env.render_mode == 'human') or \
                   (self.env_name in ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']):
                    # Direct rendering to environment's own window
                    self.env.render()
                    return
                
                # For other environments, try to get image data
                try:
                    # Check if this is a Crafter environment that supports size parameter
                    if hasattr(self.env, 'render') and 'crafter' in str(type(self.env)).lower():
                        img = self.env.render(size=self.window_size)
                    else:
                        img = self.env.render(mode='rgb_array')
                except:
                    try:
                        # Fallback: try render without parameters
                        if hasattr(self.env, 'render') and 'crafter' in str(type(self.env)).lower():
                            img = self.env.render(size=self.window_size)
                        else:
                            img = self.env.render()
                        if img is None:
                            img = self._create_state_visualization()
                    except:
                        img = self._create_state_visualization()
                
                if img is not None and len(img.shape) == 3:
                    # Resize image to fit window
                    img_surface = self._numpy_to_pygame_surface(img)
                    img_surface = pygame.transform.scale(img_surface, self.window_size)
                    self.screen.blit(img_surface, (0, 0))
                else:
                    # Fallback: create simple visualization
                    self.screen.fill((50, 50, 50))
                    self._draw_state_info()
            else:
                # No render method available
                img = self._create_state_visualization()
                img_surface = self._numpy_to_pygame_surface(img)
                img_surface = pygame.transform.scale(img_surface, self.window_size)
                self.screen.blit(img_surface, (0, 0))
        
        except Exception as e:
            # Fallback rendering
            if hasattr(self, 'screen') and self.screen is not None:
                self.screen.fill((100, 0, 0))  # Red background for errors
                self._draw_error_info(str(e))
    
    def _create_state_visualization(self) -> np.ndarray:
        """Create a visual representation of the current state"""
        # Create a simple visualization based on observation
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        if self.current_obs is not None:
            obs = np.array(self.current_obs)
            if obs.ndim == 1:
                # 1D observation - create bar chart visualization
                height, width = img.shape[:2]
                bar_width = width // len(obs)
                
                for i, val in enumerate(obs):
                    # Normalize value to 0-255 range
                    normalized_val = int(abs(val) * 50) % 255
                    color = (normalized_val, 100, 255 - normalized_val)
                    
                    x = i * bar_width
                    bar_height = int(abs(val) * 100) % height
                    
                    pygame.draw.rect(
                        pygame.surfarray.make_surface(img.transpose(1, 0, 2)),
                        color,
                        (x, height - bar_height, bar_width - 1, bar_height)
                    )
        
        return img
    
    def _numpy_to_pygame_surface(self, img: np.ndarray):
        """Convert numpy array to pygame surface"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img.transpose(1, 0, 2)  # Pygame expects (width, height, channels)
            return pygame.surfarray.make_surface(img)
        else:
            # Convert grayscale to RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            img = img.transpose(1, 0, 2)
            return pygame.surfarray.make_surface(img)
    
    def _draw_state_info(self):
        """Draw state information on screen"""
        font = pygame.font.Font(None, 36)
        
        # Draw episode info
        info_text = [
            f"Episode: {self.total_episodes}",
            f"Steps: {self.episode_steps}", 
            f"Reward: {self.episode_reward:.2f}",
            f"Current Reward: {self.current_reward:.2f}",
            f"Done: {self.current_done}"
        ]
        
        # Add interactive mode info
        if self.interactive_mode:
            info_text.append("Mode: Interactive")
            if self.waiting_for_user:
                info_text.append("Waiting for input...")
        else:
            info_text.append("Mode: Observation")
        
        for i, text in enumerate(info_text):
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 40))
    
    def _draw_error_info(self, error_msg: str):
        """Draw error information"""
        font = pygame.font.Font(None, 24)
        surface = font.render(f"Render Error: {error_msg}", True, (255, 255, 255))
        self.screen.blit(surface, (10, 10))
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation"""
        obs = self.env.reset()
        
        # Handle different gym versions
        if isinstance(obs, tuple):
            obs = obs[0]  # New gym versions return (obs, info)
        
        self.current_obs = obs
        self.current_reward = 0.0
        self.current_done = False
        self.current_info = {}
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes += 1
        
        # Only start our own GUI if environment doesn't have render_mode='human'
        # For render_mode='human', the environment handles its own window
        env_render_mode = getattr(self.env, 'render_mode', None)
        if not self.running and env_render_mode != 'human':
            self.start_gui()
        
        return self._process_observation(obs)
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return observation, reward, done, info"""
        # Handle different action space types
        if hasattr(self.env.action_space, 'n'):
            # Discrete action space
            action = int(action) % self.env.action_space.n
        else:
            # Continuous action space - convert to appropriate format
            if hasattr(self.env.action_space, 'shape'):
                action_dim = self.env.action_space.shape[0]
                action = np.array([action] * action_dim, dtype=np.float32)
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        # Execute step
        result = self.env.step(action)
        
        # Handle different gym versions
        if len(result) == 4:
            obs, reward, done, info = result
        else:  # New gym versions return 5 values
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        # Update state
        self.current_obs = obs
        self.current_reward = reward
        self.current_done = done
        self.current_info = info
        self.episode_steps += 1
        self.episode_reward += reward
        
        # Update Crafter-specific tracking
        if self.is_crafter:
            self.crafter_total_reward += reward
            self.crafter_total_steps += 1
        
        processed_obs = self._process_observation(obs)
        
        # Add environment info
        info.update({
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'total_episodes': self.total_episodes
        })
        
        # Handle episode completion
        if done:
            # Track Crafter achievements if this is a Crafter environment
            if self.is_crafter and 'achievements' in info:
                self.crafter_achievements_history.append(info['achievements'])
                self._print_crafter_stats()
        
        return processed_obs, reward, done, info
    
    def _process_observation(self, obs: Union[np.ndarray, List]) -> Dict[str, Any]:
        """Process Gym observation to Bottom-Up Agent format"""
        # Convert observation to numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Create screen representation
        screen = self._create_screen_from_obs(obs)
        
        # Extract state features
        state_feature = self._extract_state_features(obs)
        
        return {
            'screen': screen,
            'state_feature': state_feature,
            'raw_obs': obs
        }
    
    def _create_screen_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """Create screen image from observation"""
        try:
            # Try to get rendered image
            if hasattr(self.env, 'render'):
                try:
                    screen = self.env.render(mode='rgb_array')
                    if screen is not None:
                        return screen.astype(np.uint8)
                except:
                    pass
            
            # Fallback: create visualization from observation
            if len(obs.shape) == 3 and obs.shape[2] in [1, 3, 4]:
                # Image-like observation
                if obs.shape[2] == 1:
                    screen = np.repeat(obs, 3, axis=2)
                elif obs.shape[2] == 4:
                    screen = obs[:, :, :3]  # Remove alpha channel
                else:
                    screen = obs
                
                # Ensure uint8 format
                if screen.dtype != np.uint8:
                    if screen.max() <= 1.0:
                        screen = (screen * 255).astype(np.uint8)
                    else:
                        screen = screen.astype(np.uint8)
                
                return screen
            else:
                # Create simple visualization for non-image observations
                return self._create_state_visualization()
        
        except Exception:
            # Ultimate fallback
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def _extract_state_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract compact state features from observation"""
        # Flatten observation
        features = obs.flatten().astype(np.float32)
        
        # Normalize features
        if features.max() > 1.0:
            features = features / 255.0
        
        # Pad or truncate to desired dimension
        if len(features) > self.state_feature_dim:
            features = features[:self.state_feature_dim]
        elif len(features) < self.state_feature_dim:
            features = np.pad(features, (0, self.state_feature_dim - len(features)))
        
        return features
    
    def map_operation_to_action(self, operation: Dict[str, Any]) -> int:
        """Map Bottom-Up Agent operation to Gym action"""
        operate = operation.get('operate', 'noop')
        
        # Handle Crafter-specific operations first
        if self.is_crafter:
            crafter_operation_mapping = {
                'move_up': 3,
                'move_down': 4,
                'move_left': 1,
                'move_right': 2,
                'interact': 5,
                'sleep': 6,
                'place_stone': 7,
                'place_table': 8,
                'place_furnace': 9,
                'place_plant': 10,
                'plant_sapling': 10,  # Alias for place_plant
                'craft_wood_pickaxe': 11,
                'craft_stone_pickaxe': 12,
                'craft_iron_pickaxe': 13,
                'craft_wood_sword': 14,
                'craft_stone_sword': 15,
                'craft_iron_sword': 16
            }
            
            # Try direct mapping for Crafter operations
            if operate in crafter_operation_mapping:
                action = crafter_operation_mapping[operate]
                print(f"🎯 Crafter operation '{operate}' mapped to action {action}")
                return action
            
            # Handle variations and partial matches for Crafter
            operate_lower = operate.lower()
            if 'move' in operate_lower:
                if 'up' in operate_lower:
                    return 3
                elif 'down' in operate_lower:
                    return 4
                elif 'left' in operate_lower:
                    return 1
                elif 'right' in operate_lower:
                    return 2
            
            if any(word in operate_lower for word in ['interact', 'collect', 'attack', 'do']):
                return 5  # Do action
            
            if 'sleep' in operate_lower:
                return 6
            
            if 'place' in operate_lower:
                if 'stone' in operate_lower:
                    return 7
                elif 'table' in operate_lower:
                    return 8
                elif 'furnace' in operate_lower:
                    return 9
                elif 'plant' in operate_lower or 'sapling' in operate_lower:
                    return 10
            
            if 'craft' in operate_lower or 'make' in operate_lower:
                if 'wood' in operate_lower and 'pickaxe' in operate_lower:
                    return 11
                elif 'stone' in operate_lower and 'pickaxe' in operate_lower:
                    return 12
                elif 'iron' in operate_lower and 'pickaxe' in operate_lower:
                    return 13
                elif 'wood' in operate_lower and 'sword' in operate_lower:
                    return 14
                elif 'stone' in operate_lower and 'sword' in operate_lower:
                    return 15
                elif 'iron' in operate_lower and 'sword' in operate_lower:
                    return 16
        
        # Handle different operation types (generic)
        if operate in ['Click', 'LeftSingle']:
            params = operation.get('params', {})
            if 'coordinate' in params:
                x, y = params['coordinate']
            else:
                x, y = params.get('x', 0), params.get('y', 0)
            
            # Map coordinates to actions based on screen regions
            screen_width, screen_height = self.window_size
            
            if x < screen_width // 3:
                return self.action_mapping.get('left', 0)
            elif x > 2 * screen_width // 3:
                return self.action_mapping.get('right', 1)
            else:
                return self.action_mapping.get('fire', 0)
        
        elif operate == 'KeyPress':
            key = operation.get('params', {}).get('key', '')
            key_action_map = {
                'left': 'left', 'right': 'right', 'up': 'up', 'down': 'down',
                'space': 'fire', 'enter': 'action_0'
            }
            action_name = key_action_map.get(key, 'action_0')
            return self.action_mapping.get(action_name, 0)
        
        # Default action
        print(f"⚠️ Unknown operation '{operate}', using default action 0")
        return 0
    
    def detect_objects(self, screen: np.ndarray) -> List[Dict[str, Any]]:
        """Detect interactive objects in the environment screen"""
        objects = []
        
        # Use configured detector if available
        if self.detector:
            try:
                detected_objects = self.detector.get_detected_objects(screen)
                objects.extend(detected_objects)
                print(f"✅ Detected {len(detected_objects)} objects using {self.detector.detector_type} detector")
                
                # For crafter_api detector, grid objects are already included in detected_objects
                # No need to add them separately to avoid duplication
                if self.detector.detector_type == 'crafter_api' and self.grid_extractor:
                    print(f"🔲 Grid objects already included in crafter_api detection (total: {len(detected_objects)})")
                    # Debug: Check if we have the expected 7x9=63 objects
                    if len(detected_objects) == 63:
                        print(f"✅ Correct grid size: 7x9 = 63 objects detected")
                    else:
                        print(f"⚠️ Unexpected object count: {len(detected_objects)} (expected 63 for 7x9 grid)")
                
                return objects
            except Exception as e:
                print(f"⚠️ Error using {self.detector.detector_type} detector: {e}")
                print("Falling back to default object detection")
        
        # Fallback: Add basic action regions based on environment type
        screen_height, screen_width = screen.shape[:2]
        
        if 'CartPole' in self.env_name:
            objects.extend([
                {
                    'id': 'move_left',
                    'type': 'action',
                    'content': 'Move Left',
                    'center': [screen_width // 4, screen_height // 2],
                    'interactivity': 'clickable'
                },
                {
                    'id': 'move_right', 
                    'type': 'action',
                    'content': 'Move Right',
                    'center': [3 * screen_width // 4, screen_height // 2],
                    'interactivity': 'clickable'
                }
            ])
        
        elif 'LunarLander' in self.env_name:
            objects.extend([
                {
                    'id': 'fire_left',
                    'type': 'action', 
                    'content': 'Fire Left Engine',
                    'center': [screen_width // 4, screen_height // 3],
                    'interactivity': 'clickable'
                },
                {
                    'id': 'fire_main',
                    'type': 'action',
                    'content': 'Fire Main Engine', 
                    'center': [screen_width // 2, screen_height // 4],
                    'interactivity': 'clickable'
                },
                {
                    'id': 'fire_right',
                    'type': 'action',
                    'content': 'Fire Right Engine',
                    'center': [3 * screen_width // 4, screen_height // 3],
                    'interactivity': 'clickable'
                }
            ])
        
        else:
            # Generic action regions for unknown environments
            for i in range(min(4, len(self.action_mapping))):
                objects.append({
                    'id': f'action_{i}',
                    'type': 'action',
                    'content': f'Action {i}',
                    'center': [(i + 1) * screen_width // 5, screen_height // 2],
                    'interactivity': 'clickable'
                })
        
        return objects
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            'env_name': self.env_name,
            'action_space': str(self.env.action_space),
            'observation_space': str(self.env.observation_space),
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'total_episodes': self.total_episodes,
            'current_done': self.current_done
        }
    
    def render(self):
        """Render the environment - ensure GUI is running and updated"""
        # For environments with render_mode='human', render directly
        if hasattr(self.env, 'render_mode') and self.env.render_mode == 'human':
            # Environment handles its own window, just call render
            self.env.render()
            return
        
        # For other environments, use pygame window
        if not self.running:
            self.start_gui()
        
        # Force a render update by setting a flag
        if hasattr(self, 'screen') and self.screen is not None:
            self._render_environment()
            pygame.display.flip()
    
    def _compute_crafter_success_rates(self):
        """Calculate achievement success rates for Crafter"""
        if not self.crafter_achievements_history:
            return {name: 0.0 for name in self.crafter_achievements}
        
        success_rates = {}
        total_episodes = len(self.crafter_achievements_history)
        
        for achievement in self.crafter_achievements:
            success_count = sum(1 for episode_achievements in self.crafter_achievements_history 
                              if episode_achievements.get(achievement, 0) > 0)
            success_rates[achievement] = (success_count / total_episodes) * 100
        
        return success_rates
    
    def _compute_crafter_score(self, success_rates):
        """Calculate Crafter score (geometric mean, offset by 1%)"""
        rates = [success_rates[name] for name in self.crafter_achievements]
        
        # Use geometric mean to calculate score, offset by 1%
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Geometric mean: exp(mean(log(1 + rates))) - 1
            log_rates = [np.log(1 + rate) for rate in rates if rate >= 0]
            if log_rates:
                score = np.exp(np.mean(log_rates)) - 1
            else:
                score = 0.0
        
        return score
    
    def _print_crafter_stats(self):
        """Print Crafter-specific statistics"""
        if not self.is_crafter or not self.crafter_achievements_history:
            return
        
        latest_achievements = self.crafter_achievements_history[-1]
        total_achievements = sum(latest_achievements.values())
        
        # Calculate success rates and score
        success_rates = self._compute_crafter_success_rates()
        crafter_score = self._compute_crafter_score(success_rates)
        
        print(f"\n🏆 Crafter Episode Stats:")
        print(f"   Episodes: {len(self.crafter_achievements_history)}")
        print(f"   Total Achievements: {total_achievements}")
        print(f"   Crafter Score: {crafter_score:.2f}")
        print(f"   Total Reward: {self.crafter_total_reward:.2f}")
        print(f"   Total Steps: {self.crafter_total_steps}")
        
        # Show achieved milestones
        achieved = [name for name, count in latest_achievements.items() if count > 0]
        if achieved:
            print(f"   Achieved: {', '.join(achieved[:5])}{'...' if len(achieved) > 5 else ''}")
    
    def close(self):
        """Close the environment and GUI"""
        self.stop_gui()
        
        # Clean up Eye module if initialized
        if self.grid_extractor:
            try:
                # Eye module doesn't need special cleanup like CrafterGridExtractor
                print(f"🛑 Eye module cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning up Eye module: {e}")
            finally:
                self.grid_extractor = None
        
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()


class GymAdapterFactory:
    """Factory class for creating Gym adapters for different environments"""
    
    @staticmethod
    def create_adapter(env_name: str, config: Dict[str, Any] = None) -> GymEnvironmentAdapter:
        """Create a Gym adapter for the specified environment"""
        if config is None:
            config = {}
        
        # Set environment-specific defaults
        # Only set game_name if it's not already provided in config
        if 'game_name' not in config:
            config['game_name'] = env_name
        
        # Environment-specific configurations
        if 'CartPole' in env_name:
            config.setdefault('window_size', (600, 400))
            config.setdefault('fps', 60)
        elif 'LunarLander' in env_name:
            config.setdefault('window_size', (600, 400))
            config.setdefault('fps', 50)
        elif 'Breakout' in env_name or 'Pong' in env_name:
            config.setdefault('window_size', (160, 210))
            config.setdefault('fps', 60)
        else:
            config.setdefault('window_size', (800, 600))
            config.setdefault('fps', 60)
        
        return GymEnvironmentAdapter(config)
    
    @staticmethod
    def list_available_environments() -> List[str]:
        """List all available Gym environments"""
        try:
            from gym import envs
            return [env_spec.id for env_spec in envs.registry.all()]
        except:
            # Fallback list of common environments
            return [
                'CartPole-v1', 'LunarLander-v2', 'MountainCar-v0',
                'Acrobot-v1', 'Pendulum-v1', 'BipedalWalker-v3',
                'Breakout-v4', 'Pong-v4', 'SpaceInvaders-v4'
            ]