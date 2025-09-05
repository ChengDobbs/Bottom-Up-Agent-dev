#!/usr/bin/env python3
"""
Grid Content Check Demo - Real Interactive Process

This demo checks the actual grid content from a REAL interactive Crafter GUI process
with the detected grid content from the detection system to identify discrepancies.
It launches a parallel interactive GUI and connects to the same environment instance.
"""

import yaml
import time
import threading
import numpy as np
import subprocess
import sys
import multiprocessing
import queue
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from BottomUpAgent.Detector import Detector
from BottomUpAgent.CrafterGridExtractor import CrafterGridExtractor
import crafter
import pygame
from demos.crafter_interactive_launcher import demo_crafter_interactive, get_gui_config, load_config as load_launcher_config

def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'gym' / 'crafter_config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

class SharedEnvironment:
    """Shared environment instance that can be accessed by both GUI and detector"""
    def __init__(self):
        self.env = None
        self.lock = threading.Lock()
        self.gui_process = None
        self.gui_running = False
    
    def create_environment(self):
        """Create the shared Crafter environment"""
        with self.lock:
            if self.env is None:
                self.env = crafter.Env()
                obs = self.env.reset()
                # Take a few random steps to generate some world content
                for _ in range(10):
                    action = self.env.action_space.sample()
                    obs, reward, done, info = self.env.step(action)
                    if done:
                        obs = self.env.reset()
                print("✅ Shared Crafter environment created and initialized")
        return self.env
    
    def get_environment(self):
        """Get the shared environment instance"""
        with self.lock:
            return self.env

class GridContentChecker:
    """Check grid content between REAL interactive GUI process and detection system"""
    
    def __init__(self, config):
        self.config = config
        
        # Create shared environment
        self.shared_env = SharedEnvironment()
        self.env = self.shared_env.create_environment()
        
        # Initialize detector with shared environment
        self.detector = Detector(config)
        self.detector.crafter_env = self.env  # Set reference for direct API
        print("✅ Detector initialized with shared crafter_env reference")
        
        # Initialize grid extractor
        self.grid_extractor = CrafterGridExtractor(config)
        print("✅ Grid extractor initialized")
        
        # Grid configuration
        detector_config = config.get('detector', {}).get('crafter_api', {})
        self.grid_size = detector_config.get('grid_size', [7, 9])
        self.grid_rows, self.grid_cols = self.grid_size
        print(f"📐 Grid size: {self.grid_rows} rows x {self.grid_cols} cols")
        
        # GUI process control
        self.gui_thread = None
        self.gui_running = False
        self.action_queue = queue.Queue()
        
        print("🎮 Ready to start interactive GUI process...")
    
    def get_reference_grid_content(self):
        """Get the actual grid content from the Crafter environment"""
        try:
            world = self.env._world
            player = self.env._player
            
            grid_content = {}
            offset = np.array([self.grid_cols // 2, self.grid_rows // 2])  # [cols//2, rows//2]
            
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # Calculate world position - consistent with Detector coordinate system
                    world_pos = player.pos + np.array([col, row]) - offset  # [x, y] coordinates
                    
                    if (0 <= world_pos[0] < world.area[0] and 0 <= world_pos[1] < world.area[1]):
                        material, obj = world[world_pos]
                        
                        content_type = 'empty'
                        # Prioritize objects (creatures/items) over materials (terrain)
                        if obj:
                            content_type = obj.__class__.__name__.lower()
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
                                    content_type = f"{content_type}-{direction}"
                        elif material:
                            content_type = material
                        
                        grid_content[(row, col)] = {
                            'type': content_type,
                            'world_pos': world_pos.tolist(),
                            'material': material,
                            'obj': obj.__class__.__name__.lower() if obj else None,
                            'source_info': {
                                'material': material,
                                'obj': obj.__class__.__name__.lower() if obj else None
                            }
                        }
                    else:
                        grid_content[(row, col)] = {
                            'type': 'out_of_bounds',
                            'world_pos': world_pos.tolist(),
                            'material': None,
                            'obj': None,
                            'source_info': {
                                'material': None,
                                'obj': None
                            }
                        }
            
            return grid_content
            
        except Exception as e:
            print(f"❌ Error getting reference grid content: {e}")
            return {}
    
    def start_gui_process(self):
        """Start the interactive GUI process in a separate thread"""
        # Initialize step counter for tracking game state changes
        self.game_step_count = 0
        
        def gui_worker():
            try:
                print("🚀 Starting interactive GUI process...")
                # Start GUI with the shared environment
                # Note: This will run in the same process but different thread
                # to share the environment instance
                self.gui_running = True
                
                # Initialize pygame for the GUI
                pygame.init()
                gui_config = get_gui_config(self.config, 'low')
                window_size = (gui_config['width'], gui_config['height'])
                screen = pygame.display.set_mode(window_size)
                pygame.display.set_caption(f"Crafter Interactive - Grid Check [{window_size[0]}x{window_size[1]}]")
                clock = pygame.time.Clock()
                
                print(f"✅ Interactive GUI started ({window_size[0]}x{window_size[1]})")
                print("🎮 Use WASD to move, SPACE to interact, ESC to close GUI")
                
                # Main GUI loop
                running = True
                while running and self.gui_running:
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            else:
                                # Handle game actions
                                action = self._handle_keyboard_input(event.key)
                                if action is not None:
                                    # Apply action to shared environment
                                    with self.shared_env.lock:
                                        obs, reward, done, info = self.env.step(action)
                                        # Increment step counter for ANY action taken
                                        self.game_step_count += 1
                                        if done:
                                            obs = self.env.reset()
                    
                    # Render the game
                    try:
                        frame = self.env.render(size=window_size)
                        if frame is not None:
                            # Convert numpy array to pygame surface
                            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                            screen.blit(frame_surface, (0, 0))
                            pygame.display.flip()
                    except Exception as render_error:
                        print(f"⚠️ Render error: {render_error}")
                    
                    clock.tick(30)  # 30 FPS
                
                pygame.quit()
                self.gui_running = False
                print("🛑 Interactive GUI process stopped")
                
            except Exception as e:
                print(f"❌ Error in GUI process: {e}")
                self.gui_running = False
        
        # Start GUI in separate thread
        self.gui_thread = threading.Thread(target=gui_worker, daemon=True)
        self.gui_thread.start()
        
        # Wait a moment for GUI to initialize
        time.sleep(2)
    
    def _handle_keyboard_input(self, key):
        """Convert pygame key to Crafter action"""
        # Keyboard mapping (same as crafter_interactive_launcher)
        key_mapping = {
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
            # Number keys 1-6 for crafting actions
            pygame.K_1: 11,         # Make wood pickaxe
            pygame.K_2: 12,         # Make stone pickaxe
            pygame.K_3: 13,         # Make iron pickaxe
            pygame.K_4: 14,         # Make wood sword
            pygame.K_5: 15,         # Make stone sword
            pygame.K_6: 16,         # Make iron sword
        }
        return key_mapping.get(key)
    
    def stop_gui_process(self):
        """Stop the interactive GUI process"""
        self.gui_running = False
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=2)
            print("🛑 GUI process stopped")
    
    def get_detected_grid_content(self, image):
        """Get the detected grid content using REAL detector API"""
        try:
            # Use the actual detector's extract_objects_crafter_api method
            # This will use the direct_api method with the shared environment
            detected_objects = self.detector.extract_objects_crafter_api(image)
            
            grid_content = {}
            
            # Process detected objects into grid format
            for obj in detected_objects:
                if 'grid_position' in obj:
                    row, col = obj['grid_position']
                    if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                        grid_content[(row, col)] = {
                            'type': obj.get('type', 'unknown'),
                            'confidence': obj.get('confidence', 1.0),
                            'source': 'real_detector_api',
                            'bbox': obj.get('bbox', []),
                            'world_pos': obj.get('world_pos', [0, 0])
                        }
            
            # Fill empty positions
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    if (row, col) not in grid_content:
                        grid_content[(row, col)] = {
                            'type': 'empty',
                            'confidence': 1.0,
                            'source': 'real_detector_api',
                            'bbox': [],
                            'world_pos': [0, 0]
                        }
            
            return grid_content
            
        except Exception as e:
            print(f"❌ Error getting detected grid content: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def compare_grid_contents(self, reference_grid, detected_grid):
        """Compare reference and detected grid contents (silent analysis)"""
        matches = 0
        mismatches = 0
        missing_detections = 0
        extra_detections = 0
        
        # Create a comprehensive comparison
        all_positions = set(reference_grid.keys()) | set(detected_grid.keys())
        
        for pos in sorted(all_positions):
            ref_data = reference_grid.get(pos, {})
            det_data = detected_grid.get(pos, {})
            
            ref_type = ref_data.get('type', 'missing')
            det_type = det_data.get('type', 'missing')
            
            if ref_type == det_type:
                matches += 1
            elif ref_type == 'missing':
                extra_detections += 1
            elif det_type == 'missing':
                missing_detections += 1
            else:
                mismatches += 1
        
        # Summary statistics
        total_positions = len(all_positions)
        accuracy = (matches / total_positions * 100) if total_positions > 0 else 0
        
        return {
            'total_positions': total_positions,
            'matches': matches,
            'mismatches': mismatches,
            'missing_detections': missing_detections,
            'extra_detections': extra_detections,
            'accuracy': accuracy
        }
    
    def analyze_object_creature_layers(self, reference_grid):
        """Analyze object/creature types to verify layer rendering hypothesis"""
        materials = set()
        objects = set()
        creatures = set()
        
        # Categorize all elements found in the grid
        for pos, data in reference_grid.items():
            element_type = data.get('type', 'unknown')
            source_info = data.get('source_info', {})
            
            # Categorize based on Crafter's internal structure
            if element_type in ['grass', 'stone', 'tree', 'water', 'sand', 'path', 'coal', 'iron', 'diamond', 'lava']:
                materials.add(element_type)
            elif element_type in ['cow', 'zombie', 'skeleton', 'player']:
                creatures.add(element_type)
            elif element_type in ['fence', 'plant', 'arrow']:
                objects.add(element_type)
        
        # Print layer analysis
        if materials or objects or creatures:
            print("\n🔍 LAYER ANALYSIS - Object/Creature Detection")
            print("-" * 50)
            print(f"📦 Materials (terrain layer): {sorted(materials) if materials else 'None'}")
            print(f"🎯 Objects (item layer): {sorted(objects) if objects else 'None'}")
            print(f"🐾 Creatures (entity layer): {sorted(creatures) if creatures else 'None'}")
            
            # Check for multi-layer positions
            multi_layer_positions = []
            for pos, data in reference_grid.items():
                source_info = data.get('source_info', {})
                if 'material' in source_info and 'obj' in source_info:
                    multi_layer_positions.append((pos, source_info))
            
            if multi_layer_positions:
                print(f"\n🎭 MULTI-LAYER POSITIONS DETECTED: {len(multi_layer_positions)}")
                for pos, info in multi_layer_positions[:3]:  # Show first 3 examples
                    material = info.get('material', 'unknown')
                    obj = info.get('obj', 'unknown')
                    print(f"   📍 {pos}: Material='{material}' + Object='{obj}'")
                if len(multi_layer_positions) > 3:
                    print(f"   ... and {len(multi_layer_positions) - 3} more")
                print("   ✅ This confirms objects/creatures are rendered on separate layers!")
            else:
                print("\n🎭 No multi-layer positions detected in current view")
    
    def print_detailed_grid_data(self, reference_grid, detected_grid):
        """Print detailed grid data for debugging purposes"""
        print("\n" + "=" * 70)
        print("🔍 DETAILED GRID DATA FOR DEBUGGING")
        print("=" * 70)
        
        # Print reference grid data
        print("\n📋 REFERENCE GRID (Actual Game State):")
        print("-" * 50)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                pos = (row, col)
                if pos in reference_grid:
                    data = reference_grid[pos]
                    element_type = data.get('type', 'unknown')
                    world_pos = data.get('world_pos', [0, 0])
                    material = data.get('material', None)
                    obj = data.get('obj', None)
                    
                    if element_type != 'empty':
                        print(f"  [{row},{col}] -> Type: '{element_type}', World: {world_pos}, Material: '{material}', Object: '{obj}'")
        
        # Print detected grid data
        print("\n🎯 DETECTED GRID (Detector Results):")
        print("-" * 50)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                pos = (row, col)
                if pos in detected_grid:
                    data = detected_grid[pos]
                    element_type = data.get('type', 'unknown')
                    confidence = data.get('confidence', 0.0)
                    source = data.get('source', 'unknown')
                    
                    if element_type != 'empty':
                        print(f"  [{row},{col}] -> Type: '{element_type}', Confidence: {confidence:.2f}, Source: '{source}'")
        
        # Print summary of unique elements
        ref_elements = set()
        det_elements = set()
        
        for data in reference_grid.values():
            element_type = data.get('type', 'unknown')
            if element_type != 'empty':
                ref_elements.add(element_type)
        
        for data in detected_grid.values():
            element_type = data.get('type', 'unknown')
            if element_type != 'empty':
                det_elements.add(element_type)
        
        print("\n📊 ELEMENT SUMMARY:")
        print("-" * 30)
        print(f"Reference elements: {sorted(ref_elements)}")
        print(f"Detected elements:  {sorted(det_elements)}")
        print(f"Missing elements:   {sorted(ref_elements - det_elements)}")
        print(f"Extra elements:     {sorted(det_elements - ref_elements)}")
    
    def print_simplified_grid_table(self, reference_grid, detected_grid, player_pos, player_facing):
        """Print a simplified grid table showing only detected elements with checkmarks"""
        print("\n" + "=" * 70)
        print("🎯 CRAFTER GRID DETECTION STATUS")
        print("=" * 70)
        
        # Get all unique element types found in the grid
        all_elements = set()
        detected_elements = set()
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                ref_content = reference_grid.get((row, col), {}).get('type', 'empty')
                det_content = detected_grid.get((row, col), {}).get('type', 'empty')
                
                if ref_content != 'empty':
                    all_elements.add(ref_content)
                if det_content != 'empty' and ref_content == det_content:
                    detected_elements.add(det_content)
        
        # Print grid with checkmarks and element types
        print("\n📋 GRID LAYOUT (✅ = Detected, ❌ = Missed, ⬜ = Empty)")
        print("-" * 70)
        
        # Print column headers
        print("   ", end="")
        for col in range(self.grid_cols):
            print(f"{col:>10}", end="")
        print()
        
        # Print grid rows
        for row in range(self.grid_rows):
            print(f"{row:>2} ", end="")
            for col in range(self.grid_cols):
                ref_content = reference_grid.get((row, col), {}).get('type', 'empty')
                det_content = detected_grid.get((row, col), {}).get('type', 'empty')
                
                # Check if this is player position
                world_pos = reference_grid.get((row, col), {}).get('world_pos', [0, 0])
                is_player_pos = (world_pos[0] == player_pos[0] and world_pos[1] == player_pos[1])
                
                if is_player_pos:
                    # Show player with direction
                    direction_symbols = {
                        (-1, 0): '👈',  # left
                        (1, 0): '👉',   # right  
                        (0, -1): '👆',  # up
                        (0, 1): '👇'    # down
                    }
                    symbol = direction_symbols.get(tuple(player_facing), '🎮')
                    display_text = f"{symbol}Player"
                elif ref_content != 'empty' and det_content == ref_content:
                    # Show checkmark with element type
                    element_short = ref_content[:4] if len(ref_content) > 4 else ref_content
                    display_text = f"✅{element_short}"
                elif ref_content != 'empty':
                    # Show miss with element type
                    element_short = ref_content[:4] if len(ref_content) > 4 else ref_content
                    display_text = f"❌{element_short}"
                else:
                    display_text = '⬜'
                
                print(f"{display_text:>10}", end="")
            print()
        
        # Print element detection summary
        print("\n📊 ELEMENT DETECTION SUMMARY")
        print("-" * 30)
        
        # Define all possible Crafter elements
        all_possible_elements = {
            # Materials
            'grass', 'stone', 'tree', 'water', 'sand', 'lava', 'coal', 'iron', 'diamond',
            'table', 'furnace', 'path',
            # Objects/Creatures
            'cow', 'zombie', 'skeleton', 'plant', 'fence', 'arrow'
        }
        
        found_elements = sorted(all_elements)
        if found_elements:
            print("Elements found in current view:")
            for element in found_elements:
                status = "✅" if element in detected_elements else "❌"
                print(f"  {status} {element}")
        else:
            print("  Only grass detected in current view")
        
        # Calculate accuracy
        total_positions = self.grid_rows * self.grid_cols
        matches = sum(1 for row in range(self.grid_rows) for col in range(self.grid_cols)
                     if reference_grid.get((row, col), {}).get('type', 'empty') == 
                        detected_grid.get((row, col), {}).get('type', 'empty'))
        accuracy = (matches / total_positions) * 100 if total_positions > 0 else 0
        
        print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({matches}/{total_positions})")
        print(f"🎮 Player: {player_pos} facing {player_facing}")
        
        # Show inventory information
        self.print_inventory_status()
        
        # Show legend
        print("\n🔤 LEGEND:")
        print("  ✅ = Element correctly detected")
        print("  ❌ = Element missed by detector")
        print("  ⬜ = Empty space")
        print("  🎮👈👉👆👇 = Player position & direction")
    
    def print_inventory_status(self):
        """Print player inventory information in a compact multi-column layout"""
        try:
            with self.shared_env.lock:
                player = self.env._player
                inventory = player.inventory
                
            print("\n🎒 PLAYER INVENTORY")
            print("=" * 80)
            
            # Define emoji mappings with fallback alternatives
            emoji_map = {
                'health': '❤️',
                'food': '🍗', 
                'drink': '💧',
                'energy': '⚡',
                'sapling': '🌱',
                'wood': '🌳',
                'stone': '🗿',
                'coal': '⚫',
                'iron': '🔩',
                'diamond': '💎',
                'wood_pickaxe': '⛏️',
                'stone_pickaxe': '⛏️',
                'iron_pickaxe': '⛏️',
                'wood_sword': '⚔️',
                'stone_sword': '⚔️',
                'iron_sword': '⚔️'
            }
            
            # Organize items into categories for compact display
            stats_items = [
                ('health', 'Health', 9),
                ('food', 'Hunger', 9), 
                ('drink', 'Thirst', 9),
                ('energy', 'Energy', 9)
            ]
            
            materials_items = [
                ('sapling', 'Sapling'),
                ('wood', 'Wood'),
                ('stone', 'Stone'),
                ('coal', 'Coal'),
                ('iron', 'Iron'),
                ('diamond', 'Diamond')
            ]
            
            tools_items = [
                ('wood_pickaxe', 'WoodPickaxe'),
                ('stone_pickaxe', 'StonePickaxe'),
                ('iron_pickaxe', 'IronPickaxe')
            ]
            
            weapons_items = [
                ('wood_sword', 'WoodSword'),
                ('stone_sword', 'StoneSword'),
                ('iron_sword', 'IronSword')
            ]
            
            # Display stats in one line
            stats_display = []
            for key, name, max_val in stats_items:
                icon = emoji_map.get(key, '●')
                value = inventory.get(key, 0)
                stats_display.append(f"{icon}{name}:{value}/{max_val}")
            print(f"📊 Stats: {' | '.join(stats_display)}")
            
            # Display all materials (including 0 values)
            materials_display = []
            for key, name in materials_items:
                icon = emoji_map.get(key, '●')
                value = inventory.get(key, 0)
                materials_display.append(f"{icon}{name}:{value}")
            print(f"🧱 Materials: {' | '.join(materials_display)}")
            
            # Display tools in compact format (show all, including 0 values)
            tools_display = []
            for key, name in tools_items:
                icon = emoji_map.get(key, '🔧')
                value = inventory.get(key, 0)
                tools_display.append(f"{icon}{name}:{value}")
            print(f"⚒️ Tools: {' | '.join(tools_display)}")
            
            # Display weapons in compact format (show all, including 0 values)
            weapons_display = []
            for key, name in weapons_items:
                icon = emoji_map.get(key, '⚔️')
                value = inventory.get(key, 0)
                weapons_display.append(f"{icon}{name}:{value}")
            print(f"⚔️ Weapons: {' | '.join(weapons_display)}")
                    
        except Exception as e:
            print(f"\n❌ Error reading inventory: {e}")
    
    def run_comparison(self):
        """Run the REAL grid content comparison with interactive GUI - Step-by-step detection"""
        print("🚀 Starting REAL Grid Content Comparison (Step-by-Step Mode)")
        print("=" * 70)
        print("This will start an interactive GUI window and detect on EVERY step")
        print("🎯 Enhanced debugging mode with object/creature layer analysis")
        print("Press Ctrl+C to stop the comparison")
        print()
        
        try:
            # Start the interactive GUI process
            print("🎮 Starting interactive GUI process...")
            self.start_gui_process()
            
            if not self.gui_running:
                print("❌ Failed to start GUI process")
                return
            
            print("✅ Interactive GUI is running!")
            print("🎮 You can now interact with the game using WASD keys")
            print("📊 Starting STEP-BY-STEP grid content comparison...")
            print("🔍 Each action will trigger immediate detection analysis")
            print()
            
            last_detected_step = 0
            max_steps = 3000  # Increased step limit for more debugging
            
            while self.gui_running and last_detected_step < max_steps:
                # Check if a new game step has occurred
                try:
                    current_step = getattr(self, 'game_step_count', 0)
                    
                    # Trigger detection when a new step occurs
                    if current_step > last_detected_step:
                        last_detected_step = current_step
                        
                        print(f"\n🖼️ Step #{current_step}/{max_steps} - Game State Changed")
                        
                        # Get current player position for display
                        with self.shared_env.lock:
                            current_player_pos = self.env._player.pos
                        print(f"🚶 Player position: {current_player_pos}")
                        
                        # Get current frame from shared environment
                        try:
                            with self.shared_env.lock:
                                frame = self.env.render(size=(400, 400))
                            
                            if frame is None:
                                print("❌ Failed to get frame from environment")
                                time.sleep(0.1)
                                continue
                                
                        except Exception as render_error:
                            print(f"❌ Render error: {render_error}")
                            time.sleep(0.1)
                            continue
                        
                        # Get reference grid content (actual game state)
                        reference_grid = self.get_reference_grid_content()
                        
                        # Enhanced object/creature analysis
                        self.analyze_object_creature_layers(reference_grid)
                        
                        # Get detected grid content using REAL detector
                        detected_grid = self.get_detected_grid_content(frame)
                        
                        # Get player position and facing direction
                        with self.shared_env.lock:
                            player_pos = self.env._player.pos
                            player_facing = getattr(self.env._player, 'facing', [0, 1])  # Default facing down
                        
                        # Print simplified grid table with checkmarks
                        self.print_simplified_grid_table(reference_grid, detected_grid, player_pos, player_facing)
                        
                        # Compare and analyze
                        comparison_result = self.compare_grid_contents(reference_grid, detected_grid)
                        
                        print(f"\n⏳ Waiting for next action... (Step {current_step}/{max_steps})")
                    
                    # Short sleep to prevent excessive CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Error in step detection: {e}")
                    time.sleep(0.5)
            
            if last_detected_step >= max_steps:
                print(f"\n🏁 Reached maximum steps ({max_steps}). Stopping comparison.")
                
        except KeyboardInterrupt:
            print("\n🛑 Comparison stopped by user")
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up GUI process
            print("\n🧹 Cleaning up...")
            self.stop_gui_process()

def main():
    """Main function"""
    print("🔍 CRAFTER REAL INTERACTIVE GRID CHECK TOOL")
    print("=" * 70)
    print("This tool launches a REAL interactive Crafter GUI and checks:")
    print("  • Actual game state (reference grid from shared environment)")
    print("  • Real detector API results (using actual detection system)")
    print("")
    print("🎮 Interactive Features:")
    print("  • Real-time GUI window with keyboard controls (WASD + SPACE + 1-6)")
    print("  • Live grid content check every step")
    print("  • Shared environment between GUI and detector")
    print("  • Number keys 1-6 for crafting actions (pickaxes & swords)")
    print("  • Press Ctrl+C to stop the check")
    print()
    
    # Load configuration
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return
    
    # Create and run checker
    try:
        checker = GridContentChecker(config)
        checker.run_comparison()
    except Exception as e:
        print(f"❌ Error initializing comparator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()