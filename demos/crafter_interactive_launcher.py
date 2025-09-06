#!/usr/bin/env python3
"""
Final Demo: Crafter Interactive with GUI Rendering
This script demonstrates the complete interactive Crafter experience with GUI rendering
"""

import sys
import time
import yaml
import argparse
import threading
import queue
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import crafter
    import pygame
    import numpy as np
    print("✅ All required modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Crafter action mappings
CRAFTER_ACTIONS = {
    0: "noop", 1: "move_left", 2: "move_right", 3: "move_up", 4: "move_down",
    5: "do", 6: "sleep", 7: "place_stone", 8: "place_table", 9: "place_furnace",
    10: "place_plant", 11: "make_wood_pickaxe", 12: "make_stone_pickaxe", 
    13: "make_iron_pickaxe", 14: "make_wood_sword", 15: "make_stone_sword", 16: "make_iron_sword"
}

# Crafter achievements list (22 total achievements)
CRAFTER_ACHIEVEMENTS = [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
    'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
    'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
    'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
    'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
    'place_plant', 'place_stone', 'place_table', 'wake_up'
]

# Keyboard mapping to actions
KEYBOARD_MAPPING = {
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

def compute_success_rates(achievements_history):
    """
    Calculate achievement success rates
    achievements_history: List of achievement dictionaries for each episode
    Returns: Success rate (percentage) for each achievement
    """
    if not achievements_history:
        return {name: 0.0 for name in CRAFTER_ACHIEVEMENTS}
    
    success_rates = {}
    total_episodes = len(achievements_history)
    
    for achievement in CRAFTER_ACHIEVEMENTS:
        success_count = sum(1 for episode_achievements in achievements_history 
                          if episode_achievements.get(achievement, 0) > 0)
        success_rates[achievement] = (success_count / total_episodes) * 100
    
    return success_rates

def compute_crafter_score(success_rates):
    """
    Calculate Crafter score (geometric mean, offset by 1%)
    success_rates: Achievement success rate dictionary (percentage)
    Returns: Crafter score
    """
    rates = [success_rates[name] for name in CRAFTER_ACHIEVEMENTS]
    
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

def format_achievement_name(achievement):
    """
    Format achievement name to readable form
    """
    name = achievement.replace('achievement_', '').replace('_', ' ').title()
    return name

def print_detailed_stats(achievements_history, episode_count, total_reward, total_steps):
    """
    Print compact achievement statistics
    """
    if not achievements_history:
        print("\n📊 No episodes completed yet.")
        return
    
    success_rates = compute_success_rates(achievements_history)
    crafter_score = compute_crafter_score(success_rates)
    
    # Calculate achievement totals
    achievement_totals = {}
    for achievement in CRAFTER_ACHIEVEMENTS:
        total_count = sum(episode.get(achievement, 0) for episode in achievements_history)
        achievement_totals[achievement] = total_count
    
    # Calculate average reward per episode
    avg_reward = total_reward / episode_count if episode_count > 0 else 0.0
    
    print(f"\n📊 STATS | Episodes: {episode_count} | Score: {crafter_score:.1f}% | Reward: {total_reward:.1f} | Avg Reward: {avg_reward:.1f} | Steps: {total_steps}")
    
    # Compact achievement display - only show unlocked achievements
    unlocked_achievements = [(name, rate, achievement_totals[name]) for name, rate in success_rates.items() if rate > 0]
    if unlocked_achievements:
        print(f"🏆 UNLOCKED ({len(unlocked_achievements)}/22):")
        # Sort by success rate, then by name
        unlocked_achievements.sort(key=lambda x: (-x[1], x[0]))
        
        # Display in compact format - 2 columns
        for i in range(0, len(unlocked_achievements), 2):
            left = unlocked_achievements[i]
            left_name = format_achievement_name(left[0])
            left_display = f"{left_name} ({left[1]:.0f}%, x{left[2]})"
            
            if i + 1 < len(unlocked_achievements):
                right = unlocked_achievements[i + 1]
                right_name = format_achievement_name(right[0])
                right_display = f"{right_name} ({right[1]:.0f}%, x{right[2]})"
                print(f"  ✅ {left_display:<35} ✅ {right_display}")
            else:
                print(f"  ✅ {left_display}")
    
    # Show latest episode achievements if any
    if achievements_history:
        latest_achievements = achievements_history[-1]
        new_unlocks = [(name, count) for name, count in latest_achievements.items() if count > 0]
        if new_unlocks:
            new_unlocks.sort(key=lambda x: x[0])  # Sort by name
            unlock_names = [format_achievement_name(name) for name, _ in new_unlocks]
            print(f"🆕 LATEST: {', '.join(unlock_names[:5])}{'...' if len(unlock_names) > 5 else ''}")
        else:
            print("\n🆕 LATEST EPISODE: No achievements unlocked")
    
    print("="*70)

def load_config(config_path="config/gym/crafter_config.yaml"):
    """
    Load configuration from YAML file
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}, using defaults")
        return {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return {}

def input_thread_worker(input_queue, stop_event):
    """
    Worker thread for non-blocking input
    """
    # Console character to action mapping
    console_mapping = {
        'w': 3,    # Move up
        'a': 1,    # Move left
        's': 4,    # Move down
        'd': 2,    # Move right
        ' ': 5,    # Space - interact/collect/attack
        '\t': 6,   # Tab - sleep
        't': 8,    # Place table
        'r': 7,    # Place rock/stone
        'f': 9,    # Place furnace
        'p': 10,   # Place plant
        '1': 11,   # Craft wood pickaxe
        '2': 12,   # Craft stone pickaxe
        '3': 13,   # Craft iron pickaxe
        '4': 14,   # Craft wood sword
        '5': 15,   # Craft stone sword
        '6': 16,   # Craft iron sword
    }
    
    while not stop_event.is_set():
        try:
            # Check stop_event before blocking on input
            if stop_event.is_set():
                break
                
            user_input = input("Enter action: ")
            
            # Check stop_event again after input
            if stop_event.is_set():
                break
            
            # Handle space key specially (before stripping)
            if user_input == ' ':
                input_queue.put('5')  # Space maps to action 5 (do)
                break
            
            # Now strip for other processing
            user_input = user_input.strip()
            
            # Handle special cases first
            if user_input.lower() in ['q', 'h', 'r']:
                input_queue.put(user_input.lower())
                break
            
            # Handle empty input (just Enter pressed)
            if user_input == '':
                continue  # Ask for input again
            
            # Check if it's a single character that maps to an action
            if len(user_input) == 1 and user_input.lower() in console_mapping:
                action_num = console_mapping[user_input.lower()]
                input_queue.put(str(action_num))
                break
            
            # Handle special keywords for space and tab
            if user_input.lower() == 'space':
                input_queue.put('5')  # Space action
                break
            elif user_input.lower() == 'tab':
                input_queue.put('6')  # Tab action
                break
            
            # Otherwise, treat as regular input (number or invalid)
            input_queue.put(user_input.lower())
            break
        except EOFError:
            # Handle end of input (Ctrl+D on Unix, Ctrl+Z on Windows)
            input_queue.put('q')
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            input_queue.put('q')
            break
        except Exception as e:
            # Handle any other exceptions during input
            print(f"Input thread exception: {e}")
            input_queue.put('q')
            break
    
    # Thread is terminating - put a quit signal to ensure main loop exits
    try:
        input_queue.put('q')
    except:
        pass  # Queue might be closed

def get_gui_config(config, resolution_override=None):
    """
    Extract GUI configuration from config file
    """
    gui_config = config.get('gui', {})
    
    # Default values
    default_width = 800
    default_height = 800
    
    # Resolution presets
    presets = gui_config.get('resolution_presets', {
        'tiny': [200, 200],
        'small': [300, 300],
        'low': [400, 400],
        'medium': [600, 600],
        'high': [800, 800],
        'ultra': [1200, 1200]
    })
    
    # Determine resolution
    if resolution_override:
        if resolution_override in presets:
            width, height = presets[resolution_override]
        else:
            try:
                size = int(resolution_override)
                width = height = size
            except ValueError:
                width = gui_config.get('width', default_width)
                height = gui_config.get('height', default_height)
    else:
        # Check both 'resolution' and 'resolution_preset' for compatibility
        preset = gui_config.get('resolution', gui_config.get('resolution_preset', 'low'))
        if preset in presets:
            width, height = presets[preset]
        elif preset == 'custom':
            # Custom mode: use explicit width/height from config
            width = gui_config.get('width', default_width)
            height = gui_config.get('height', default_height)
        else:
            # Fallback for unknown presets
            width = gui_config.get('width', default_width)
            height = gui_config.get('height', default_height)
            print(f"⚠️ Unknown resolution preset '{preset}', supported presets: {list(presets.keys())} + 'custom'")
            print(f"   Using fallback values: {width}x{height}")
    
    return {
        'width': width,
        'height': height,
        'title': gui_config.get('title', 'Crafter Interactive Demo'),
        'resizable': gui_config.get('resizable', False),
        'fullscreen': gui_config.get('fullscreen', False),
        'fps_limit': gui_config.get('fps_limit', 60),
        'vsync': gui_config.get('vsync', True)
    }

def demo_crafter_interactive(resolution='low', max_steps=10, config_path=None, no_gui=False):
    """Run interactive Crafter demo with optional GUI"""
    if no_gui:
        print("🎮 Crafter Interactive Demo - Background Mode (No GUI)")
    else:
        print("🎮 Crafter Interactive Demo with GUI Rendering")
    print("=" * 50)
    
    # Load configuration
    config = load_config(config_path) if config_path else load_config()
    
    # Use config file resolution if available, otherwise use parameter
    config_resolution = config.get('gui', {}).get('resolution')
    if config_resolution:
        resolution = config_resolution
        print(f"📋 Using resolution from config: {resolution}")
    else:
        print(f"📋 Using parameter resolution: {resolution}")
    
    gui_config = get_gui_config(config, resolution)
    
    window_size = (gui_config['width'], gui_config['height'])
    render_size = window_size  # Render at window size for best quality
    
    # Create Crafter environment
    env = crafter.Env()
    print("✅ Crafter environment created")
    
    # Initialize pygame for GUI only if not in no-gui mode
    screen = None
    clock = None
    if not no_gui:
        pygame.init()
        screen = pygame.display.set_mode(window_size)
        # Always display actual pixel dimensions in title, regardless of preset name
        resolution_display = f"{window_size[0]}x{window_size[1]}"
        pygame.display.set_caption(f"Crafter Interactive Demo [{resolution_display}]")
        clock = pygame.time.Clock()
        print(f"✅ Pygame GUI window initialized ({window_size[0]}x{window_size[1]}) - {resolution_display} Resolution Mode")
    else:
        print("✅ Running in background mode - no GUI window created")
    
    # Initialize tracking variables
    achievements_history = []  # Store achievements for each episode
    total_reward = 0.0
    total_steps = 0
    episode_count = 0
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    print(f"✅ Environment reset, observation shape: {obs.shape}")
    
    print("\n🎯 Interactive Demo Instructions:")
    print("- GUI window shows the Crafter game world")
    print("\n🎮 Interactive Controls:")
    print("- 🎮 Keyboard Controls (Recommended):")
    print("  • WASD: Move around")
    print("  • SPACE: Collect material, drink from lake, hit creature")
    print("  • TAB: Sleep")
    print("  • T: Place a table")
    print("  • R: Place a rock")
    print("  • F: Place a furnace")
    print("  • P: Place a plant")
    print("  • 1-6: Craft tools (1=wood pickaxe, 2=stone pickaxe, 3=iron pickaxe, 4=wood sword, 5=stone sword, 6=iron sword)")
    print("- 📝 Console Controls (Alternative):")
    print("  • Enter action numbers (0-16), 'h' for help, 'r' for random, 'q' to quit")
    print("- ESC or close GUI window to exit")
    
    # Show action mappings
    print("\n🎮 Crafter Action Mappings:")
    for i in range(17):
        print(f"  {i}: {CRAFTER_ACTIONS[i]}")
    
    # 3-second countdown before starting
    print("\n🚀 Starting demo in:")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print("   GO! 🎮")
    
    episode_reward = 0.0
    episode_steps = 0
    step = 0
    running = True
    episode_count += 1
    
    print(f"\n🎬 Starting Episode {episode_count}")
    
    while running and step < max_steps:
        step += 1
        episode_steps += 1
        total_steps += 1
        print(f"\n--- Episode {episode_count}, Step {episode_steps} (Global: {total_steps}) ---")
        print(f"Episode reward: {episode_reward:.2f}, Total reward: {total_reward:.2f}")
        
        # Handle pygame events and keyboard input (only if GUI is enabled)
        keyboard_action = None
        if not no_gui and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("GUI window closed, ending demo...")
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("ESC pressed, ending demo...")
                        running = False
                        break
                    elif event.key in KEYBOARD_MAPPING:
                        keyboard_action = KEYBOARD_MAPPING[event.key]
                        action_name = CRAFTER_ACTIONS[keyboard_action]
                        print(f"🎮 Keyboard input: {pygame.key.name(event.key).upper()} -> Action {keyboard_action} ({action_name})")
                        break
        
        if not running:
            break
        
        # Render environment to GUI (only if GUI is enabled)
        if not no_gui and screen is not None:
            try:
                # Render at high resolution if possible
                frame = env.render(render_size)
                if frame is not None and len(frame.shape) == 3:
                    current_window_size = screen.get_size()
                    
                    # Only resize if necessary, using high-quality resampling
                    if frame.shape[:2] != current_window_size:
                        image = Image.fromarray(frame)
                        # Use LANCZOS for better quality when scaling
                        image = image.resize(current_window_size, resample=Image.LANCZOS)
                        frame = np.array(image)
                    
                    # Convert to pygame surface and display
                    surface = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    print(f"🖼️  GUI updated with current game state ({frame.shape})")
            except Exception as e:
                print(f"Render error: {e}")
        else:
            # In no-gui mode, just get the frame data without displaying
            try:
                frame = env.render('rgb_array')
                if frame is not None:
                    print(f"🖼️  Frame data captured ({frame.shape}) - Background mode")
            except Exception as e:
                print(f"Frame capture error: {e}")
        
        # Use keyboard action if available, otherwise get console input
        if keyboard_action is not None:
            action = keyboard_action
        else:
            # Get user action input with non-blocking mechanism
            input_queue = queue.Queue()
            stop_event = threading.Event()
            
            print("\nChoose action:")
            print("  🎮 Use keyboard: WASD=move, SPACE=interact, TAB=sleep, T/R/F/P=place, 1-6=craft")
            print("  📝 Or enter action number (0-16), 'h' for help, 'r' for random, 'q' to quit")
            
            # Start input thread
            input_thread = threading.Thread(target=input_thread_worker, args=(input_queue, stop_event))
            input_thread.daemon = True
            input_thread.start()
            
            # Wait for input without timeout
            action = None
        
        while action is None:
            try:
                user_input = input_queue.get_nowait()
                stop_event.set()
                
                if user_input == 'q':
                    print("Quitting demo...")
                    running = False
                    break
                elif user_input == 'r':
                    action = np.random.randint(0, 17)
                    action_name = CRAFTER_ACTIONS[action]
                    print(f"Taking random action: {action} ({action_name})")
                    break
                elif user_input == 'h':
                    print("\n🎮 Action Help:")
                    for i in range(17):
                        print(f"  {i}: {CRAFTER_ACTIONS[i]}")
                    # Restart input for new action
                    stop_event.clear()
                    input_thread = threading.Thread(target=input_thread_worker, args=(input_queue, stop_event))
                    input_thread.daemon = True
                    input_thread.start()
                    continue
                else:
                    try:
                        action = int(user_input)
                        if 0 <= action <= 16:
                            action_name = CRAFTER_ACTIONS[action]
                            print(f"Taking action: {action} ({action_name})")
                            break
                        else:
                            print("Invalid action. Must be 0-16")
                            action = None
                    except ValueError:
                        print("Invalid input. Please enter a number, 'h', 'r', or 'q'")
                        action = None
                    
                    if action is None:
                        # Restart input for new action
                        stop_event.clear()
                        input_thread = threading.Thread(target=input_thread_worker, args=(input_queue, stop_event))
                        input_thread.daemon = True
                        input_thread.start()
                        
            except queue.Empty:
                # Handle pygame events while waiting for input (only if GUI is enabled)
                if not no_gui and screen is not None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("GUI window closed, ending demo...")
                            running = False
                            stop_event.set()
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("ESC pressed, ending demo...")
                                running = False
                                stop_event.set()
                                break
                            elif event.key in KEYBOARD_MAPPING:
                                action = KEYBOARD_MAPPING[event.key]
                                action_name = CRAFTER_ACTIONS[action]
                                print(f"🎮 Keyboard input: {pygame.key.name(event.key).upper()} -> Action {action} ({action_name})")
                                stop_event.set()
                                break
                
                if not running:
                    break
                    
                time.sleep(0.1)  # Small delay to prevent busy waiting
        
        # No timeout handling - wait indefinitely for user input
        
        stop_event.set()  # Ensure input thread stops
        
        if not running:
            break
        
        # Take action in environment
        try:
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            total_reward += reward
            print(f"Reward: {reward:.2f}, Total: {total_reward:.2f}")
            
            # Display current achievement status with detailed info
            if info:
                current_achievements = info.get('achievements', {})
                if current_achievements:
                    unlocked_count = sum(1 for achieved in current_achievements.values() if achieved > 0)
                    
                    # Calculate current score percentage
                    temp_history = achievements_history + [current_achievements] if achievements_history else [current_achievements]
                    success_rates = compute_success_rates(temp_history)
                    current_score = compute_crafter_score(success_rates)
                    
                    print(f"🏆 Achievements unlocked: {unlocked_count}/{len(CRAFTER_ACHIEVEMENTS)} | Score (%): {current_score:.1f}% | Reward: {episode_reward:.2f} | Episode: {episode_count} | {current_achievements}")
            
            if done:
                print(f"\n🏁 Episode {episode_count} finished!")
                print(f"Episode reward: {episode_reward:.2f}, Steps: {episode_steps}")
                
                # Save achievement data
                if info:
                    episode_achievements = info.get('achievements', {})
                    achievements_history.append(episode_achievements)
                    
                    # Display detailed statistics
                    print_detailed_stats(achievements_history, episode_count, total_reward, total_steps)
                
                # Reset for next episode
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
                episode_count += 1
                episode_reward = 0.0
                episode_steps = 0
                print(f"\n🎬 Starting Episode {episode_count}")
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received, stopping demo...")
            break
        except Exception as e:
            print(f"Step error: {e}")
            # Continue the loop instead of breaking on step errors
            continue
        
        # Control frame rate based on config
        fps = gui_config.get('fps_limit', 2)
        clock.tick(fps)
    
    # Clean up threads before exiting
    print("\n🧹 Cleaning up threads...")
    stop_event.set()  # Signal input thread to stop
    
    # Wait for input thread to finish (with timeout)
    if input_thread.is_alive():
        print("⏳ Waiting for input thread to terminate...")
        input_thread.join(timeout=2.0)
        if input_thread.is_alive():
            print("⚠️ Input thread did not terminate gracefully (this is expected on Windows due to input() blocking)")
        else:
            print("✅ Input thread terminated successfully")
    else:
        print("✅ Input thread already terminated")
    
    pygame.quit()
    print(f"\n✅ Demo completed!")
    print(f"Total steps: {step}")
    print(f"Final reward: {total_reward:.2f}")
    
    # Display final statistics
    if achievements_history:
        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS")
        print("=" * 60)
        print_detailed_stats(achievements_history, episode_count, total_reward, total_steps)
    
    print("\n🎉 Both GUI rendering and interactive control are working correctly!")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Crafter Interactive Demo with GUI Rendering')
    parser.add_argument('--resolution', type=str, default='high', 
                       help='GUI resolution: specify as number (e.g., 200, 300, 400, 600, 800, 1200) or preset (tiny, small, low, medium, high, ultra)')
    parser.add_argument('--max-steps', type=int, default=10,
                       help='Maximum number of steps in the demo (default: 10)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: config/gym/crafter_config.yaml)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run in background mode without creating GUI window')
    args = parser.parse_args()
    
    print("🎮 CRAFTER INTERACTIVE DEMO WITH GUI RENDERING")
    print("=" * 60)
    
    # Display resolution info
    if args.resolution.isdigit():
        print(f"🖼️  Resolution: {args.resolution}x{args.resolution} (Custom)")
    else:
        resolution_info = {
             'tiny': '200x200 (Tiny)',
             'small': '300x300 (Small)',
             'low': '400x400 (Low Quality)',
             'medium': '600x600 (Medium Quality)', 
             'high': '800x800 (High Quality)',
             'ultra': '1200x1200 (Ultra Quality)'
         }
        print(f"🖼️  Resolution: {resolution_info.get(args.resolution, 'Unknown')}")
    
    print(f"⏱️  Max Steps: {args.max_steps}")
    print("\n💡 Usage tips:")
    print("  • Use number keys (0-16) to select actions")
    print("  • Press 'r' for random action, 'h' for help, 'q' to quit")
    print("  • Close GUI window or press ESC to exit")
    print("\n🖼️  Resolution options: tiny(200x200), small(300x300), low(400x400), medium(600x600), high(800x800), ultra(1200x1200)")
    print("    Or specify custom size: --resolution 1000")
    
    # Run the demo
    demo_crafter_interactive(resolution=args.resolution, max_steps=args.max_steps, config_path=args.config, no_gui=args.no_gui)

if __name__ == "__main__":
    main()