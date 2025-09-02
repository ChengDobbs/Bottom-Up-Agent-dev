#!/usr/bin/env python3
"""
Demo script for testing Gym environment configurations
This script demonstrates how to use the created gym config files
"""

import yaml
import numpy as np
from pathlib import Path
import gymnasium as gym
import crafter
import argparse
import time
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Optional pygame import for GUI display
try:
    import pygame
    from PIL import Image
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not available - Crafter GUI display disabled")

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_environment(config):
    """Create environment based on configuration"""
    # Try both 'gym' and 'gym_env' keys for compatibility
    gym_config = config.get('gym', config.get('gym_env', {}))
    env_type = gym_config.get('env_type', 'gymnasium')
    
    if env_type == 'crafter_direct':
        # Create Crafter environment directly
        render_mode = gym_config.get('render_mode', 'rgb_array')
        env = crafter.Env()
        
        # Store render mode for later use in testing
        env._demo_render_mode = render_mode
        print(f"✅ Created Crafter environment directly with render_mode: {render_mode}")
        
        if render_mode == 'human':
            print("⚠️  Note: Crafter GUI display requires pygame interaction loop.")
            print("    For full GUI experience, use: python -m crafter.run_gui")
    else:
        # Use gymnasium.make for standard environments
        env_id = gym_config.get('env_id')
        render_mode = gym_config.get('render_mode', 'rgb_array')
        max_episode_steps = gym_config.get('max_episode_steps')
        
        kwargs = {'render_mode': render_mode}
        if max_episode_steps:
            kwargs['max_episode_steps'] = max_episode_steps
            
        env = gym.make(env_id, **kwargs)
        print(f"✅ Created {env_id} environment via gymnasium")
    
    return env

def test_environment(env, config, num_steps=10, show_gui=False, resolution='high'):
    """Test environment with random actions"""
    game_name = config.get('game_name', 'Unknown')
    print(f"\n🎮 Testing {game_name} environment...")
    
    # Print environment info
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Get action names if available
    gym_config = config.get('gym', config.get('gym_env', {}))
    action_names = gym_config.get('action_space', {}).get('actions', [])
    if action_names:
        print(f"Available actions: {action_names}")
    
    # Check if this is Crafter with human render mode
    is_crafter_gui = hasattr(env, '_demo_render_mode') and env._demo_render_mode == 'human'
    if is_crafter_gui:
        print("🖼️  Crafter GUI mode detected - will render frames")
    
    # Initialize pygame for GUI display if requested and available
    pygame_screen = None
    pygame_clock = None
    window_size = None
    render_size = None
    if show_gui and is_crafter_gui and PYGAME_AVAILABLE:
        pygame.init()
        
        # Resolution mapping - support both presets and numeric values
        resolution_map = {
            'low': (400, 400),
            'medium': (600, 600), 
            'high': (800, 800),
            'ultra': (1200, 1200)
        }
        
        # Handle numeric resolution or preset
        if resolution.isdigit():
            size = int(resolution)
            window_size = (size, size)
        else:
            window_size = resolution_map.get(resolution, (800, 800))
        render_size = window_size  # Render at window size for best quality
        
        pygame_screen = pygame.display.set_mode(window_size)
        resolution_display = resolution.upper() if not resolution.isdigit() else f"{resolution}x{resolution}"
        pygame.display.set_caption(f"Crafter - {game_name} [{resolution_display}]")
        pygame_clock = pygame.time.Clock()
        print(f"🖼️  Pygame GUI window created ({window_size[0]}x{window_size[1]}) - {resolution_display} Resolution Mode")
    elif show_gui and is_crafter_gui and not PYGAME_AVAILABLE:
        print("⚠️  GUI requested but pygame not available")
    elif show_gui and not is_crafter_gui:
        print("⚠️  GUI only supported for Crafter environments")
    
    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new gymnasium format
    
    print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    # Render initial frame for Crafter GUI
    if is_crafter_gui:
        try:
            # Use high resolution for initial frame if GUI is active
            if pygame_screen is not None and render_size is not None:
                frame = env.render(render_size)
                print(f"  🖼️  Rendered initial frame (High-Res): {frame.shape}")
            else:
                frame = env.render()
                print(f"  🖼️  Rendered initial frame: {frame.shape}")
        except Exception as e:
            print(f"  ⚠️  Render warning: {e}")
    
    # Run test episode
    total_reward = 0
    running = True
    
    for step in range(num_steps):
        if not running:
            break
            
        # Handle pygame events if GUI is active
        if pygame_screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break
        
        if not running:
            break
        
        # Generate random action
        if hasattr(env.action_space, 'sample'):
            action = env.action_space.sample()
        else:
            # For Crafter's custom DiscreteSpace
            action = np.random.randint(0, env.action_space.n)
        
        # Take step
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        total_reward += reward
        
        # Render frame for Crafter GUI
        if is_crafter_gui:
            try:
                # Render at high resolution if GUI is active
                if pygame_screen is not None and render_size is not None:
                    frame = env.render(render_size)  # Render at target size
                else:
                    frame = env.render()  # Default rendering
                
                # Display in pygame window if available
                if pygame_screen is not None and frame is not None:
                    current_window_size = pygame_screen.get_size()
                    
                    # Only resize if necessary, using high-quality resampling
                    if frame.shape[:2] != current_window_size:
                        image = Image.fromarray(frame)
                        # Use LANCZOS for better quality when scaling
                        image = image.resize(current_window_size, resample=Image.LANCZOS)
                        frame = np.array(image)
                    
                    # Convert to pygame surface and display
                    surface = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
                    pygame_screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    
                    # Control frame rate - slightly faster for better experience
                    pygame_clock.tick(5)  # 5 FPS for demo (increased from 3)
                
                # Print frame info (less frequently to avoid spam)
                if frame is not None and step % 5 == 0:
                    print(f"  🖼️  Rendered frame {step + 1}: {frame.shape}")
                    
            except Exception as e:
                if step == 0:  # Only print render error once
                    print(f"  ⚠️  Render warning: {e}")
        
        # Print step info
        action_name = action_names[action] if action < len(action_names) else f"action_{action}"
        print(f"  Step {step + 1}: {action_name} -> reward={reward:.3f}, done={done}")
        
        if info and any(info.values() if isinstance(info, dict) else [info]):
            print(f"    Info: {info}")
        
        if done:
            print(f"  Episode ended at step {step + 1}")
            break
        
        # Small delay for better visualization
        if pygame_screen is not None:
            time.sleep(0.1)
    
    # Clean up pygame
    if pygame_screen is not None:
        pygame.quit()
        print("🖼️  Pygame GUI window closed")
    
    print(f"✅ Test completed: {step + 1} steps, total reward: {total_reward:.3f}")
    return total_reward

def demo_config(config_path, show_gui=False, resolution='high'):
    """Demo a single configuration file"""
    print(f"\n{'='*60}")
    print(f"TESTING CONFIG: {config_path.name}")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        config = load_config(config_path)
        print(f"✅ Loaded configuration: {config.get('project_name', 'Unknown')}")
        
        # Create environment
        env = create_environment(config)
        
        # Test environment
        total_reward = test_environment(env, config, num_steps=15, show_gui=show_gui, resolution=resolution)
        
        # Clean up (if method exists)
        if hasattr(env, 'close'):
            env.close()
        
        print(f"✅ {config_path.name} test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {config_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo Gym environment configurations')
    parser.add_argument('--gui', action='store_true', 
                       help='Show GUI window for Crafter environment (requires pygame)')
    parser.add_argument('--config', type=str, 
                       help='Test specific config file (e.g., crafter_config.yaml)')
    parser.add_argument('--resolution', type=str, default='800', 
                       help='GUI resolution: specify as number (e.g., 400, 600, 800, 1200) or preset (low, medium, high, ultra)')
    args = parser.parse_args()
    
    print("🎮 GYM ENVIRONMENT CONFIGURATIONS DEMO")
    print("=" * 60)
    
    if args.gui:
        if PYGAME_AVAILABLE:
            print("🖼️  GUI mode enabled - Crafter will show pygame window")
        else:
            print("⚠️  GUI mode requested but pygame not available")
            print("   Install pygame: pip install pygame")
    else:
        print("📊 Headless mode - no GUI windows (use --gui for Crafter GUI)")
    
    # Find all config files
    config_dir = Path('./config/gym')
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return
    
    if args.config:
        # Test specific config file
        config_file = config_dir / args.config
        if not config_file.exists():
            print(f"❌ Config file not found: {config_file}")
            return
        config_files = [config_file]
    else:
        # Test all config files
        config_files = list(config_dir.glob('*.yaml'))
        if not config_files:
            print(f"❌ No YAML config files found in {config_dir}")
            return
    
    print(f"Found {len(config_files)} configuration file(s):")
    for config_file in config_files:
        print(f"  - {config_file.name}")
    
    # Test each configuration
    passed = 0
    total = len(config_files)
    
    for config_file in sorted(config_files):
        if demo_config(config_file, show_gui=args.gui, resolution=args.resolution):
            passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DEMO RESULTS: {passed}/{total} configurations tested successfully")
    
    if passed == total:
        print("🎉 All gym configurations are working correctly!")
        print("\n📝 Next steps:")
        print("  1. Choose a configuration file for your agent")
        print("  2. Integrate with your Bottom-Up Agent framework")
        print("  3. Start training or testing your agent")
        print("\n💡 Usage tips:")
        print("  • python demo_gym_configs.py --gui (show Crafter GUI)")
        print("  • python demo_gym_configs.py --gui --resolution ultra (max quality)")
        print("  • python demo_gym_configs.py --config crafter_config.yaml")
        print("  • python -m crafter.run_gui (full interactive Crafter)")
        print("\n🖼️  Resolution options: low(400x400), medium(600x600), high(800x800), ultra(1200x1200)")
    else:
        print("⚠️  Some configurations failed. Check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()