#!/usr/bin/env python3
"""
Universal Bottom-Up Agent Runner
Supports both standard BottomUpAgent and GymBottomUpAgent with various modes
"""

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
            print(f"❌ Window '{window_name}' not found on {temp_eye.platform}, please launch the game before starting the run.")
            return False
            
        print(f"✅ Window '{window_name}' found, proceeding with agent initialization.")
        return True
        
    except Exception as e:
        print(f"❌ Error checking window: {e}")
        return False


def main(config, mode='demo', episodes=1, max_steps=1000, resolution='medium', analysis_interval=5, no_gui=False):
    """Main entry point for running Bottom-Up Agent"""
    
    # Check if this is a Gym environment configuration
    is_gym_env = (
        'gym' in config or 
        config.get('game_name') in ['Crafter', 'CrafterReward-v1'] or
        'gym_environment' in config
    )
    
    if is_gym_env:
        # Import GymBottomUpAgent only when needed (includes pygame dependencies)
        try:
            from BottomUpAgent.GymAgent import GymBottomUpAgent
        except ImportError as e:
            print(f"❌ Failed to import GymBottomUpAgent: {e}")
            print("💡 Make sure pygame and gym dependencies are installed for gym environments")
            sys.exit(1)
            
        # Check if simple mode is requested (to avoid initialization issues)
        simple_mode = config.get('simple_mode', False)
        if simple_mode:
            print(f"🎮 Creating GymBottomUpAgent in SIMPLE MODE for {config.get('game_name', 'Unknown')}")
            print("⚡ Simple mode bypasses: WandB Logger, Eye module, CLIP models, Detector")
        else:
            print(f"🎮 Creating GymBottomUpAgent for {config.get('game_name', 'Unknown')}")
        agent = GymBottomUpAgent(config, simple_mode=simple_mode)
        
        try:
            if mode == 'interactive':
                print("🚀 Interactive BottomUp Mode with GUI Control...")
                stats = agent.run_interactive(max_steps=max_steps)
                print(f"Interactive session completed: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
            elif mode == 'parallel_interactive':
                print("🚀 Parallel Interactive Mode (User + MCP Control)...")
                stats = agent.run_parallel_interactive(max_steps=max_steps)
                print(f"Parallel interactive session completed: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
            elif mode == 'crafter_launcher':
                print("🎮 Crafter Interactive Launcher Mode...")
                stats = agent.start_crafter_interactive_launcher(
                    max_steps=max_steps,
                    resolution=resolution,
                    no_gui=no_gui
                )
                if stats:
                    print(f"Crafter session completed: {stats}")
            elif mode == 'crafter_detection':
                print("🔍 Crafter Detection Analysis Mode...")
                stats = agent.start_crafter_interactive_with_detection(max_steps=max_steps)
                if stats:
                    print(f"Detection session completed: accuracy {stats['detection_accuracy']:.1f}%, {stats['steps']} steps")
            elif mode == 'hybrid':
                print("🤝 Hybrid Mode (User Control + Agent Suggestions)...")
                stats = agent.start_hybrid_crafter_mode(
                    max_steps=max_steps,
                    analysis_interval=analysis_interval
                )
                if stats:
                    print(f"Hybrid session completed: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
            else:  # demo mode
                print(f"🎯 Demo mode: {episodes} episodes")
                for i in range(episodes):
                    episode_stats = agent.run_episode()
                    print(f"Episode {i+1}: {episode_stats['steps']} steps, {episode_stats['total_reward']:.2f} reward")
                    
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            agent.close()
    else:
        print(f"🤖 Creating standard BottomUpAgent for {config.get('game_name', 'Unknown')}")
        
        # Check if window exists before creating agent
        if not check_window_exists(config):
            print("❌ Cannot proceed without game window. Please launch the game first.")
            sys.exit(1)
            
        agent = BottomUpAgent(config)
        task = config.get('task', 'Play the game')
        
        try:
            # Standard BottomUpAgent run method
            agent.run(task=task, max_step=max_steps)
        except Exception as e:
            print(f"❌ Error during standard agent execution: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universal Bottom-Up Agent Runner')
    
    # Config file argument
    parser.add_argument('--config', required=True, help='Path to config file')
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['demo', 'interactive', 'parallel_interactive', 'crafter_launcher', 'crafter_detection', 'hybrid'], 
                       default='demo', 
                       help='Run mode')
    
    # General options
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    
    # Crafter-specific options
    parser.add_argument('--resolution', default='medium', 
                       help='GUI resolution for Crafter modes (tiny/small/low/medium/high/ultra)')
    parser.add_argument('--analysis_interval', type=int, default=5, 
                       help='Steps between Agent analysis in hybrid mode')
    parser.add_argument('--no_gui', action='store_true', 
                       help='Run without GUI (background mode)')
    
    args = parser.parse_args()
    config_path = args.config
    
    # Load configuration
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Store the config file path for use by Brain class
        config['_config_path'] = config_path
        print(f"✅ Configuration loaded from {config_path}")
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)
    
    # Use mode from config file if not explicitly provided via command line
    if args.mode == 'demo' and 'mode' in config:
        args.mode = config['mode']
        print(f"📋 Using mode from config: {args.mode}")
    
    # Run the agent
    main(
        config=config,
        mode=args.mode,
        episodes=args.episodes,
        max_steps=args.max_steps,
        resolution=args.resolution,
        analysis_interval=args.analysis_interval,
        no_gui=args.no_gui
    )