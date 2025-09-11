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
from BottomUpAgent.GymAgent import GymBottomUpAgent


def main(config, mode='demo', interactive=False, episodes=1, max_steps=1000, resolution='medium', analysis_interval=5, no_gui=False):
    """Main entry point for running Bottom-Up Agent"""
    
    # Check if this is a Gym environment configuration
    is_gym_env = (
        'gym' in config or 
        config.get('game_name') in ['Crafter', 'CrafterReward-v1', 'CartPole-v1'] or
        'gym_environment' in config
    )
    
    if is_gym_env:
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
    
    # Config file argument (support both --config and --config_file for backward compatibility)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--config', help='Path to config file')
    config_group.add_argument('--config_file', help='Path to config file (deprecated, use --config)')
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['demo', 'interactive', 'parallel_interactive', 'crafter_launcher', 'crafter_detection', 'hybrid'], 
                       default='demo', 
                       help='Run mode')
    
    # General options
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (deprecated, use --mode interactive)')
    
    # Crafter-specific options
    parser.add_argument('--resolution', default='medium', 
                       help='GUI resolution for Crafter modes (tiny/small/low/medium/high/ultra)')
    parser.add_argument('--analysis_interval', type=int, default=5, 
                       help='Steps between Agent analysis in hybrid mode')
    parser.add_argument('--no_gui', action='store_true', 
                       help='Run without GUI (background mode)')
    
    args = parser.parse_args()
    
    # Determine config file path
    config_path = args.config or args.config_file
    
    # Load configuration
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)
    
    # Use mode from config file if not explicitly provided via command line
    if args.mode == 'demo' and 'mode' in config:
        args.mode = config['mode']
        print(f"📋 Using mode from config: {args.mode}")
    
    # Handle deprecated --interactive flag
    if args.interactive and args.mode == 'demo':
        args.mode = 'interactive'
        print("⚠️ --interactive flag is deprecated, use --mode interactive instead")
    
    # Run the agent
    main(
        config=config,
        mode=args.mode,
        interactive=args.interactive,
        episodes=args.episodes,
        max_steps=args.max_steps,
        resolution=args.resolution,
        analysis_interval=args.analysis_interval,
        no_gui=args.no_gui
    )