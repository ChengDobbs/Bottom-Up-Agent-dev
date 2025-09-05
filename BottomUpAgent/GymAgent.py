#!/usr/bin/env python3
"""
Gym Agent - Universal Bottom-Up Agent for Gym Environments
Integrates GymAdapter with the Bottom-Up Agent framework
"""

import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from BottomUpAgent.BottomUpAgent import BottomUpAgent
from BottomUpAgent.GymAdapter import GymEnvironmentAdapter, GymAdapterFactory
from BottomUpAgent.Eye import Eye
from BottomUpAgent.Hand import Hand
from BottomUpAgent.Detector import Detector
from BottomUpAgent.Brain import Brain

class GymBottomUpAgent(BottomUpAgent):
    """
    Bottom-Up Agent specialized for Gym environments.
    Extends BottomUpAgent to work with any Gym environment through GymAdapter.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize gym adapter with unified config structure
        self.gym_config = config.get('gym', {})
        
        # Determine environment name from multiple possible sources
        self.env_name = (
            self.gym_config.get('env_id') or 
            config.get('game_name') or 
            'CartPole-v1'
        )
        
        # Prepare complete config for gym adapter
        gym_adapter_config = self.gym_config.copy()
        gym_adapter_config['env_name'] = self.env_name
        gym_adapter_config['game_name'] = config.get('game_name', self.env_name)
        
        # Create gym adapter
        self.gym_adapter = GymAdapterFactory.create_adapter(
            self.env_name, 
            gym_adapter_config
        )
        
        # Adapt config for Bottom-Up Agent framework
        adapted_config = self._adapt_config_for_gym(config)
        
        # Initialize parent BottomUpAgent
        super().__init__(adapted_config)
        
        # Override components for Gym integration
        self._setup_gym_components()
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.best_reward = float('-inf')
        
        print(f"GymBottomUpAgent initialized for {self.env_name}")
    
    def _adapt_config_for_gym(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt configuration for Gym environment"""
        gym_config = config.copy()
        
        # Set game name to environment name
        gym_config['game_name'] = self.env_name
        
        # Adjust eye configuration to match gym window
        if 'eye' not in gym_config:
            gym_config['eye'] = {}
        
        window_size = self.gym_config.get('window_size', [600, 400])
        gym_config['eye']['width'] = window_size[0]
        gym_config['eye']['height'] = window_size[1]
        
        # Set up exploration parameters for RL
        if 'exploration' not in gym_config:
            gym_config['exploration'] = {}
        
        gym_config['exploration'].update({
            'method': config.get('training', {}).get('exploration', {}).get('method', 'epsilon_greedy'),
            'intrinsic_weight': config.get('training', {}).get('reward', {}).get('intrinsic_weight', 0.1),
            'extrinsic_weight': config.get('training', {}).get('reward', {}).get('extrinsic_weight', 1.0)
        })
        
        return gym_config
    
    def _setup_gym_components(self):
        """Setup components specifically for Gym environment interaction"""
        # Override eye to work with gym adapter
        self.eye = GymEye(self.config, self.gym_adapter)
        
        # Override hand to work with gym actions
        self.hand = GymHand(self.config, self.gym_adapter)
        
        # Keep existing detector and brain
        # They should work with the adapted observations
    
    def reset_environment(self) -> Dict[str, Any]:
        """Reset the Gym environment"""
        obs = self.gym_adapter.reset()
        self.current_episode += 1
        
        print(f"\nEpisode {self.current_episode} started")
        return obs
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from Gym environment"""
        # Use the current observation from gym adapter
        if hasattr(self.gym_adapter, 'current_obs') and self.gym_adapter.current_obs is not None:
            return self.gym_adapter._process_observation(self.gym_adapter.current_obs)
        else:
            # Fallback: reset if no current observation
            return self.reset_environment()
    
    def execute_operation(self, operation: Dict[str, Any]) -> str:
        """Execute operation in Gym environment"""
        try:
            # Map operation to gym action
            action = self.gym_adapter.map_operation_to_action(operation)
            
            # Execute action in environment
            obs, reward, done, info = self.gym_adapter.step(action)
            
            # Update episode tracking
            episode_steps = info.get('episode_steps', 0)
            episode_reward = info.get('episode_reward', 0.0)
            
            # Check if episode is done
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_steps.append(episode_steps)
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                
                print(f"Episode {self.current_episode} finished:")
                print(f"  Steps: {episode_steps}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Best reward: {self.best_reward:.2f}")
                
                return 'Done'
            
            return 'Success'
            
        except Exception as e:
            print(f"Error executing operation: {e}")
            return 'Failed'
    
    def run_episode(self, task="Complete the game", max_steps: int = None, render=True, interactive_mode=False) -> Dict[str, Any]:
        """Run a single episode"""
        if max_steps is None:
            max_steps = self.gym_config.get('max_episode_steps', 500)
        
        # Reset environment
        obs = self.reset_environment()
        
        # Start GUI in appropriate mode
        if render:
            self.gym_adapter.start_gui(interactive_mode=interactive_mode)
        
        episode_stats = {
            'episode': self.current_episode,
            'steps': 0,
            'total_reward': 0.0,
            'actions': [],
            'rewards': []
        }
        
        print(f"Starting episode with task: {task}")
        if interactive_mode:
            print("Running in INTERACTIVE mode - waiting for user decisions")
        
        # Run episode
        for step in range(max_steps):
            # Get current state
            state = self.get_observation()
            
            # Get potential actions (objects in the environment)
            objects = self.gym_adapter.detect_objects(state['screen'])
            
            # Select action based on mode
            if interactive_mode:
                # Use interactive decision making
                decision_result = self.brain.interactive_decision(step, task, state, objects)
                
                if decision_result["action_type"] == "user_interaction_required":
                    # Wait for user decision through GUI
                    user_decision = self.gym_adapter.wait_for_user_decision(objects, timeout=30)
                    
                    if user_decision["action_type"] == "quit":
                        print("User requested quit")
                        break
                    elif user_decision["action_type"] == "skip":
                        action = self.gym_adapter.env.action_space.sample()
                        obs, reward, done, info = self.gym_adapter.step(action)
                    elif user_decision["action_type"] == "random":
                        action = self.gym_adapter.env.action_space.sample()
                        obs, reward, done, info = self.gym_adapter.step(action)
                    elif user_decision["action_type"] == "click" and objects:
                        obj_idx = user_decision.get("object_index", 0)
                        if obj_idx < len(objects):
                            operation = {
                                'operate': 'Click',
                                'params': {
                                    'coordinate': objects[obj_idx]['center']
                                }
                            }
                            result = self.execute_operation(operation)
                        else:
                            action = self.gym_adapter.env.action_space.sample()
                            obs, reward, done, info = self.gym_adapter.step(action)
                    else:
                        action = self.gym_adapter.env.action_space.sample()
                        obs, reward, done, info = self.gym_adapter.step(action)
                else:
                    action = self.gym_adapter.env.action_space.sample()
                    obs, reward, done, info = self.gym_adapter.step(action)
            else:
                # Original AI decision making
                if objects:
                    # Use do_operation_mcp instead of select_action
                    result = self.brain.do_operation_mcp(step, task, state, objects)
                    if result and 'operation' in result:
                        operation = result['operation']
                        result = self.execute_operation(operation)
                    else:
                        # Fallback: random action
                        action = self.gym_adapter.env.action_space.sample()
                        obs, reward, done, info = self.gym_adapter.step(action)
                else:
                    # No objects detected, take random action
                    action = self.gym_adapter.env.action_space.sample()
                    obs, reward, done, info = self.gym_adapter.step(action)
            
            # Update stats
            episode_stats['steps'] = step + 1
            episode_stats['total_reward'] = getattr(self.gym_adapter, 'episode_reward', 0.0)
            episode_stats['actions'].append(getattr(self, 'last_operation', action))
            episode_stats['rewards'].append(getattr(self.gym_adapter, 'current_reward', 0.0))
            
            if render:
                self.gym_adapter.render()
            
            # Check if episode is done
            if hasattr(self.gym_adapter, 'current_obs'):
                _, _, done, _ = self.gym_adapter.env.envs[0] if hasattr(self.gym_adapter.env, 'envs') else (None, None, False, None)
                if done or (hasattr(self, 'last_result') and self.last_result == 'Done'):
                    break
        
        return episode_stats
    
    def train(self, num_episodes: int = 100) -> List[Dict[str, Any]]:
        """Train the agent for multiple episodes"""
        print(f"\nStarting training for {num_episodes} episodes on {self.env_name}")
        
        training_stats = []
        
        for episode in range(num_episodes):
            episode_stats = self.run_episode()
            training_stats.append(episode_stats)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([stats['total_reward'] for stats in training_stats[-10:]])
                avg_steps = np.mean([stats['steps'] for stats in training_stats[-10:]])
                
                print(f"\nEpisodes {episode - 8}-{episode + 1}:")
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Average steps: {avg_steps:.1f}")
                print(f"  Best reward so far: {self.best_reward:.2f}")
        
        return training_stats
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent"""
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_steps = []
        
        for episode in range(num_episodes):
            episode_stats = self.run_episode()
            eval_rewards.append(episode_stats['total_reward'])
            eval_steps.append(episode_stats['steps'])
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'std_steps': np.std(eval_steps)
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Reward range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  Mean steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
        
        return results
    
    def close(self):
        """Close the agent and environment"""
        if hasattr(self, 'gym_adapter'):
            self.gym_adapter.close()
        print(f"GymBottomUpAgent closed")


class GymEye(Eye):
    """Eye component adapted for Gym environments"""
    
    def __init__(self, config: Dict[str, Any], gym_adapter: GymEnvironmentAdapter):
        super().__init__(config)
        self.gym_adapter = gym_adapter
    
    def get_screenshot_cv(self) -> np.ndarray:
        """Get screenshot from Gym environment"""
        if hasattr(self.gym_adapter, 'current_obs') and self.gym_adapter.current_obs is not None:
            return self.gym_adapter._create_screen_from_obs(self.gym_adapter.current_obs)
        else:
            # Return a black screen if no observation available
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class GymHand(Hand):
    """Hand component adapted for Gym environments"""
    
    def __init__(self, config: Dict[str, Any], gym_adapter: GymEnvironmentAdapter):
        super().__init__(config)
        self.gym_adapter = gym_adapter
    
    def click(self, x: int, y: int) -> str:
        """Convert click to Gym action"""
        operation = {
            'operate': 'Click',
            'params': {'coordinate': [x, y]}
        }
        
        action = self.gym_adapter.map_operation_to_action(operation)
        obs, reward, done, info = self.gym_adapter.step(action)
        
        return 'Success' if not done else 'Done'
    
    def key_press(self, key: str) -> str:
        """Convert key press to Gym action"""
        operation = {
            'operate': 'KeyPress',
            'params': {'key': key}
        }
        
        action = self.gym_adapter.map_operation_to_action(operation)
        obs, reward, done, info = self.gym_adapter.step(action)
        
        return 'Success' if not done else 'Done'


def create_gym_agent(env_name: str, config_path: str = None) -> GymBottomUpAgent:
    """Factory function to create a Gym agent"""
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'project_name': 'GymAgent',
            'game_name': env_name,
            'gym_environment': {
                'env_name': env_name,
                'window_size': [600, 400],
                'fps': 60
            },
            'brain': {
                'base_model': 'gpt-4o',
                'evaluate_model': 'gpt-4o'
            },
            'detector': {
                'type': 'sam',
                'sam_weights': 'weights/sam_vit_b_01ec64.pth',
                'sam_type': 'vit_b',
                'clip_model': 'ViT-B/32'
            },
            'eye': {
                'width': 600,
                'height': 400
            }
        }
    
    # Handle both 'gym' and 'gym_environment' config keys
    if 'gym' in config and 'gym_environment' not in config:
        config['gym_environment'] = config['gym']
    elif 'gym_environment' not in config:
        config['gym_environment'] = {}
    
    # Use environment name from config if available, otherwise use provided env_name
    # Priority: game_name > gym.env_id > provided env_name
    actual_env_name = config.get('game_name')
    if not actual_env_name and 'gym' in config:
        actual_env_name = config['gym'].get('env_id')
    if not actual_env_name:
        actual_env_name = env_name
    
    config['gym_environment']['env_name'] = actual_env_name
    config['game_name'] = actual_env_name
    
    return GymBottomUpAgent(config)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Gym Bottom-Up Agent')
    parser.add_argument('--env', help='Gym environment name (overrides config file)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--mode', choices=['train', 'eval', 'demo'], default='demo', help='Run mode')
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_gym_agent(args.env, args.config)
    
    try:
        if args.mode == 'train':
            agent.train(args.episodes)
        elif args.mode == 'eval':
            agent.evaluate(args.episodes)
        else:  # demo
            print(f"Running demo for {args.episodes} episodes...")
            for i in range(args.episodes):
                stats = agent.run_episode()
                print(f"Episode {i+1}: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
    
    finally:
        agent.close()