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
        
        # Store config for later use
        self.config = adapted_config
        
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
            # Get max_steps from config file
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
        
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
                        action = np.random.randint(0, self.gym_adapter.env.action_space.n)
                        obs, reward, done, info = self.gym_adapter.step(action)
                else:
                    # No objects detected, take random action
                    action = np.random.randint(0, self.gym_adapter.env.action_space.n)
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
    
    def start_crafter_interactive_with_detection(self, max_steps=None):
        """启动Crafter交互式模式，支持并行GUI和检测分析"""
        if self.env_name.lower() != 'crafter':
            print(f"❌ 交互式检测模式仅支持Crafter环境，当前环境: {self.env_name}")
            return None
            
        if max_steps is None:
            # Get max_steps from config file
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
            
        from demos.demo_grid_content_check import GridContentChecker
        import threading
        import time
        
        print("🚀 启动Crafter交互式检测模式")
        print("=" * 60)
        print("🎮 支持键盘/鼠标控制 + BottomUp Agent智能分析")
        print("🔍 实时对比GUI环境与检测系统的网格内容")
        print()
        
        checker = GridContentChecker(self.config)
        
        checker.start_gui_process()
        
        time.sleep(3)
        
        episode_stats = {
            'steps': 0,
            'total_reward': 0.0,
            'detection_accuracy': 0.0,
            'gui_running': True
        }
        
        try:
            step_count = 0
            while checker.gui_running and step_count < max_steps:
                step_count += 1
                
                current_obs = self.get_observation()
                
                reference_grid = checker.get_reference_grid_content()
                
                detected_grid = checker.get_detected_grid_content(current_obs['screen'])
                
                comparison = checker.compare_grid_contents(reference_grid, detected_grid)
                
                if step_count % 10 == 0:
                    print(f"\n📊 Step {step_count} - 检测准确率: {comparison['accuracy']:.1f}%")
                    print(f"   匹配: {comparison['matches']}, 不匹配: {comparison['mismatches']}")
                    
                    # 让BottomUp Agent分析当前状态
                    try:
                        objects = self.gym_adapter.detect_objects(current_obs['screen'])
                        if objects:
                            result = self.brain.do_operation_mcp(step_count, "探索并收集资源", current_obs, objects)
                            if result and 'operation' in result:
                                print(f"🧠 BottomUp建议: {result['operation']}")
                    except Exception as e:
                        print(f"⚠️ BottomUp分析错误: {e}")
                
                episode_stats['steps'] = step_count
                episode_stats['detection_accuracy'] = comparison['accuracy']
                
                time.sleep(0.5)  # 控制检查频率
                
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        finally:
            # 停止GUI进程
            checker.stop_gui_process()
            print("🛑 交互式检测模式结束")
        
        return episode_stats
    
    def start_crafter_interactive_launcher(self, max_steps=None, resolution='medium', no_gui=False):
        """启动纯Crafter交互式游戏模式（直接调用launcher）"""
        if self.env_name.lower() != 'crafter':
            print(f"❌ Crafter交互式模式仅支持Crafter环境，当前环境: {self.env_name}")
            return None
            
        if max_steps is None:
            # Get max_steps from config file
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
            
        from demos.crafter_interactive_launcher import demo_crafter_interactive
        
        print("🚀 启动Crafter交互式游戏模式")
        print("=" * 60)
        print("🎮 纯游戏体验 - 键盘控制 + 成就追踪")
        print("📋 WASD移动, SPACE交互, TAB睡觉, 1-6制作工具")
        print()
        
        try:
            # 使用当前配置文件路径
            config_path = getattr(self, 'config_path', 'config/gym/crafter_config.yaml')
            
            # 调用交互式launcher
            result = demo_crafter_interactive(
                resolution=resolution,
                max_steps=max_steps,
                config_path=config_path,
                no_gui=no_gui
            )
            
            print("🏁 Crafter交互式游戏结束")
            return result
            
        except Exception as e:
            print(f"❌ 启动Crafter交互式模式失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def start_hybrid_crafter_mode(self, max_steps=1000, analysis_interval=5):
        """启动混合Crafter模式：用户控制 + Agent分析建议"""
        if self.env_name.lower() != 'crafter':
            print(f"❌ 混合模式仅支持Crafter环境，当前环境: {self.env_name}")
            return None
            
        import pygame
        import threading
        import time
        from demos.crafter_interactive_launcher import get_gui_config
        
        print("🚀 启动混合Crafter模式")
        print("=" * 60)
        print("🎮 用户键盘控制 + 🤖 Agent智能建议")
        print("🔄 Agent会定期分析游戏状态并提供建议")
        print()
        
        # 初始化环境
        obs = self.reset_environment()
        self.gui_running = False
        
        # 共享状态
        shared_state = {
            'obs': obs,
            'total_reward': 0,
            'steps': 0,
            'done': False,
            'last_analysis_step': 0
        }
        
        def gui_worker():
            try:
                self.gui_running = True
                
                # 初始化pygame
                pygame.init()
                gui_config = get_gui_config(self.config)
                window_size = (gui_config['width'], gui_config['height'])
                screen = pygame.display.set_mode(window_size)
                pygame.display.set_caption(f"混合Crafter模式 - 用户控制+Agent建议 [{window_size[0]}x{window_size[1]}]")
                clock = pygame.time.Clock()
                
                print(f"✅ 混合模式GUI启动 ({window_size[0]}x{window_size[1]})")
                
                running = True
                while running and self.gui_running and not shared_state['done']:
                    # 处理pygame事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            else:
                                # 处理用户输入
                                action = self._handle_keyboard_input(event.key)
                                if action is not None:
                                    # 执行动作
                                    obs_new, reward, done_new, info = self.gym_adapter.step(action)
                                    shared_state['obs'] = obs_new
                                    shared_state['total_reward'] += reward
                                    shared_state['steps'] += 1
                                    shared_state['done'] = done_new
                                    
                                    print(f"\n🎮 用户动作: {action}, 奖励: {reward:.2f}, 总步数: {shared_state['steps']}")
                                    
                                    # 定期进行Agent分析
                                    if (shared_state['steps'] - shared_state['last_analysis_step']) >= analysis_interval:
                                        try:
                                            state = self.get_observation()
                                            objects = self.gym_adapter.detect_objects(state['screen'])
                                            if objects:
                                                result = self.brain.do_operation_mcp(
                                                    shared_state['steps'], 
                                                    "探索世界并收集资源制作工具", 
                                                    state, 
                                                    objects
                                                )
                                                if result and 'operation' in result:
                                                    print(f"💡 Agent建议: {result['operation']}")
                                                    if 'reasoning' in result:
                                                        print(f"🤔 分析原因: {result['reasoning'][:100]}...")
                                        except Exception as e:
                                            print(f"⚠️ Agent分析错误: {e}")
                                        
                                        shared_state['last_analysis_step'] = shared_state['steps']
                                    
                                    if shared_state['done']:
                                        print(f"🏁 游戏结束! 总奖励: {shared_state['total_reward']:.2f}")
                                        running = False
                    
                    # 渲染游戏
                    try:
                        self.gym_adapter._render_environment()
                        pygame.display.flip()
                    except Exception as render_error:
                        print(f"⚠️ 渲染错误: {render_error}")
                    
                    clock.tick(30)
                
                pygame.quit()
                self.gui_running = False
                print("🛑 混合模式GUI停止")
                
            except Exception as e:
                print(f"❌ GUI工作线程错误: {e}")
                import traceback
                traceback.print_exc()
                self.gui_running = False
        
        # 启动GUI线程
        gui_thread = threading.Thread(target=gui_worker, daemon=True)
        gui_thread.start()
        
        # 等待完成
        try:
            while self.gui_running and shared_state['steps'] < max_steps and not shared_state['done']:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
            self.gui_running = False
        finally:
            self.gui_running = False
            if gui_thread.is_alive():
                gui_thread.join(timeout=2.0)
        
        episode_stats = {
            'steps': shared_state['steps'],
            'total_reward': shared_state['total_reward'],
            'done': shared_state['done'],
            'mode': 'hybrid_user_agent'
        }
        
        print(f"\n🏁 混合模式结束: {shared_state['steps']} 步, {shared_state['total_reward']:.2f} 总奖励")
        return episode_stats
    
    def run_interactive(self, max_steps=None):
        """Run interactive mode with GUI control and detection analysis"""
        if max_steps is None:
            # Get max_steps from config file
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
            
        import pygame
        import threading
        import time
        from demos.crafter_interactive_launcher import get_gui_config
        
        print("🚀 Starting Interactive BottomUp Mode with GUI Control")
        print("=" * 60)
        print("This will start an interactive GUI window with BottomUp detection")
        print("🎮 Use WASD to move, SPACE to interact, ESC to close")
        print("🤖 BottomUp Agent will analyze each step")
        print()
        
        # Initialize the environment
        obs = self.reset_environment()
        self.gui_running = False
        self.game_step_count = 0
        
        # Shared state variables
        shared_state = {
            'obs': obs,
            'total_reward': 0,
            'steps': 0,
            'done': False
        }
        
        def gui_worker():
            try:
                print("🚀 Starting interactive GUI process...")
                self.gui_running = True
                
                # Initialize pygame for the GUI
                pygame.init()
                # Use actual GUI config from crafter_config.yaml instead of forcing 'low'
                gui_config = get_gui_config(self.config)
                window_size = (gui_config['width'], gui_config['height'])
                screen = pygame.display.set_mode(window_size)
                pygame.display.set_caption(f"BottomUp Interactive - {self.config.get('game_name', 'Game')} [{window_size[0]}x{window_size[1]}]")
                clock = pygame.time.Clock()
                
                print(f"✅ Interactive GUI started ({window_size[0]}x{window_size[1]})")
                print("🎮 Use WASD to move, SPACE to interact, ESC to close GUI")
                
                # Main GUI loop
                running = True
                while running and self.gui_running and not shared_state['done']:
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
                                    # Apply action through BottomUp framework
                                    obs_new, reward, done_new, info = self.gym_adapter.step(action)
                                    shared_state['obs'] = obs_new
                                    shared_state['total_reward'] += reward
                                    shared_state['steps'] += 1
                                    shared_state['done'] = done_new
                                    self.game_step_count += 1
                                    
                                    # Trigger BottomUp analysis
                                    print(f"\n--- Step {self.game_step_count} ---")
                                    print(f"🤖 BottomUp Analysis - Action: {action}, Reward: {reward:.2f}")
                                    
                                    # Get BottomUp agent's analysis of the current state
                                    try:
                                        state = self.get_observation()
                                        objects = self.gym_adapter.detect_objects(state['screen'])
                                        if objects:
                                            result = self.brain.do_operation_mcp(self.game_step_count, "Complete the game", state, objects)
                                            if result and 'operation' in result:
                                                print(f"🧠 BottomUp would choose: {result['operation']}")
                                    except Exception as e:
                                        print(f"⚠️ BottomUp analysis error: {e}")
                                    
                                    if shared_state['done']:
                                        print(f"🏁 Episode completed! Total reward: {shared_state['total_reward']:.2f}")
                                        running = False
                    
                    # Render the game
                    try:
                        # Use GymAdapter's rendering method which handles size properly
                        self.gym_adapter._render_environment()
                        pygame.display.flip()
                    except Exception as render_error:
                        print(f"⚠️ Render error: {render_error}")
                    
                    clock.tick(30)  # 30 FPS
                
                pygame.quit()
                self.gui_running = False
                print("🛑 GUI process stopped")
                
            except Exception as e:
                print(f"❌ GUI worker error: {e}")
                import traceback
                traceback.print_exc()
                self.gui_running = False
        
        # Start GUI in separate thread
        gui_thread = threading.Thread(target=gui_worker, daemon=True)
        gui_thread.start()
        
        # Wait for GUI to complete or max steps
        try:
            while self.gui_running and shared_state['steps'] < max_steps and not shared_state['done']:
                time.sleep(0.1)  # Small delay to prevent busy waiting
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            self.gui_running = False
        finally:
            # Ensure GUI thread is properly stopped
            self.gui_running = False
            if gui_thread.is_alive():
                gui_thread.join(timeout=2.0)
                if gui_thread.is_alive():
                    print("⚠️ GUI thread did not stop gracefully")
        
        episode_stats = {
            'steps': shared_state['steps'],
            'total_reward': shared_state['total_reward'],
            'done': shared_state['done']
        }
        
        print(f"\n🏁 Interactive session finished: {shared_state['steps']} steps, {shared_state['total_reward']:.2f} total reward")
        return episode_stats
    
    def _handle_keyboard_input(self, key):
        """Convert pygame key to Crafter action (same as demo_grid_content_check.py)"""
        import pygame
        
        # Keyboard mapping (same as crafter_interactive_launcher and demo)
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
    parser.add_argument('--mode', choices=['train', 'eval', 'demo', 'interactive', 'crafter_launcher', 'crafter_detection', 'hybrid'], default='demo', help='Run mode')
    parser.add_argument('--resolution', default='medium', help='GUI resolution for Crafter modes (tiny/small/low/medium/high/ultra)')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps for interactive modes')
    parser.add_argument('--analysis_interval', type=int, default=5, help='Steps between Agent analysis in hybrid mode')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI (background mode)')
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_gym_agent(args.env, args.config)
    
    try:
        if args.mode == 'train':
            agent.train(args.episodes)
        elif args.mode == 'eval':
            agent.evaluate(args.episodes)
        elif args.mode == 'interactive':
            print("🚀 启动标准交互模式...")
            stats = agent.run_interactive(max_steps=args.max_steps)
            print(f"交互模式结束: {stats['steps']} 步, {stats['total_reward']:.2f} 奖励")
        elif args.mode == 'crafter_launcher':
            print("🎮 启动Crafter交互式游戏模式...")
            stats = agent.start_crafter_interactive_launcher(
                max_steps=args.max_steps,
                resolution=args.resolution,
                no_gui=args.no_gui
            )
            if stats:
                print(f"Crafter游戏结束: {stats}")
        elif args.mode == 'crafter_detection':
            print("🔍 启动Crafter检测分析模式...")
            stats = agent.start_crafter_interactive_with_detection(max_steps=args.max_steps)
            if stats:
                print(f"检测模式结束: 准确率 {stats['detection_accuracy']:.1f}%, {stats['steps']} 步")
        elif args.mode == 'hybrid':
            print("🤝 启动混合模式 (用户控制 + Agent建议)...")
            stats = agent.start_hybrid_crafter_mode(
                max_steps=args.max_steps,
                analysis_interval=args.analysis_interval
            )
            if stats:
                print(f"混合模式结束: {stats['steps']} 步, {stats['total_reward']:.2f} 奖励")
        else:  # demo
            print(f"Running demo for {args.episodes} episodes...")
            for i in range(args.episodes):
                stats = agent.run_episode()
                print(f"Episode {i+1}: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
    
    finally:
        agent.close()