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
from BottomUpAgent.pre_knowledge import get_pre_knowledge

class GymBottomUpAgent(BottomUpAgent):
    """
    Bottom-Up Agent specialized for Gym environments.
    Extends BottomUpAgent to work with any Gym environment through GymAdapter.
    """
    
    def __init__(self, config: Dict[str, Any], simple_mode: bool = False):
        # Initialize gym adapter with unified config structure
        self.gym_config = config.get('gym', {})
        self.simple_mode = simple_mode
        
        # Determine environment name from multiple possible sources
        self.env_name = (
            self.gym_config.get('env_id') or 
            'CartPole-v1'
        )
        
        # Prepare complete config for gym adapter
        gym_adapter_config = config.copy()  # Use full config instead of just gym_config
        gym_adapter_config.update(self.gym_config)  # Override with gym-specific settings
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
        
        if simple_mode:
            # Simple mode: Skip complex BottomUpAgent initialization
            print("🚀 Simple mode enabled - bypassing complex initialization")
            self._simple_init(adapted_config)
        else:
            # Initialize parent BottomUpAgent
            super().__init__(adapted_config)
            
            # Override components for Gym integration
            self._setup_gym_components()
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.best_reward = float('-inf')
        
        # Parallel execution support
        self.parallel_enabled = True
        self.gui_process = None
        self.detection_thread = None
        
        # Initialize parallel launcher if enabled
        if self.parallel_enabled:
            self._setup_parallel_launcher()
        
        # Initialize enhanced skill generation system (inspired by BottomUpAgent)
        self.new_skills = []
        skill_config = config.get('automatic_gameplay', {}).get('skill_generation', {})
        self.skill_generation_enabled = skill_config.get('enabled', True)
        self.skill_generation_threshold = skill_config.get('threshold', 3)  # Generate skill after N successful operations
        
        # Enhanced skill management
        self.skill_evolution_enabled = skill_config.get('evolution_enabled', True)
        self.skill_fitness_threshold = skill_config.get('fitness_threshold', 2.0)
        self.skill_observation_threshold = skill_config.get('observation_threshold', 4)
        self.max_skills_per_cluster = skill_config.get('max_skills_per_cluster', 10)
        
        # Skill generation tracking
        self.skill_generation_history = []
        self.last_skill_evolution_step = 0
        self.skill_evolution_interval = skill_config.get('evolution_interval', 50)  # Evolve skills every N steps
        
        # Initialize automatic gameplay system from config
        auto_config = config.get('automatic_gameplay', {})
        self.automatic_gameplay_enabled = auto_config.get('enabled', True)
        self.scene_analysis_interval = auto_config.get('scene_analysis_interval', 3)  # Analyze scene every N steps
        self.max_automatic_skills = auto_config.get('max_automatic_skills', 5)
        self.last_scene_analysis_step = 0
        self.scene_context_history = []  # Track scene changes for context
        self.automatic_skill_queue = []  # Queue of automatically generated skills
        
        # Initialize scene detection configuration
        scene_config = auto_config.get('scene_detection', {})
        self.scene_complexity_threshold = scene_config.get('complexity_threshold', 5)
        self.context_history_size = scene_config.get('context_history_size', 10)
        self.new_object_threshold = scene_config.get('new_object_threshold', 2)
        self.exploration_trigger_objects = scene_config.get('exploration_trigger_objects', 5)
        
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
    
    def _simple_init(self, config: Dict[str, Any]):
        """Simple initialization that bypasses complex BottomUpAgent components"""
        # Only initialize essential attributes needed for GUI
        self.logger = None  # Skip WandB logger
        self.eye = None     # Skip Eye module
        self.hand = None    # Skip Hand module
        self.detector = None # Skip Detector/CLIP
        self.brain = None   # Skip Brain
        self.teacher = None # Skip Teacher
        
        # Initialize basic attributes that GUI expects
        self.config = config
        self.running = True
        self.paused = False
        
        print("✅ Simple initialization completed - complex components bypassed")
    
    def _setup_gym_components(self):
        """Setup components specifically for Gym environment interaction"""
        # Override eye to work with gym adapter
        self.eye = GymEye(self.config, self.gym_adapter)
        
        # Override hand to work with gym actions
        self.hand = GymHand(self.config, self.gym_adapter)
        
        # Keep existing detector and brain
        # They should work with the adapted observations
    
    def _setup_parallel_launcher(self):
        """Setup parallel launcher for GUI and detection"""
        import threading
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor
        
        # Create thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Setup detection queue for parallel processing
        self.detection_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        
        print("🚀 Parallel launcher initialized")
    
    def start_parallel_gui(self):
        """Start GUI in parallel process"""
        if not self.parallel_enabled:
            return
            
        try:
            # Start gym environment rendering in separate process
            import multiprocessing
            
            def gui_worker():
                """Worker function for GUI process"""
                try:
                    # Enable rendering for gym environment
                    if hasattr(self.gym_adapter.env, 'render'):
                        self.gym_adapter.env.render()
                    return True
                except Exception as e:
                    print(f"GUI process error: {e}")
                    return False
            
            # Submit GUI task to thread pool
            self.gui_future = self.thread_pool.submit(gui_worker)
            print("🖼️ GUI started in parallel")
            
        except Exception as e:
            print(f"Failed to start parallel GUI: {e}")
            self.parallel_enabled = False
    
    def start_parallel_detection(self):
        """Start detection in parallel thread"""
        if not self.parallel_enabled:
            return
            
        try:
            def detection_worker():
                """Worker function for detection thread"""
                while self.parallel_enabled:
                    try:
                        # Get current observation
                        obs = self.get_observation()
                        if obs is not None:
                            # Perform detection
                            detection_result = self.detector.get_detected_objects(obs['screen'])
                            # Put result in queue
                            if not self.result_queue.full():
                                self.result_queue.put(detection_result)
                        
                        # Small delay to prevent excessive CPU usage
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Detection worker error: {e}")
                        break
            
            # Start detection thread
            self.detection_future = self.thread_pool.submit(detection_worker)
            print("👁️ Detection started in parallel")
            
        except Exception as e:
             print(f"Failed to start parallel detection: {e}")
             self.parallel_enabled = False
    
    def _cleanup_parallel_components(self):
        """Cleanup parallel components"""
        try:
            # Stop parallel detection
            self.parallel_enabled = False
            
            # Wait for futures to complete
            if hasattr(self, 'gui_future'):
                self.gui_future.cancel()
            if hasattr(self, 'detection_future'):
                self.detection_future.cancel()
            
            # Clear queues
            if hasattr(self, 'detection_queue'):
                while not self.detection_queue.empty():
                    try:
                        self.detection_queue.get_nowait()
                    except:
                        break
            
            if hasattr(self, 'result_queue'):
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        break
            
            print("🧹 Parallel components cleaned up")
            
        except Exception as e:
            print(f"Error cleaning up parallel components: {e}")
    
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
            result = self.gym_adapter._process_observation(self.gym_adapter.current_obs)
            return result
        else:
            # Fallback: reset if no current observation
            result = self.reset_environment()
            return result
    
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
    
    def run_step(self, step, task):
        """Run a single step using BottomUpAgent pipeline"""
        self.logger.log({"step": step}, step)
        
        # Get observation from gym environment
        ob = self.get_observation()
        
        # Perform step-by-step detection
        print(f"\n--- Step {step} Detection ---")
        print(f"🔍 Analyzing game state and detecting objects...")
        
        # Ensure detector has access to crafter environment
        if hasattr(self.gym_adapter, 'env') and hasattr(self, 'detector'):
            # Set crafter environment reference for detector
            if hasattr(self.detector, 'crafter_env'):
                self.detector.crafter_env = self.gym_adapter.env
            
            # Get current screen/frame for detection
            screen = ob.get('screen')
            if screen is not None:
                try:
                    # Extract objects using crafter API detection
                    detected_objects = self.detector.extract_objects_crafter_api(screen)
                    print(f"✅ Detected {len(detected_objects)} objects in step {step}")
                    
                    # Log object types for debugging
                    if len(detected_objects) > 0:
                        object_types = {}
                        for obj in detected_objects:
                            obj_type = obj.get('type', 'unknown')
                            object_types[obj_type] = object_types.get(obj_type, 0) + 1
                        print(f"📊 Object distribution: {dict(sorted(object_types.items()))}")
                    else:
                        print("⚠️ No objects detected - this may cause the process to hang")
                        print("🔧 Checking detector configuration...")
                        
                        # Debug detector configuration
                        if hasattr(self.detector, 'config'):
                            detector_config = self.detector.config.get('detector', {})
                            print(f"🔧 Detector type: {detector_config.get('type', 'unknown')}")
                            print(f"🔧 Detection method: {detector_config.get('crafter_api', {}).get('detection_method', 'unknown')}")
                        
                        # Try alternative detection method if available
                        if hasattr(self.detector, 'get_detected_objects'):
                            try:
                                alt_objects = self.detector.get_detected_objects(screen)
                                print(f"🔄 Alternative detection found {len(alt_objects)} objects")
                                if len(alt_objects) > 0:
                                    detected_objects = alt_objects
                            except Exception as alt_e:
                                print(f"❌ Alternative detection failed: {alt_e}")
                    
                    # Update observation with detected objects
                    ob['detected_objects'] = detected_objects
                    
                except Exception as detection_error:
                    print(f"❌ Detection error in step {step}: {detection_error}")
                    print("🔧 Falling back to empty detection result")
                    ob['detected_objects'] = []
            else:
                print(f"⚠️ No screen data available for detection in step {step}")
                ob['detected_objects'] = []
        else:
            print(f"⚠️ Detector or environment not properly configured for step {step}")
            ob['detected_objects'] = []
        
        # Get or create state using long-term memory (if available)
        if hasattr(self, 'brain') and self.brain is not None and hasattr(self.brain, 'long_memory') and self.brain.long_memory is not None:
            state = self.brain.long_memory.get_state(ob)
            if state is None:
                print("No state found, create state")
                from BottomUpAgent.Mcts import MCTS
                mcts = MCTS()
                
                state = {
                    'id': None,
                    'state_feature': ob['state_feature'],
                    'object_ids': [],
                    'mcts': mcts,
                    'skill_clusters': [],
                }
                state['id'] = self.brain.long_memory.save_state(state)
            
            # Get skill clusters
            skill_clusters_ids = state['skill_clusters']
            skill_clusters = self.brain.long_memory.get_skill_clusters_by_ids(skill_clusters_ids)
        else:
            # Simple mode: create basic state without long-term memory
            print("🚀 Simple mode: using basic state management")
            from BottomUpAgent.Mcts import MCTS
            mcts = MCTS()
            
            state = {
                'id': f'simple_state_{step}',
                'state_feature': ob.get('state_feature', {}),
                'object_ids': [],
                'mcts': mcts,
                'skill_clusters': [],
            }
            skill_clusters = []
        
        if len(skill_clusters) == 0:
            skill_cluster = None
            skills = []
        else:
            # Check if brain components are available
            if hasattr(self, 'brain') and self.brain is not None and hasattr(self.brain, 'select_skill_cluster'):
                skill_cluster_id = self.brain.select_skill_cluster(step, ob, skill_clusters)
                if skill_cluster_id is None:
                    return 'Continue'
                
                skill_cluster = self.brain.long_memory.get_skill_clusters_by_id(skill_cluster_id)
                skills = self.brain.long_memory.get_skills_by_ids(skill_cluster['members'])
                print(f"selected skill_cluster id: {skill_cluster['id']} name: {skill_cluster['name']} description: {skill_cluster['description']}")
            else:
                # Simple mode: use basic skill selection
                print("🚀 Simple mode: using basic skill selection")
                skill_cluster = None
                skills = []
        
        # Automatic gameplay: analyze scene and generate skills if enabled
        if self.automatic_gameplay_enabled and (step - self.last_scene_analysis_step) >= self.scene_analysis_interval:
            self._analyze_scene_and_generate_automatic_skills(step, ob, state)
            self.last_scene_analysis_step = step
        
        # Skill evolution: periodically evolve and optimize skills
        if (self.skill_evolution_enabled and 
            hasattr(self, 'brain') and self.brain is not None and 
            (step - self.last_skill_evolution_step) >= self.skill_evolution_interval):
            self._perform_skill_evolution(step, state)
            self.last_skill_evolution_step = step
        
        # Process automatic skill queue first
        if len(self.automatic_skill_queue) > 0:
            auto_skill = self.automatic_skill_queue.pop(0)
            print(f"🤖 Executing automatic skill: {auto_skill['name']} - {auto_skill['description']}")
            if self.use_mcp:
                result = self.exploit_mcp(step, task, auto_skill)
            else:
                result = self.exploit(step, task, auto_skill)
            if result != 'Fail':
                return result
        
        # Main decision loop: explore vs exploit
        result = 'Retry'
        suspended_skill_ids = []
        while result == 'Retry':
            # Check if brain is available for skill selection
            if hasattr(self, 'brain') and self.brain is not None and hasattr(self.brain, 'select_skill'):
                skill, suspend_flag = self.brain.select_skill(skills, skill_cluster, suspended_skill_ids, self.close_explore)
            else:
                # Simple mode: create basic explore skill
                print("🚀 Simple mode: using basic explore skill")
                skill = {
                    'name': 'Explore',
                    'id': f'simple_explore_{step}',
                    'description': 'Basic exploration in simple mode'
                }
                suspend_flag = False
            
            if skill['name'] == 'Explore':
                self.logger.log({"decision": 0, "decision_text": "Explore"}, step)
                print("🔍 Exploring new skills...")
                
                result = self.explore(step, state, skill, skill_clusters)
                if skill_cluster is not None and hasattr(self, 'brain') and self.brain is not None:
                    self.brain.long_memory.update_skill_cluster_explore_nums(skill_cluster['id'], skill_cluster['explore_nums'] + 1)
                break
            else:
                self.logger.log({"decision": 1, "decision_text": "Exploit"})
                print(f"🎯 Exploiting skill: {skill['name']}")
                
                suspended_skill_ids.append(skill['id'])
                
                # Enhanced exploit with MCP support
                if self.use_mcp:
                    result = self.exploit_mcp(step, task, skill)
                else:
                    result = self.exploit(step, task, skill)
                    
                if result == 'Fail':
                    suspended_skill_ids.append(skill['id'])
                    result = 'Retry'
                    
            if result == 'ExploreFail' and suspend_flag:
                print("Explore failed, suspend skill cluster")
                if not hasattr(self, 'suspended_skill_cluster_ids'):
                    self.suspended_skill_cluster_ids = []
                self.suspended_skill_cluster_ids.append(skill_cluster['id'])
            elif result == 'Continue':
                if hasattr(self, 'suspended_skill_cluster_ids'):
                    self.suspended_skill_cluster_ids.clear()
                    
            self.state_reset()
        
        # Skill evolution
        self.brain.skill_evolution(step, skills, skill_cluster)
        
        return result
    
    def run_episode(self, task=None, max_steps: int = None, render=True, interactive_mode=False) -> Dict[str, Any]:
        """Run a single episode using BottomUpAgent pipeline"""
        if max_steps is None:
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_episode_steps', 1000)
        
        # Reset environment and start GUI
        obs = self.reset_environment()
        
        # Start parallel components
        if self.parallel_enabled:
            self.start_parallel_gui()
            self.start_parallel_detection()
        
        if render:
            self.gym_adapter.start_gui(interactive_mode=interactive_mode)
        
        episode_stats = {
            'episode': self.current_episode,
            'steps': 0,
            'total_reward': 0.0,
            'actions': [],
            'rewards': []
        }
        
        # Use dynamic task planning if no task is provided
        if task is None:
            task = self._plan_current_task(obs, 0)
        
        print(f"🎮 Starting episode {self.current_episode} with task: {task}")
        
        # Initialize step counter
        step = self.logger.last_value('step') + 1 if self.logger.last_value('step') is not None else 0
        
        # Run episode using BottomUpAgent pipeline
        finished = False
        while not finished and step < max_steps:
            print(f"\n--- Step {step} ---")
            
            # Run single step using BottomUpAgent pipeline
            result = self.run_step(step, task)
            
            # Update episode stats
            episode_stats['steps'] = step + 1
            
            # Check if episode should end
            if result == 'Finished':
                finished = True
                print("🏁 Episode completed successfully!")
            elif result == 'Done':
                finished = True
                print("🎯 Task completed!")
            
            # Check if gym environment episode is done
            if hasattr(self.gym_adapter, 'last_done') and self.gym_adapter.last_done:
                finished = True
                episode_reward = getattr(self.gym_adapter, 'episode_reward', 0.0)
                episode_stats['total_reward'] = episode_reward
                print(f"🏆 Gym episode finished with reward: {episode_reward:.2f}")
            
            step += 1
            
            # Small delay for observation
            import time
            time.sleep(0.1)
        
        # Cleanup parallel components
        if self.parallel_enabled:
            self._cleanup_parallel_components()
        
        return episode_stats
    
    def explore(self, step, state, skill, skill_clusters):
        """Explore new skills using MCTS and skill augmentation"""
        print(f"🔍 Begin explore with GYM mode")
        if self.close_explore:
            return 'ExploreFail'
        
        mcts = state['mcts']
        if skill['mcts_node_id'] is None:
            mcts_node_id = 0
        else:
            mcts_node_id = skill['mcts_node_id']
        
        mcts_node = mcts.get_node(mcts_node_id)
        print(f"mcts_node_id: {mcts_node_id}")
        parent_node = mcts.get_node(mcts_node.parent_id) if mcts_node.parent_id is not None else None
        
        new_skill_num = 0
        stop_flag = False
        new_skills = []
        
        # Explore from parent node
        print(f"🔍 Begin explore skill augment from parent node")
        if (parent_node is not None) and (len(parent_node.operations) < self.max_operation_length):
            while (not parent_node.is_fixed) and (not stop_flag) and (new_skill_num < 3):
                new_skill, stop_flag = self.skill_augment(step, state, parent_node)
                if new_skill is not None:
                    new_skill_num += 1
                    new_skills.append(new_skill)
        
        # Explore from selected node
        print(f"🔍 Begin explore skill augment from selected node")
        new_skill_num = 0
        if (len(mcts_node.operations) < self.max_operation_length):
            while (not mcts_node.is_fixed) and (not stop_flag) and (new_skill_num < 3):
                new_skill, stop_flag = self.skill_augment(step, state, mcts_node)
                if new_skill is not None:
                    new_skill_num += 1
                    new_skills.append(new_skill)
        
        # Enhanced MCTS exploration: try breadth-first selection for better coverage
        if len(new_skills) < 2 and not stop_flag:
            print(f"🔍 Enhanced MCTS exploration: breadth-first node selection")
            bsf_node = mcts.random_select_bsf()
            if bsf_node is not None and bsf_node.node_id != mcts_node.node_id:
                print(f"🎯 Selected BSF node {bsf_node.node_id} with {len(bsf_node.operations)} operations")
                exploration_attempts = 0
                while (not bsf_node.is_fixed) and (not stop_flag) and (exploration_attempts < 2):
                    new_skill, stop_flag = self.skill_augment(step, state, bsf_node)
                    if new_skill is not None:
                        new_skills.append(new_skill)
                    exploration_attempts += 1
        
        # Process skill generation if enabled
        if self.skill_generation_enabled and len(new_skills) > 0:
            try:
                # Get current skill clusters for merging
                current_skill_clusters = self.brain.long_memory.get_skill_clusters_by_ids(state['skill_clusters'])
                
                # Generate and save skills through Brain
                obs = self.get_observation()
                for new_skill in new_skills:
                    if new_skill.get('operations'):
                        generated_skill = self.brain.generate_and_save_skill(
                            step, [obs], new_skill['operations'], state['id'], 
                            new_skill.get('mcts_node_id'), self
                        )
                        if generated_skill:
                            self.new_skills.append(generated_skill)
                
                # Batch merge and save skills if we have enough
                if len(self.new_skills) >= self.skill_generation_threshold:
                    self.brain.merge_and_save_skills(step, state, current_skill_clusters, self.new_skills)
                    self.new_skills.clear()
                    
            except Exception as e:
                print(f"⚠️ Error in skill generation: {e}")
        
        # Update MCTS node values based on exploration results
        self._update_mcts_node_values(mcts, new_skills)
    
    def _update_mcts_node_values(self, mcts, new_skills):
        """Update MCTS node values based on exploration results"""
        try:
            for skill in new_skills:
                if skill.get('mcts_node_id') is not None:
                    node = mcts.get_node(skill['mcts_node_id'])
                    if node is not None:
                        # Update node value based on skill fitness
                        skill_fitness = skill.get('fitness', 0)
                        # Increase node value for successful skill generation
                        node.value += max(1, skill_fitness * 0.1)
                        node.n_visits += 1
                        
                        # Update optimal node if this node has higher value
                        if node.value > mcts.nodes[mcts.optimal_node_id].value:
                            mcts.optimal_node_id = node.node_id
                        
                        print(f"🔄 Updated MCTS node {node.node_id}: value={node.value:.2f}, visits={node.n_visits}")
        except Exception as e:
            print(f"⚠️ Error updating MCTS node values: {e}")
    
    def _analyze_scene_and_generate_automatic_skills(self, step, observation, state):
        """Analyze current scene and generate automatic skills based on objects and context"""
        try:
            print(f"🔍 Analyzing scene for automatic skill generation (step {step})")
            
            # Get current screen and detect objects
            screen = observation['screen']
            
            # Check if detector is available (not in simple mode)
            if hasattr(self, 'detector') and self.detector is not None:
                detected_objects = self.detector.get_detected_objects(screen)
            else:
                # In simple mode, create basic object detection from observation
                detected_objects = self._get_basic_detected_objects(observation)
            
            # Generate comprehensive scene summary
            scene_summary = self._generate_scene_summary_json(step, observation, detected_objects, state)
            
            # Print JSON scene summary for MCP context
            print("\n" + "=" * 80)
            print("📋 JSON SCENE SUMMARY FOR MCP CONTEXT")
            print("=" * 80)
            import json
            print(json.dumps(scene_summary, indent=2, ensure_ascii=False))
            print("=" * 80)
            
            # Update scene context history with enhanced information
            scene_context = {
                'step': step,
                'objects': detected_objects,
                'screen_hash': hash(screen.tobytes()) if hasattr(screen, 'tobytes') else hash(str(screen)),
                'scene_summary': scene_summary
            }
            self.scene_context_history.append(scene_context)
            
            # Keep only recent history based on config
            if len(self.scene_context_history) > self.context_history_size:
                self.scene_context_history.pop(0)
            
            # Analyze scene changes and object interactions
            if len(self.scene_context_history) >= 2:
                prev_context = self.scene_context_history[-2]
                current_context = self.scene_context_history[-1]
                
                # Detect new objects or significant changes
                new_objects = self._detect_new_objects(prev_context['objects'], current_context['objects'])
                
                if len(new_objects) >= self.new_object_threshold:
                    print(f"🆕 Detected {len(new_objects)} new objects for automatic skill generation")
                    
                    # Generate automatic skills based on new objects (limit based on config)
                    max_skills_to_add = min(len(new_objects), self.max_automatic_skills - len(self.automatic_skill_queue))
                    for obj in new_objects[:max_skills_to_add]:
                        auto_skill = self._generate_automatic_skill_for_object(obj, observation, state)
                        if auto_skill:
                            self.automatic_skill_queue.append(auto_skill)
                            print(f"➕ Added automatic skill: {auto_skill['name']}")
            
            # Generate exploration skills based on scene complexity (using config)
            if len(detected_objects) >= self.exploration_trigger_objects and len(self.automatic_skill_queue) < (self.max_automatic_skills // 2):
                exploration_skill = self._generate_exploration_skill(detected_objects, observation)
                if exploration_skill:
                    self.automatic_skill_queue.append(exploration_skill)
                    print(f"🔍 Added exploration skill: {exploration_skill['name']}")
                    
        except Exception as e:
            print(f"⚠️ Error in automatic scene analysis: {e}")
    
    def _get_basic_detected_objects(self, observation):
        """Basic object detection for simple mode without full detector"""
        detected_objects = []
        
        try:
            # For Crafter environment, extract basic information from observation
            if 'inventory' in observation:
                inventory = observation['inventory']
                for item_name, count in inventory.items():
                    if count > 0:
                        detected_objects.append({
                            'name': item_name,
                            'type': 'inventory_item',
                            'count': count,
                            'confidence': 1.0
                        })
            
            # Add basic screen analysis (simplified)
            screen = observation.get('screen', None)
            if screen is not None:
                # Simple heuristic: assume there are always some basic objects
                detected_objects.extend([
                    {'name': 'ground', 'type': 'terrain', 'confidence': 0.9},
                    {'name': 'player', 'type': 'character', 'confidence': 1.0}
                ])
                
        except Exception as e:
            print(f"⚠️ Error in basic object detection: {e}")
            # Fallback: return minimal objects
            detected_objects = [
                {'name': 'environment', 'type': 'general', 'confidence': 0.5}
            ]
        
        return detected_objects
    
    def _get_grid_content_from_environment(self):
        """Get grid content directly from Crafter environment (parameterized grid around player)"""
        try:
            grid_content = {}
            
            # Get environment reference
            env = self.gym_adapter.env if hasattr(self.gym_adapter, 'env') else None
            if not env:
                return grid_content
            
            # Access Crafter environment properly
            crafter_env = env
            if hasattr(env, 'unwrapped'):
                crafter_env = env.unwrapped
            elif hasattr(env, 'env'):
                crafter_env = env.env
            
            if not hasattr(crafter_env, '_world') or not hasattr(crafter_env, '_player'):
                return grid_content
            
            world = crafter_env._world
            player = crafter_env._player
            player_pos = getattr(player, 'pos', np.array([0, 0]))
            
            # Get grid dimensions from config (default to 7x9 for Crafter)
            crafter_settings = getattr(self.config, 'crafter_settings', {})
            grid_size = crafter_settings.get('grid_size', [7, 9])
            grid_rows, grid_cols = grid_size[0], grid_size[1]
            
            # Calculate center positions (player is at center of grid)
            center_row = grid_rows // 2
            center_col = grid_cols // 2
            
            # Calculate grid positions relative to player
            for row in range(grid_rows):
                for col in range(grid_cols):
                    # Calculate world position for this grid cell
                    # Grid is centered on player at (center_row, center_col)
                    world_x = player_pos[0] + (col - center_col)
                    world_y = player_pos[1] + (row - center_row)
                    world_pos = np.array([world_x, world_y])
                    
                    # Check if position is within world bounds using world.area
                    if (0 <= world_x < world.area[0] and 
                        0 <= world_y < world.area[1]):
                        
                        # Get material and object at this position
                        material, obj = world[world_pos]
                        
                        # Determine content type
                        content_type = 'empty'
                        if obj:
                            content_type = obj.__class__.__name__.lower()
                            # Add direction for facing objects
                            if hasattr(obj, 'facing'):
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
                            'material': material,
                            'obj': obj.__class__.__name__.lower() if obj else None
                        }
                    else:
                        grid_content[(row, col)] = {
                            'type': 'out_of_bounds',
                            'material': None,
                            'obj': None
                        }
            
            return grid_content
            
        except Exception as e:
            print(f"⚠️ Error getting grid content from environment: {e}")
            return {}
    
    def _generate_scene_summary_json(self, step, observation, detected_objects, state):
        """Generate scene summary JSON matching demo format"""
        try:
            import json
            import numpy as np
            
            # Helper function to convert numpy types to Python types

            
            # Helper function to compress consecutive coordinates
            def compress_coordinates(coords):
                """Convert list of coordinates to compact representation"""
                if not coords:
                    return []
                
                # Sort coordinates by row, then by column
                sorted_coords = sorted(coords)
                
                # Group by rows
                row_groups = {}
                for row, col in sorted_coords:
                    if row not in row_groups:
                        row_groups[row] = []
                    row_groups[row].append(col)
                
                # Compress each row's columns
                compressed = []
                for row in sorted(row_groups.keys()):
                    cols = sorted(row_groups[row])
                    
                    # Find consecutive sequences in columns
                    if len(cols) == 1:
                        compressed.append([row, cols[0]])
                    else:
                        # Group consecutive columns
                        i = 0
                        while i < len(cols):
                            start_col = cols[i]
                            end_col = start_col
                            
                            # Find end of consecutive sequence
                            while i + 1 < len(cols) and cols[i + 1] == cols[i] + 1:
                                i += 1
                                end_col = cols[i]
                            
                            # Add to compressed format
                            if start_col == end_col:
                                compressed.append([row, start_col])
                            else:
                                compressed.append([row, [start_col, end_col]])
                            
                            i += 1
                
                return compressed
            
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Get grid content from environment
            grid_content = self._get_grid_content_from_environment()
            
            # Get player information
            player_inventory = {}
            player_pos = [0, 0]
            player_facing = [0, 1]  # Default facing south
            
            try:
                env = self.gym_adapter.env if hasattr(self.gym_adapter, 'env') else None
                if env and hasattr(env, '_player'):
                    # Get inventory (only non-zero items)
                    if hasattr(env._player, 'inventory'):
                        for item, count in env._player.inventory.items():
                            if count > 0:
                                player_inventory[item] = int(count)
                    
                    # Get player position and facing
                    if hasattr(env._player, 'pos'):
                        player_pos = convert_numpy_types(env._player.pos)
                    if hasattr(env._player, 'facing'):
                        player_facing = convert_numpy_types(env._player.facing)
            except Exception as e:
                print(f"⚠️ Could not get player info: {e}")
            
            # Get grid dimensions from config (default to 7x9 for Crafter)
            crafter_settings = getattr(self.config, 'crafter_settings', {})
            grid_size = crafter_settings.get('grid_size', [7, 9])
            grid_rows, grid_cols = grid_size[0], grid_size[1]
            
            # Calculate center positions (player is at center of grid)
            center_row = grid_rows // 2
            center_col = grid_cols // 2
            
            # Find player's grid position (center of parameterized grid)
            player_grid_pos = [center_row, center_col]  # Player is always at center of grid
            
            # Get immediate surroundings (only non-empty cells)
            surroundings = {}
            directions = {
                (-1, -1): "northwest", (-1, 0): "north", (-1, 1): "northeast",
                (0, -1): "west", (0, 1): "east",
                (1, -1): "southwest", (1, 0): "south", (1, 1): "southeast"
            }
            
            for (dr, dc), direction in directions.items():
                r, c = player_grid_pos[0] + dr, player_grid_pos[1] + dc
                if 0 <= r < grid_rows and 0 <= c < grid_cols:  # Parameterized grid bounds
                    cell = grid_content.get((r, c), {})
                    content = cell.get('type', 'empty')
                    if content != 'empty' and content != 'grass':  # Skip empty and grass
                        surroundings[direction] = content
            
            # Count visible elements with positions (excluding grass)
            element_counts = {}
            element_positions = {}
            
            for row in range(grid_rows):
                for col in range(grid_cols):
                    cell = grid_content.get((row, col), {})
                    content = cell.get('type', 'empty')
                    
                    if content != 'empty' and content != 'grass':
                        element_counts[content] = element_counts.get(content, 0) + 1
                        
                        # Store position information using [row, col] format
                        if content not in element_positions:
                            element_positions[content] = []
                        element_positions[content].append([row, col])
            
            # Generate facing direction text
            facing_text = {
                (-1, 0): "west", (1, 0): "east", (0, -1): "north", (0, 1): "south"
            }.get(tuple(convert_numpy_types(player_facing)), "unknown")
            
            # Generate compact context
            context_parts = []
            if player_inventory:
                inv_summary = ", ".join([f"{k}:{v}" for k, v in player_inventory.items()])
                context_parts.append(f"Inventory: {inv_summary}")
            
            if surroundings:
                surr_summary = ", ".join([f"{direction}:{content}" for direction, content in surroundings.items()])
                context_parts.append(f"Surroundings: {surr_summary}")
            
            if element_counts:
                elem_summary = ", ".join([f"{elem_type}({count})" for elem_type, count in element_counts.items()])
                context_parts.append(f"Elements: {elem_summary}")
            
            # Create scene summary matching demo format
            scene_summary = {
                "step": step,
                "player": {
                    "position": convert_numpy_types(player_pos),
                    "grid_pos": convert_numpy_types(player_grid_pos),
                    "facing": facing_text,
                    "inventory": player_inventory if player_inventory else "empty"
                },
                "surroundings": surroundings if surroundings else "only grass terrain",
                "visible_elements": {
                    "counts": element_counts if element_counts else "none (only grass)",
                    "positions": {k: compress_coordinates(v) for k, v in element_positions.items()} if element_positions else {}
                },
                "context": "; ".join(context_parts) if context_parts else "Empty grassland"
            }
            
            return convert_numpy_types(scene_summary)
            
        except Exception as e:
            print(f"⚠️ Error generating scene summary JSON: {e}")
            return {
                "step": step,
                "context": "Scene analysis failed",
                "error": str(e)
            }
    
    def _detect_new_objects(self, prev_objects, current_objects):
        """Detect new objects by comparing with previous frame"""
        try:
            prev_ids = {obj.get('id', f"{obj.get('name', 'unknown')}_{obj.get('x', 0)}_{obj.get('y', 0)}") for obj in prev_objects}
            new_objects = []
            
            for obj in current_objects:
                obj_id = obj.get('id', f"{obj.get('name', 'unknown')}_{obj.get('x', 0)}_{obj.get('y', 0)}")
                if obj_id not in prev_ids:
                    new_objects.append(obj)
            
            return new_objects
        except Exception as e:
            print(f"⚠️ Error detecting new objects: {e}")
            return []
    
    def _generate_automatic_skill_for_object(self, obj, observation, state):
        """Generate an automatic skill for interacting with a specific object"""
        try:
            obj_name = obj.get('name', 'unknown')
            obj_type = obj.get('type', 'interactive')
            
            # Create basic interaction skill
            auto_skill = {
                'id': f"auto_{obj_name}_{int(time.time())}",
                'name': f"Auto_{obj_name}_Interaction",
                'description': f"Automatically interact with {obj_name}",
                'operations': [
                    {
                        'operate': 'click',
                        'object_id': obj.get('id'),
                        'coordinate': [obj.get('x', 0), obj.get('y', 0)],
                        'params': {'button': 'left'}
                    }
                ],
                'fitness': 1.0,
                'num': 1,
                'mcts_node_id': None,
                'auto_generated': True
            }
            
            return auto_skill
        except Exception as e:
            print(f"⚠️ Error generating automatic skill for object: {e}")
            return None
    
    def _generate_exploration_skill(self, detected_objects, observation):
        """Generate an exploration skill based on current scene complexity"""
        try:
            # Create exploration skill that moves around to discover more
            auto_skill = {
                'id': f"auto_explore_{int(time.time())}",
                'name': "Auto_Scene_Exploration",
                'description': f"Explore scene with {len(detected_objects)} objects",
                'operations': [
                    {
                        'operate': 'key_press',
                        'params': {'key': 'w'},  # Move forward/up
                    },
                    {
                        'operate': 'key_press', 
                        'params': {'key': 'd'},  # Move right
                    }
                ],
                'fitness': 0.8,
                'num': 1,
                'mcts_node_id': None,
                'auto_generated': True
            }
            
            return auto_skill
        except Exception as e:
            print(f"⚠️ Error generating exploration skill: {e}")
            return None
        
        if len(new_skills) == 0:
            print("❌ No new skills generated")
            return 'ExploreFail'
        else:
            print(f"✅ New skills generated: {len(new_skills)}")
            self.brain.merge_and_save_skills(step, state, skill_clusters, new_skills)
            print(f"📊 skill_clusters num: {len(skill_clusters)}")
            self.brain.long_memory.update_state(state)
            self.logger.log({f"skills/skills_generate_num": len(new_skills)}, step)
            return 'Continue'
    
    def exploit(self, step, task, skill):
        """Exploit existing skill in gym environment"""
        print(f"🎯 Begin exploit with GYM mode")
        self.state_reset(step)
        
        obs = [self.get_observation()]
        print(f"🎯 Selected skill id: {skill['id']} name: {skill['name']} description: {skill['description']} fitness: {skill['fitness']} num: {skill['num']} operations: {skill['operations']}")
        
        skill_fitness = skill['fitness']
        exec_chain = []
        operations = skill['operations']
        
        for operation in operations:
            ob = self.get_observation()
            operation_ = self.operate_grounding(operation, ob)
            
            if operation_ is None:
                print("❌ Operation grounding failed")
                return 'Fail'
            
            # Execute operation in gym environment
            result = self._execute_gym_operation(operation_)
            if result == 'Fail':
                print("❌ Gym operation execution failed")
                return 'Fail'
            
            print("⏳ Wait for operations to finish...")
            time.sleep(self.exec_duration)
        
        obs.append(self.get_observation())
        
        # Check if action was executed successfully
        if not self.eye.detect_acted_cv(obs[-2]['screen'], obs[-1]['screen']):
            print("❌ Action not acted")
            self.logger.log({"eval/skill_acted": 0}, step)
            return 'Fail'
        
        # Generate skill from operation if screen changed significantly
        if len(operations) > 0:
            screen_change_result = self._detect_screen_changes_and_update_interactivity(
                obs, operations[-1], operated_object_id=operations[-1].get('object_id')
            )
            screen_change_ratio = screen_change_result['screen_change_ratio']
            
            if self._is_significant_screen_change(screen_change_ratio):
                print(f"📈 Significant screen change detected (ratio: {screen_change_ratio:.3f}), generating skill")
                temp_state = {'id': f'gym_temp_{step}'}
                temp_node_id = f'gym_node_{step}'
                
                new_skill = self.brain.generate_and_save_skill(
                    step, obs, operations, temp_state['id'], temp_node_id, self
                )
                
                if new_skill:
                    print(f"✅ Successfully generated skill from gym operation: {new_skill.get('name', 'Unknown')}")
        
        # Skill evaluation (similar to BottomUpAgent)
        if not self.close_evaluate:
            time0 = time.time()
            is_consistent, is_progressive = self.brain.skill_evaluate(step, task, obs, skill)
            elapsed_time = time.time() - time0
            print(f"🔍 Skill evaluate elapsed_time: {elapsed_time:.3f}s")
            print(f"📊 is_consistent: {is_consistent}, is_progressive: {is_progressive}")
            
            if is_consistent is not None and is_progressive is not None:
                self.logger.log({"eval/skill_consistent": int(is_consistent), "eval/skill_progressive": int(is_progressive)}, step)
                accumulated_consistent = self.logger.last_value('eval/accumulated_skill_consistent') if self.logger.last_value('eval/accumulated_skill_consistent') is not None else 0
                accumulated_progressive = self.logger.last_value('eval/accumulated_skill_progressive') if self.logger.last_value('eval/accumulated_skill_progressive') is not None else 0
            
                if is_consistent:
                    skill_fitness += 1
                    accumulated_consistent += 1

                if is_progressive:
                    skill_fitness += 1
                    accumulated_progressive += 1

                self.logger.log({"eval/accumulated_skill_consistent": accumulated_consistent, "eval/accumulated_skill_progressive": accumulated_progressive}, step)

                num = skill['num'] + 1
                skill['fitness'] = skill_fitness
                skill['num'] = num

                self.brain.long_memory.update_skill(skill['id'], skill_fitness, num)

                if is_consistent and is_progressive:
                    result = 'Continue'
                else:
                    result = 'Fail'
            else:
                result = 'Continue'
        else:
            result = 'Continue'
        
        self.logger.log({"eval/skill_acted": 1}, step)
        print(f"✅ Skill exploitation result: {result}")
        return result
    
    def exploit_mcp(self, step, task, skill):
        """Enhanced exploit with MCP (Model-based Control Planning) for gym environment"""
        print(f"🎯 Begin exploit with GYM MCP mode")
        self.state_reset(step)
        
        obs = [self.get_observation()]
        print(f"🎯 Selected skill id: {skill['id']} name: {skill['name']} description: {skill['description']} fitness: {skill['fitness']}")
        
        operations = skill['operations']
        
        # Get comprehensive detected objects context (similar to BottomUpAgent)
        historical_objects = []
        try:
            # Get recent objects from long memory for richer MCP context
            recent_objects = self.brain.long_memory.get_recent_objects(limit=50)
            if recent_objects:
                historical_objects = recent_objects
                print(f"Retrieved {len(historical_objects)} historical objects for MCP context")
        except Exception as e:
            print(f"Could not retrieve historical objects: {e}")
        
        # Use comprehensive object detection that includes both new and historical objects
        detected_objects = self.detector.get_detected_objects_with_context(obs[0]['screen'], historical_objects)
        print(f"Detected {len(detected_objects)} objects for MCP decision (including historical context)")
        
        # Use Brain's MCP decision making (same as BottomUpAgent)
        task_context = f"Executing skill '{skill['name']}': {skill['description']}. Choose the best action to achieve this goal."
        
        try:
            print(f"🔄 Using Brain MCP for intelligent decision making...")
            # Get pre_knowledge for the current game
            game_name = getattr(self, 'env_name', 'Crafter')
            pre_knowledge = get_pre_knowledge(game_name)
            
            mcp_result = self.brain.do_operation_mcp(
                step, task_context, obs[0], detected_objects, pre_knowledge=pre_knowledge
            )
            
            if mcp_result and mcp_result.get('action_type') == 'direct_operation':
                # Convert MCP result to gym operation format
                select_operation = {
                    'operate': mcp_result['operate'],
                    'params': mcp_result['params']
                }
                
                # Add object_id if available from MCP result
                if 'object_id' in mcp_result and mcp_result['object_id']:
                    select_operation['object_id'] = mcp_result['object_id']
                    print(f"🤖 Brain MCP selected: {select_operation['operate']} on object {select_operation['object_id']}")
                else:
                    print(f"🤖 Brain MCP selected: {select_operation['operate']} at coordinates")
                    
            else:
                print(f"❌ Brain MCP failed or returned non-operation result: {mcp_result}")
                select_operation = None
                
        except Exception as mcp_error:
            print(f"❌ Brain MCP failed: {mcp_error}")
            print(f"⚠️ Falling back to simple operation selection")
            
            # Fallback to simple selection when Brain MCP fails
            import random
            objects = self.gym_adapter.detect_objects(obs[0]['screen'])
            potential_operations = self.get_potential_operations(objects)
            
            if potential_operations:
                select_operation = random.choice(potential_operations)
                print(f"🎲 Fallback selected: {select_operation.get('operate', 'unknown')}")
            else:
                select_operation = None
                print("❌ No potential operations available for fallback")
        
        # Check if we have a valid operation to execute
        if select_operation:
            # Execute selected operation
            result = self._execute_gym_operation(select_operation)
            if result == 'Fail':
                print("❌ MCP operation execution failed")
                return 'Fail'
            
            obs.append(self.get_observation())
            
            # Enhanced skill generation tracking (similar to BottomUpAgent)
            if hasattr(self, 'successful_operations_history'):
                self.successful_operations_history.append({
                    'step': step,
                    'operation': select_operation,
                    'result': result,
                    'timestamp': time.time()
                })
            
            # Analyze screen changes and generate skill if significant
            screen_change_result = self._detect_screen_changes_and_update_interactivity(
                obs, select_operation, operated_object_id=select_operation.get('object_id')
            )
            
            if self._is_significant_screen_change(screen_change_result['screen_change_ratio']):
                print(f"📈 MCP: Significant screen change detected, generating skill")
                
                # Generate new skill from MCP operation using enhanced method
                if hasattr(self, 'enhanced_skill_generation') and self.enhanced_skill_generation.get('enabled', False):
                    context_info = {
                        'task': task,
                        'screen_change_ratio': screen_change_result['screen_change_ratio'],
                        'detected_objects': detected_objects,
                        'mcp_decision': True
                    }
                    
                    new_skill = self._generate_and_save_skill_from_operations(
                        [select_operation], context_info, step
                    )
                    
                    if new_skill:
                        print(f"✅ Enhanced MCP skill generated: {new_skill.get('name', 'Unknown')}")
                        # Update skill generation history
                        if hasattr(self, 'skill_generation_history'):
                            self.skill_generation_history.append({
                                'step': step,
                                'skill_id': new_skill.get('id'),
                                'generation_method': 'mcp_screen_change',
                                'context': context_info
                            })
                else:
                    # Fallback to original skill generation
                    temp_state = {'id': f'mcp_gym_temp_{step}'}
                    temp_node_id = f'mcp_gym_node_{step}'
                    
                    new_skill = self.brain.generate_and_save_skill(
                        step, obs, [select_operation], temp_state['id'], temp_node_id, self
                    )
                    
                    if new_skill:
                        print(f"✅ Successfully generated MCP skill: {new_skill.get('name', 'Unknown')}")
            
            # Check if we should generate skill from operation sequence
            if (hasattr(self, 'enhanced_skill_generation') and 
                self.enhanced_skill_generation.get('enabled', False) and
                hasattr(self, 'successful_operations_history')):
                
                operation_threshold = self.enhanced_skill_generation.get('operation_sequence_threshold', 3)
                recent_ops = [op for op in self.successful_operations_history 
                             if step - op['step'] <= operation_threshold]
                
                if len(recent_ops) >= operation_threshold:
                    print(f"🔄 MCP: Operation sequence threshold reached, generating composite skill")
                    
                    context_info = {
                        'task': task,
                        'operation_sequence': True,
                        'sequence_length': len(recent_ops),
                        'mcp_decision': True
                    }
                    
                    operations_list = [op['operation'] for op in recent_ops]
                    composite_skill = self._generate_and_save_skill_from_operations(
                        operations_list, context_info, step
                    )
                    
                    if composite_skill:
                        print(f"✅ Composite MCP skill generated: {composite_skill.get('name', 'Unknown')}")
                        # Clear recent operations to avoid duplicate skill generation
                        self.successful_operations_history = [op for op in self.successful_operations_history 
                                                            if step - op['step'] > operation_threshold]
        else:
            print("❌ No operation selected, MCP failed")
            return 'Fail'
        
        # Initialize skill fitness for evaluation
        skill_fitness = skill.get('fitness', 0)
        
        # Skill evaluation (similar to BottomUpAgent)
        if not self.close_evaluate:
            time0 = time.time()
            is_consistent, is_progressive = self.brain.skill_evaluate(step, task, obs, skill)
            elapsed_time = time.time() - time0
            print(f"🔍 Skill evaluate elapsed_time: {elapsed_time:.3f}s")
            print(f"📊 is_consistent: {is_consistent}, is_progressive: {is_progressive}")
            
            if is_consistent is not None and is_progressive is not None:
                self.logger.log({"eval/skill_consistent": int(is_consistent), "eval/skill_progressive": int(is_progressive)}, step)
                accumulated_consistent = self.logger.last_value('eval/accumulated_skill_consistent') if self.logger.last_value('eval/accumulated_skill_consistent') is not None else 0
                accumulated_progressive = self.logger.last_value('eval/accumulated_skill_progressive') if self.logger.last_value('eval/accumulated_skill_progressive') is not None else 0
            
                if is_consistent:
                    skill_fitness += 1
                    accumulated_consistent += 1

                if is_progressive:
                    skill_fitness += 1
                    accumulated_progressive += 1

                self.logger.log({"eval/accumulated_skill_consistent": accumulated_consistent, "eval/accumulated_skill_progressive": accumulated_progressive}, step)

                num = skill['num'] + 1
                skill['fitness'] = skill_fitness
                skill['num'] = num

                self.brain.long_memory.update_skill(skill['id'], skill_fitness, num)

                if is_consistent and is_progressive:
                    result = 'Continue'
                else:
                    result = 'Fail'
            else:
                result = 'Continue'
        else:
            result = 'Continue'
        
        self.logger.log({"eval/skill_acted": 1}, step)
        print(f"✅ Skill exploitation result: {result}")
        return result
    
    def skill_augment(self, step, state, node):
        """
        Unified tool for skill augmentation with screen change detection.
        Encapsulates the core logic of skill_augment as a reusable tool.
        
        Args:
            step: Current step number
            state: Current state dictionary
            node: MCTS node containing operations to execute
            
        Returns:
            tuple: (new_skill, is_state_changed) or (None, False) if failed
        """
        print(f"🔧 Skill augment for gym environment - Step {step}")
        
        # Reset state for clean execution
        self.state_reset(step)
        
        obs = [self.get_observation()]
        operations = getattr(node, 'operations', []).copy() if hasattr(node, 'operations') else []
        
        # Execute existing operations from the node
        for operation in operations:
            print(f"Executing operation: {operation}")
            operation_ = self.operate_grounding(operation, obs[-1])
            if operation_ is None:
                print(f"❌ Operation grounding failed for: {operation}")
                return None, True
            
            # Execute operation in gym environment
            result = self._execute_gym_operation(operation)
            if result == 'Fail':
                print(f"❌ Operation execution failed: {operation}")
                return None, True
                
            print("⏳ Waiting for operation to complete...")
            time.sleep(self.exec_duration if hasattr(self, 'exec_duration') else 0.5)
            obs.append(self.get_observation())

        # Update objects and get potential operations
        existed_object_ids = state.get('object_ids', [])
        existed_objects = self.brain.long_memory.get_object_by_ids(existed_object_ids)
        
        # Detect objects in current screen
        updated_objects = self.gym_adapter.detect_objects(obs[-1]['screen'])
        print(f"🔍 Detected {len(updated_objects)} objects")
        
        # Update objects in long memory
        updated_objects = self.brain.long_memory.update_objects(state, updated_objects)
        potential_operations = self.get_potential_operations(updated_objects)
        
        if not potential_operations:
            print("❌ No potential operations for skill augmentation")
            return None, True
        
        return self._process_skill_augmentation(step, state, node, obs, operations, potential_operations)
    
    def _is_significant_screen_change(self, screen_change_ratio, threshold=0.015):
        """
        Unified function to determine if screen change is significant enough for skill generation.
        
        Args:
            screen_change_ratio: The calculated screen change ratio
            threshold: The threshold for significant change (default: 0.015)
            
        Returns:
            bool: True if change is significant, False otherwise
        """
        return screen_change_ratio > threshold
    
    def _detect_screen_changes_and_update_interactivity(self, states, operation, operated_object_id=None):
        """
        Unified tool for screen change detection and object interactivity updates.
        Can be used across different execution modes (MCP, traditional, skill_augment).
        
        Args:
            states: List of observation states (before and after operation)
            operation: The operation that was executed
            operated_object_id: ID of the operated object (optional)
            
        Returns:
            dict: Contains screen_change_ratio, is_state_changed, and interactivity info
        """
        # Detect screen changes
        screen_change_ratio = self.eye.detect_acted_cv(states[-2]['screen'], states[-1]['screen'])
        is_state_changed, sim_2_states = self.brain.detect_state_changed(states[-2], states[-1])
        print(f"screen_change_ratio: {screen_change_ratio} Sim between 2 states: {sim_2_states}")
        
        interactivity = None
        
        # Update object interactivity if operation and object info available
        if operation and 'operate' in operation:
            operation_type = operation.get('operate')
            if operation_type in ['Click', 'Touch']:
                # Determine interactivity based on screen changes
                if is_state_changed:
                    if operation_type == 'Click':
                        interactivity = 'click_window_change'
                    else:  # Touch
                        interactivity = 'touch_popup'  # Touch usually doesn't cause window switch but may have large popups
                elif self._is_significant_screen_change(screen_change_ratio):
                    # Detail changes but no major state change, likely popup interaction
                    if operation_type == 'Click':
                        interactivity = 'click_popup'
                    else:  # Touch
                        interactivity = 'touch_popup'
                else:
                    # No significant changes
                    interactivity = 'no_effect'
                
                print(f"Operation interactivity: {interactivity}")
                
                # Update object interactivity if object_id is available
                if operated_object_id:
                    self.brain.long_memory.update_object_interactivity(
                        operated_object_id, operation_type, interactivity, 
                        screen_change_ratio, is_state_changed
                    )
        
        return {
            'screen_change_ratio': screen_change_ratio,
            'is_state_changed': is_state_changed,
            'sim_2_states': sim_2_states,
            'interactivity': interactivity
        }
    
    def _analyze_screen_changes_and_generate_skill(self, step, obs, operations, select_operation, 
                                                 operated_object_id, state, new_mcts_node):
        """
        Unified method for screen change analysis and skill generation.
        This tool can be reused across different execution modes.
        """
        # Use unified screen change detection
        screen_change_result = self._detect_screen_changes_and_update_interactivity(
            obs, select_operation, operated_object_id=operated_object_id
        )
        
        screen_change_ratio = screen_change_result['screen_change_ratio']
        is_state_changed = screen_change_result['is_state_changed']
        
        # Generate new skill based on screen changes
        if self._is_significant_screen_change(screen_change_ratio):
            new_skill = self.brain.generate_and_save_skill(
                step, obs, operations, state['id'], new_mcts_node.node_id, self
            )

            if self.brain.detect_state_changed(obs[0], obs[-1])[0]:
                print("State changed")
                new_mcts_node.is_fixed = True
                return new_skill, True
            else:
                return new_skill, False
        else:
            operations[-1].pop('params', None)
            print("Operation not acted")
            return None, False
     
    def _ground_object_to_coordinates(self, operation, state):
        """Ground object_id to screen coordinates using image matching."""
        # Check if object_id is None or 'None'
        if operation['object_id'] is None or operation['object_id'] == 'None':
            print(f"Operation grounding failed: object_id is None for operation {operation}")
            return None
            
        object_img = self.brain.long_memory.get_object_image_by_id(operation['object_id'])
        if object_img is None:
            print(f"Operation grounding failed: No image found for object_id {operation['object_id']}")
            return None
            
        # For gym environments, we might need to adapt the grounding approach
        try:
            from utils.utils import image_grounding_v3
            grounding_result, score = image_grounding_v3(state['screen'], object_img)
            if grounding_result:
                operation['params'] = {'x': grounding_result[0], 'y': grounding_result[1]}
                return grounding_result[0], grounding_result[1]
            else:
                print(f"Operation grounding failed: {operation['object_id']} score: {score}")
                return None
        except Exception as e:
            print(f"Error in object grounding: {e}")
            return None
    
    def _extract_coordinates_from_params(self, params):
        """Extract x, y coordinates from params, handling both coordinate and x,y formats."""
        if 'coordinate' in params:
            return params['coordinate']
        else:
            return params['x'], params['y']
    
    def _process_coordinate_based_operation(self, operation, state, operation_method):
        """Process operations that require x, y coordinates (Click, RightSingle, LeftDouble)."""
        if 'params' not in operation:
            # Need to ground object_id to coordinates
            coordinates = self._ground_object_to_coordinates(operation, state)
            if coordinates is None:
                return None
            x, y = coordinates
        else:
            # Extract coordinates from existing params
            x, y = self._extract_coordinates_from_params(operation['params'])
        
        return operation_method(x, y)
    
    def _process_drag_operation(self, operation):
        """Process Drag operation which requires x1, y1, x2, y2 coordinates."""
        if 'params' not in operation:
            print(f"Operation grounding failed: Drag operation requires params but none provided: {operation}")
            return None
        
        params = operation['params']
        if all(key in params for key in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = params['x1'], params['y1'], params['x2'], params['y2']
            from BottomUpAgent.UnifiedOperation import UnifiedOperation
            return UnifiedOperation.Drag(x1, y1, x2, y2)
        else:
            print(f"Operation grounding failed: Drag operation missing required coordinates (x1, y1, x2, y2): {params}")
            return None
    
    def _process_direct_operation(self, operation, operation_method):
        """Process operations that don't require coordinates (Type, Wait, etc.)."""
        if 'params' in operation:
            params = operation['params']
            # Handle different parameter formats for different operations
            if 'text' in params:  # Type operation
                return operation_method(params['text'])
            elif 'duration' in params:  # Wait operation
                return operation_method(params['duration'])
            elif 'keys' in params:  # Hotkey operation
                return operation_method(params['keys'])
            elif 'direction' in params and 'clicks' in params:  # Scroll operation
                return operation_method(params['direction'], params['clicks'])
            else:
                # For operations without specific params, call with no arguments
                return operation_method()
        else:
            return operation_method()
    
    def operate_grounding(self, operation, state):
        """Ground operation to UnifiedOperation object based on operation type."""
        operate_type = operation['operate']
        
        # Define operation mappings for coordinate-based operations
        coordinate_operations = {
            'Click': UnifiedOperation.Click,
            'RightSingle': UnifiedOperation.RightSingle,
            'LeftDouble': UnifiedOperation.LeftDouble
        }
        
        # Define operation mappings for direct operations (no coordinates needed)
        direct_operations = {
            'Type': UnifiedOperation.Type,
            'Wait': UnifiedOperation.Wait,
            'Finished': UnifiedOperation.Finished,
            'CallUser': UnifiedOperation.CallUser,
            'Hotkey': UnifiedOperation.Hotkey,
            'Scroll': UnifiedOperation.Scroll,
            'LongPress': UnifiedOperation.LongPress,
            'PressBack': UnifiedOperation.PressBack,
            'PressHome': UnifiedOperation.PressHome,
            'PressEnter': UnifiedOperation.PressEnter
        }
        
        # Handle coordinate-based operations
        if operate_type in coordinate_operations:
            return self._process_coordinate_based_operation(operation, state, coordinate_operations[operate_type])
        
        # Handle Drag operation (special case with 4 coordinates)
        elif operate_type == 'Drag':
            return self._process_drag_operation(operation)
        
        # Handle direct operations
        elif operate_type in direct_operations:
            return self._process_direct_operation(operation, direct_operations[operate_type])
        
        # Unsupported operation
        else:
            print(f"Unsupported operate: {operate_type}")
            return None
     
    def _select_operation_with_mcp(self, candidate_operations, step, obs, operations):
        """
        使用MCP模式智能选择操作 - 参考BottomUpAgent实现，增加异常处理和超时保护
        
        Args:
            candidate_operations: candidate operations
            step: current step
            obs: observation history
            operations: current operation list
            
        Returns:
            selected operations
        """
        # Use MCP mode for intelligent operation selection
        print(f"Using MCP mode for operation selection from {len(candidate_operations)} candidates")
        
        # Get comprehensive detected objects context (similar to exploit_mcp)
        # Get historical objects from recent states for comprehensive context
        historical_objects = []
        try:
            # Check if brain and long_memory are available
            if hasattr(self, 'brain') and self.brain and hasattr(self.brain, 'long_memory') and self.brain.long_memory:
                # Get objects from the current state's object_ids if available
                current_state = obs[-1].get('state', {})
                if 'object_ids' in current_state and current_state['object_ids']:
                    # Get more historical objects for richer MCP context
                    historical_objects = self.brain.long_memory.get_object_by_ids(current_state['object_ids'][-50:])
                    print(f"Retrieved {len(historical_objects)} historical objects from state")
                    
                # Also try to get recent objects from long memory if state doesn't have enough
                if len(historical_objects) < 30:
                    try:
                        recent_objects = self.brain.long_memory.get_recent_objects(limit=50)
                        if recent_objects:
                            # Merge with existing historical objects, avoiding duplicates
                            existing_ids = {obj.get('id') for obj in historical_objects if obj.get('id')}
                            for obj in recent_objects:
                                if obj.get('id') and obj['id'] not in existing_ids:
                                    historical_objects.append(obj)
                            print(f"Added {len(recent_objects)} recent objects, total historical: {len(historical_objects)}")
                    except Exception as e2:
                        print(f"Could not retrieve recent objects: {e2}")
            else:
                print("Brain or long_memory not available, using empty historical objects")
        except Exception as e:
            print(f"Could not retrieve historical objects: {e}")
        
        # Use comprehensive object detection that includes both new and historical objects
        detected_objects = []
        try:
            if hasattr(self, 'detector') and self.detector is not None:
                detected_objects = self.detector.get_detected_objects_with_context(obs[-1]['screen'], historical_objects)
                print(f"Detected {len(detected_objects)} objects for MCP operation selection (including historical context)")
            else:
                # Fallback to basic object detection for gym environment
                detected_objects = self._get_basic_detected_objects(obs[-1])
                print(f"Using basic object detection: {len(detected_objects)} objects")
        except Exception as e:
            print(f"Error in object detection: {e}")
            detected_objects = []
        
        # Use simplified decision making to avoid GUI blocking
        print(f"🔄 Using simplified operation selection to avoid GUI blocking...")
        
        try:
            # Simple rule-based operation selection without complex API calls
            import random
            
            if not candidate_operations:
                print("❌ No candidate operations available")
                return None
            
            # Prioritize operations based on detected objects
            if detected_objects:
                # If objects detected, prioritize interaction operations
                interaction_ops = [op for op in candidate_operations 
                                 if op.get('operate', '').lower() in ['collect', 'place', 'make', 'do']]
                if interaction_ops:
                    selected_op = random.choice(interaction_ops)
                    print(f"🤖 Simple MCP (objects detected): {selected_op.get('operate', 'unknown')}")
                    return selected_op
            
            # If no objects or no interaction ops, prefer exploration operations
            exploration_ops = [op for op in candidate_operations 
                             if op.get('operate', '').lower() in ['move', 'go', 'walk', 'turn']]
            if exploration_ops:
                selected_op = random.choice(exploration_ops)
                print(f"🤖 Simple MCP (exploration): {selected_op.get('operate', 'unknown')}")
                return selected_op
            
            # Final fallback to any available operation
            selected_op = random.choice(candidate_operations)
            print(f"🎲 Simple MCP (random): {selected_op.get('operate', 'unknown')}")
            return selected_op
            
        except Exception as selection_error:
            print(f"❌ Simplified selection failed: {selection_error}")
            
            # Final fallback
            if candidate_operations:
                import random
                selected_op = random.choice(candidate_operations)
                print(f"🎲 Error fallback: {selected_op.get('operate', 'unknown')}")
                return selected_op
            else:
                print("❌ No candidate operations available for fallback")
                return None

    
    def select_skill(self, skills, skill_cluster, suspended_skill_ids, close_exploration=False):
        """Select skill using UCT-based selection with temperature scaling"""
        if len(skills) == 0:
            print("No potential skill, explore")
            return {'name': 'Explore', 'mcts_node_id': None}, False
        else:
            num_total = skill_cluster['explore_nums']
            candidate_skills = []
            skills_info = []
            max_fitness = -5
            max_id = -1
            
            for i, skill in enumerate(skills):
                if skill['fitness'] > max_fitness:
                    max_fitness = skill['fitness']
                    max_id = i
                if skill['id'] not in suspended_skill_ids:
                    num_total += skill['num']
                    candidate_skills.append(skill)
                else:
                    skills_info.append({'id': skill['id'], 'name': skill['name'], 
                                         'num': skill['num'], 'fitness': skill['fitness'], 'prob': 0})

            if max_id != -1:
                mcts_node_id = skills[max_id]['mcts_node_id']
            else:
                mcts_node_id = None

            if len(candidate_skills) == 0:
                skills_info.append({'id':'Explore', 'name': 'Explore', 'prob': 100.00})     
                return {'name': 'Explore', 'mcts_node_id': mcts_node_id}, True

            ucts = []
            for skill in candidate_skills:
                # Add safety check to prevent division by zero and NaN values
                skill_num = max(skill['num'], 1)  # Ensure num is at least 1
                if num_total <= 0:
                    num_total = 1  # Ensure num_total is positive
                uct = skill['fitness'] + self.brain.uct_c * np.sqrt(np.log(num_total) / skill_num)
                # Check for NaN or infinite values
                if np.isnan(uct) or np.isinf(uct):
                    uct = skill['fitness']  # Fallback to fitness value only
                ucts.append(uct)

            if not close_exploration:
                # Add safety check for exploration UCT calculation
                explore_nums = max(skill_cluster['explore_nums'], 1)  # Ensure explore_nums is at least 1
                explore_uct = self.brain.uct_threshold + self.brain.uct_c * np.sqrt(np.log(num_total) / explore_nums)
                # Check for NaN or infinite values
                if np.isnan(explore_uct) or np.isinf(explore_uct):
                    explore_uct = self.brain.uct_threshold  # Fallback to threshold only
                ucts.append(explore_uct)
                candidate_skills.append({'id': 'Explore', 'name': 'Explore', 'num': skill_cluster['explore_nums'], 
                                         'fitness': self.brain.uct_threshold, 'mcts_node_id': mcts_node_id})
            
            ucts = np.array(ucts)
            # Additional safety checks for UCT array
            ucts = np.nan_to_num(ucts, nan=0.0, posinf=1e6, neginf=-1e6)
            
            temperature = max(self.brain.temperature(num_total), 1e-2)
            scaled_ucts = ucts / temperature
            scaled_ucts -= np.max(scaled_ucts)  
            exp_ucts = np.exp(scaled_ucts)
            exp_ucts = np.clip(exp_ucts, 1e-10, None)
            
            # Ensure exp_ucts doesn't contain NaN or inf values
            exp_ucts = np.nan_to_num(exp_ucts, nan=1e-10, posinf=1e6, neginf=1e-10)
            
            sum_exp_ucts = np.sum(exp_ucts)
            if sum_exp_ucts == 0 or np.isnan(sum_exp_ucts) or np.isinf(sum_exp_ucts):
                # Fallback to uniform distribution
                probs = np.ones(len(exp_ucts)) / len(exp_ucts)
            else:
                probs = exp_ucts / sum_exp_ucts
            
            # Final safety check for probabilities
            probs = np.nan_to_num(probs, nan=1.0/len(probs), posinf=1.0, neginf=0.0)
            probs = probs / np.sum(probs)  # Renormalize to ensure sum equals 1

            print(f"ucts: {ucts}, exp_ucts: {exp_ucts}, num: {num_total}, temp: {temperature}")
            print(f"probs: {probs}")
            
            for i, skill in enumerate(candidate_skills):
                skills_info.append({'id': skill['id'], 'name': skill['name'], 
                                         'num': skill['num'], 'fitness': skill['fitness'], 'prob': round(probs[i],2)})
                  
            selected_skill = np.random.choice(candidate_skills, p=probs)

            if probs[-1] > 0.9:
                return selected_skill, True
            else:
                return selected_skill, False

    def _process_skill_augmentation(self, step, state, node, observations, operations, potential_operations):
        """Process skill augmentation with Brain integration"""
        if not self.skill_generation_enabled or len(operations) < self.skill_generation_threshold:
            return None, True
        
        try:
            # Get skill clusters for current state
            skill_clusters = self.long_memory.get_skill_clusters_by_state(state['state_feature'])
            
            # Generate and save skill using Brain
            new_skill = self.brain.generate_and_save_skill(
                step, observations, operations, state['id'], getattr(node, 'id', None), agent=self
            )
            
            if new_skill:
                self.new_skills.append(new_skill)
                print(f"✅ Generated new skill: {new_skill['name']}")
                
                # Merge and save skills when we have enough
                if len(self.new_skills) >= 3:  # Batch process skills
                    self.brain.merge_and_save_skills(step, state, skill_clusters, self.new_skills)
                    print(f"🔄 Merged {len(self.new_skills)} skills into clusters")
                    self.new_skills = []  # Reset after processing
                
                return new_skill, False
            else:
                print("❌ Failed to generate skill")
                return None, True
                
        except Exception as e:
            print(f"❌ Error in skill augmentation: {e}")
            return None, True
    
    def _execute_gym_operation(self, operation):
        """Execute operation in gym environment"""
        try:
            # Map operation to gym action
            action = self.gym_adapter.map_operation_to_action(operation)
            
            # Execute action in environment
            obs, reward, done, info = self.gym_adapter.step(action)
            
            # Check if episode is done
            if done:
                episode_reward = info.get('episode_reward', 0.0)
                print(f"🏆 Episode finished with reward: {episode_reward:.2f}")
                return 'Done'
            
            return 'Success'
            
        except Exception as e:
            print(f"❌ Error executing gym operation: {e}")
            return 'Fail'
    
    def start_crafter_interactive_with_detection(self, max_steps=None):
        """        
        Note: This method uses GridContentChecker for specialized detection comparison.
        This is separate from the main scene summary functionality and serves as
        a debugging/analysis tool for detection accuracy.
        """
        if self.env_name.lower() != 'crafter':
            print(f"❌ Interactive detection mode only supports Crafter environment, current: {self.env_name}")
            return None
            
        if max_steps is None:
            # Get max_steps from config file
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
            
        # Import GridContentChecker for specialized detection analysis
        from demos.demo_grid_content_check import GridContentChecker
        import threading
        import time
        
        print("🚀 Starting Crafter Interactive Detection Mode")
        print("=" * 60)
        print("🎮 Keyboard/Mouse Control + BottomUp Agent Analysis")
        print("🔍 Real-time comparison between GUI and detection system")
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
                    print(f"\n📊 Step {step_count} - Detection Accuracy: {comparison['accuracy']:.1f}%")
                    print(f"   Matches: {comparison['matches']}, Mismatches: {comparison['mismatches']}")
                    
                    # 轻量级对象检测分析（移除MCP调用以防止阻塞）
                    try:
                        objects = self.gym_adapter.detect_objects(current_obs['screen'])
                        if objects:
                            print(f"🔍 Detected {len(objects)} objects")
                            # Show info for first 3 objects
                            for i, obj in enumerate(objects[:3]):
                                print(f"   Object{i+1}: {obj.get('id', 'unknown')} at {obj.get('center', 'unknown')}")
                            # Note: 移除MCP调用以防止GUI阻塞
                    except Exception as e:
                        print(f"⚠️ Object detection error: {e}")
                
                episode_stats['steps'] = step_count
                episode_stats['detection_accuracy'] = comparison['accuracy']
                
                time.sleep(0.5)  # 控制检查频率
                
        except KeyboardInterrupt:
            print("\n⚠️ User interrupted")
        finally:
            # 停止GUI进程
            checker.stop_gui_process()
            print("🛑 Interactive detection mode ended")
        
        return episode_stats
    
    def start_crafter_interactive_launcher(self, max_steps=None, resolution='medium', no_gui=False):
        """启动纯Crafter交互式游戏模式（直接调用launcher）"""
        if self.env_name.lower() != 'crafter':
            print(f"❌ Crafter interactive mode only supports Crafter environment, current: {self.env_name}")
            return None
            
        if max_steps is None:
            # Get max_steps from config file
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
            
        from demos.crafter_interactive_launcher import demo_crafter_interactive
        
        print("🚀 Starting Crafter Interactive Game Mode")
        print("=" * 60)
        print("🎮 Pure Gaming Experience - Keyboard Control + Achievement Tracking")
        print("📋 WASD move, SPACE interact, TAB sleep, 1-6 craft tools")
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
            
            print("🏁 Crafter interactive game ended")
            return result
            
        except Exception as e:
            print(f"❌ Failed to start Crafter interactive mode: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def start_hybrid_crafter_mode(self, max_steps=1000, analysis_interval=5):
        """启动混合Crafter模式：用户控制 + Agent分析建议"""
        if self.env_name.lower() != 'crafter':
            print(f"❌ Hybrid mode only supports Crafter environment, current: {self.env_name}")
            return None
            
        import pygame
        import threading
        import time
        from demos.crafter_interactive_launcher import get_gui_config
        
        print("🚀 Starting Hybrid Crafter Mode")
        print("=" * 60)
        print("🎮 User Keyboard Control + 🤖 Agent Smart Suggestions")
        print("🔄 Agent will periodically analyze game state and provide suggestions")
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
                
                print(f"✅ Hybrid mode GUI started ({window_size[0]}x{window_size[1]})")
                
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
                                    
                                    print(f"\n🎮 User action: {action}, reward: {reward:.2f}, total steps: {shared_state['steps']}")
                                    
                                    # 定期进行Agent分析
                                    if (shared_state['steps'] - shared_state['last_analysis_step']) >= analysis_interval:
                                        try:
                                            state = self.get_observation()
                                            objects = self.gym_adapter.detect_objects(state['screen'])
                                            if objects:
                                                # Get pre_knowledge for the current game
                                                game_name = getattr(self, 'env_name', 'Crafter')
                                                pre_knowledge = get_pre_knowledge(game_name)
                                                
                                                result = self.brain.do_operation_mcp(
                                                    shared_state['steps'], 
                                                    "探索世界并收集资源制作工具", 
                                                    state, 
                                                    objects,
                                                    pre_knowledge=pre_knowledge
                                                )
                                                if result and 'operation' in result:
                                                    print(f"💡 Agent suggestion: {result['operation']}")
                                                    if 'reasoning' in result:
                                                        print(f"🤔 Analysis reason: {result['reasoning'][:100]}...")
                                        except Exception as e:
                                            print(f"⚠️ Agent analysis error: {e}")
                                        
                                        shared_state['last_analysis_step'] = shared_state['steps']
                                    
                                    if shared_state['done']:
                                        print(f"🏁 Game over! Total reward: {shared_state['total_reward']:.2f}")
                                        running = False
                    
                    # 渲染游戏
                    try:
                        self.gym_adapter._render_environment()
                        pygame.display.flip()
                    except Exception as render_error:
                        print(f"⚠️ Render error: {render_error}")
                    
                    clock.tick(30)
                
                pygame.quit()
                self.gui_running = False
                print("🛑 Hybrid mode GUI stopped")
                
            except Exception as e:
                print(f"❌ GUI worker thread error: {e}")
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
            print("\n⚠️ User interrupted")
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
        
        print(f"\n🏁 Hybrid mode ended: {shared_state['steps']} steps, {shared_state['total_reward']:.2f} total reward")
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
                                    
                                    # Simple step logging (removed heavy MCP analysis to prevent blocking)
                                    print(f"\n--- Step {self.game_step_count} ---")
                                    print(f"🎮 User Action: {action}, Reward: {reward:.2f}")
                                    
                                    # Optional: Light analysis every 10 steps to avoid blocking
                                    if self.game_step_count % 10 == 0:
                                        try:
                                            state = self.get_observation()
                                            objects = self.gym_adapter.detect_objects(state['screen'])
                                            print(f"🔍 Detected {len(objects)} objects in current state")
                                            # Note: Removed MCP call to prevent GUI blocking
                                        except Exception as e:
                                            print(f"⚠️ Light analysis error: {e}")
                                    
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
        
        print(f"\n🏁 Parallel GUI session finished: {self.game_step_count} steps, {shared_state['total_reward']:.2f} total reward")
        
        episode_stats = {
            'steps': self.game_step_count,
            'total_reward': shared_state['total_reward'],
            'done': shared_state['done']
        }
        
        print(f"\n🏁 Interactive session finished: {shared_state['steps']} steps, {shared_state['total_reward']:.2f} total reward")
        return episode_stats
    
    def run_parallel_interactive(self, max_steps=None):
        """Run parallel interactive mode with both user input and MCP decision control - Simplified like test_parallel_gui.py"""
        if max_steps is None:
            step_settings = self.config.get('gym', {}).get('step_settings', {})
            max_steps = step_settings.get('max_total_steps', 30000)
            
        import pygame
        import threading
        import time
        import queue
        from demos.crafter_interactive_launcher import get_gui_config
        
        print("🚀 Starting Parallel Interactive Mode (User + MCP Control)")
        print("=" * 60)
        print("This will start an interactive GUI window with parallel control")
        print("🎮 Use WASD to move, SPACE to interact, ESC to close")
        print("🤖 MCP Agent will also make decisions and inject actions")
        print("📡 Both inputs will be processed in parallel")
        print("⚡ Press 'M' to toggle MCP mode, 'U' to toggle user priority")
        print()
        
        # Simple initialization like test_parallel_gui.py
        self.gui_running = True  # Set to True before starting threads
        self.game_step_count = 0
        
        # Shared state variables
        shared_state = {
            'total_reward': 0,
            'steps': 0,
            'done': False
        }
        
        # Action queue for parallel input processing
        action_queue = queue.Queue(maxsize=10)
        
        # Control flags
        control_flags = {
            'mcp_enabled': True,
            'user_priority': True,  # User input has priority over MCP
            'mcp_interval': 3,      # MCP makes decision every N seconds
            'last_mcp_time': 0
        }
        
        def mcp_decision_worker():
            """Intelligent MCP decision making thread using Brain and detection"""
            try:
                print("🤖 MCP worker thread started")
                while self.gui_running and not shared_state['done']:
                    current_time = time.time()
                    
                    # Check if it's time for MCP to make a decision
                    if (control_flags['mcp_enabled'] and 
                        current_time - control_flags['last_mcp_time'] >= control_flags['mcp_interval']):
                        
                        try:
                            print("🤖 MCP decision time reached, making intelligent decision...")
                            # Get current observation for intelligent decision making
                            obs = self.get_observation()
                            if not obs:
                                print("⚠️ No observation available, skipping MCP decision")
                                time.sleep(0.5)
                                continue
                                
                            # Get detected objects for MCP context
                            detected_objects = []
                            try:
                                if hasattr(self, 'detector') and self.detector:
                                    detected_objects = self.detector.get_detected_objects(obs['screen'])
                            except Exception as e:
                                print(f"⚠️ Object detection failed: {e}")
                            
                            # Prepare state for MCP call
                            state = {
                                'screen': obs['screen'],
                                'inventory': obs.get('inventory', {})
                            }
                            
                            # Define current task based on dynamic planning
                            task = self._plan_current_task(obs, shared_state['steps'])
                            
                            # Call Brain's MCP method for real LLM decision making
                            try:
                                # Get max_iterations from config, default to 6 for Crafter
                                max_iter = getattr(self.brain, 'max_mcp_iter', 6)
                                
                                # Get game-specific pre_knowledge for MCP guidance
                                game_name = getattr(self.gym_adapter.env, 'spec', None)
                                game_name = game_name.id if game_name else 'crafter'
                                pre_knowledge = get_pre_knowledge(game_name)
                                
                                mcp_result = self.brain.do_operation_mcp(
                                    step=shared_state['steps'],
                                    task=task,
                                    state=state,
                                    detected_objects=detected_objects,
                                    pre_knowledge=pre_knowledge,
                                    max_iterations=max_iter
                                )
                                
                                if mcp_result:
                                    print(f"✅ MCP Decision: {mcp_result.get('action_type', 'unknown')}")
                                    
                                    # Convert MCP result to Crafter action
                                    mcp_action = self._convert_mcp_result_to_action(mcp_result)
                                    if mcp_action is not None:
                                        if not action_queue.full():
                                            action_queue.put(('mcp', mcp_action, f"MCP: {mcp_result.get('action_type', 'unknown')}"))
                                            print(f"🎯 MCP Action Queued: {mcp_action}")
                                else:
                                    print("⚠️ MCP returned no result, using fallback")
                                    # Fallback to simple exploration
                                    import random
                                    fallback_action = random.choice([1, 2, 3, 4, 5])  # Move or interact
                                    if not action_queue.full():
                                        action_queue.put(('mcp', fallback_action, "MCP Fallback"))
                                        
                            except Exception as mcp_error:
                                print(f"❌ MCP call failed: {mcp_error}")
                                # Fallback to simple random action
                                import random
                                fallback_action = random.choice([1, 2, 3, 4, 5])
                                if not action_queue.full():
                                    action_queue.put(('mcp', fallback_action, "MCP Error Fallback"))
                            
                            # Get inventory and game state for logging
                            inventory = {}
                            game_state = {}
                            
                            # Get inventory from Crafter environment
                            try:
                                if hasattr(self.gym_adapter, 'env') and hasattr(self.gym_adapter.env, '_player'):
                                    player_inventory = self.gym_adapter.env._player.inventory
                                    # Only include non-zero items
                                    inventory = {item: count for item, count in player_inventory.items() if count > 0}
                                elif hasattr(self.gym_adapter, 'env') and hasattr(self.gym_adapter.env, 'unwrapped'):
                                    if hasattr(self.gym_adapter.env.unwrapped, '_player'):
                                        player_inventory = self.gym_adapter.env.unwrapped._player.inventory
                                        inventory = {item: count for item, count in player_inventory.items() if count > 0}
                            except Exception as e:
                                print(f"⚠️ Could not get inventory: {e}")
                                inventory = obs.get('inventory', {})
                            
                            # Get game state info
                            if hasattr(self.gym_adapter, 'get_info'):
                                game_state = self.gym_adapter.get_info() or {}
                            
                            # Generate comprehensive scene summary for MCP (replaces verbose grid printing)
                            scene_summary = self.brain._generate_scene_summary(detected_objects, inventory, game_state)
                            
                            print("\n=== Compact JSON Scene Summary for MCP Tools ===")
                            print(f"\n{scene_summary}\n")
                            
                            # Generate and print JSON scene summary for MCP tools
                            # try:
                            #     scene_json = self._generate_scene_summary_json(shared_state['steps'], obs, detected_objects, {'inventory': inventory, 'game_state': game_state})
                            #     print("\n=== JSON Scene Summary for MCP Tools ===")
                            #     import json
                            #     print(json.dumps(scene_json, indent=2, ensure_ascii=False))
                            #     print("=== End JSON Scene Summary ===")
                            # except Exception as json_error:
                            #     print(f"⚠️ Error generating JSON scene summary: {json_error}")
                            
                            # Update MCP timing after successful decision
                            control_flags['last_mcp_time'] = current_time
                                
                        except Exception as e:
                            print(f"⚠️ MCP decision error: {e}")
                            # Fallback to basic action on error
                            import random
                            action = random.choice([1, 2, 3, 4, 5])
                            if not action_queue.full():
                                action_queue.put(('mcp', action, "error_fallback"))
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
                    
            except Exception as e:
                print(f"❌ MCP worker error: {e}")
        
        def gui_worker():
            # GUI logic has been moved to main thread
            # This thread is no longer needed for GUI operations
            # GUI worker thread started (GUI moved to main thread)
            
            # Just wait for the main thread to finish
            try:
                while self.gui_running and not shared_state['done']:
                    time.sleep(1.0)
                
            except Exception as e:
                print(f"❌ GUI worker error: {e}")
                import traceback
                traceback.print_exc()
                self.gui_running = False
        
        # Start GUI thread first, then wait for initialization
        gui_thread = threading.Thread(target=gui_worker, daemon=True)
        gui_thread.start()
        
        print("🚀 GUI thread started, waiting for initialization...")
        
        # Wait for GUI to be ready before starting MCP
        time.sleep(2.0)  # Give GUI time to initialize
        
        # Use SharedEnvironment approach to ensure proper environment synchronization
        # Using SharedEnvironment approach for proper synchronization
        
        try:
            # Initialize pygame in main thread
            pygame.init()
            gui_config = get_gui_config(self.config)
            window_size = (gui_config['width'], gui_config['height'])
            screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption(f"Parallel Crafter - GymAgent [{window_size[0]}x{window_size[1]}]")
            clock = pygame.time.Clock()
            
            # Use existing environment from gym_adapter instead of creating new one
            if hasattr(self, 'gym_adapter') and self.gym_adapter.env is not None:
                env = self.gym_adapter.env
                print("✅ Using existing environment from gym_adapter")
                
                # Ensure environment is properly reset and ready
                try:
                    obs = env.reset()
                    self.gym_adapter.current_obs = obs
                    print("✅ Environment reset and synchronized")
                except Exception as reset_error:
                    print(f"⚠️ Environment reset error: {reset_error}")
                    # Fallback: create new environment if reset fails
                    import crafter
                    env = crafter.Env()
                    obs = env.reset()
                    self.gym_adapter.env = env
                    self.gym_adapter.current_obs = obs
                    print("🔄 Created new environment as fallback")
            else:
                # Fallback: create new environment if gym_adapter doesn't have one
                import crafter
                env = crafter.Env()
                obs = env.reset()
                if hasattr(self, 'gym_adapter'):
                    self.gym_adapter.env = env
                    self.gym_adapter.current_obs = obs
                print("🔄 Created new environment (gym_adapter had no existing env)")
            
            # Ensure detector.crafter_env points to the same environment instance
            # Critical: Synchronize BOTH self.gym_adapter.detector AND self.detector
            detectors_to_sync = []
            
            # Add gym_adapter detector if available
            if (hasattr(self, 'gym_adapter') and 
                hasattr(self.gym_adapter, 'detector') and 
                self.gym_adapter.detector and 
                hasattr(self.gym_adapter.detector, 'detector_type') and
                self.gym_adapter.detector.detector_type == 'crafter_api'):
                detectors_to_sync.append(('gym_adapter.detector', self.gym_adapter.detector))
            
            # Add GymAgent detector if available (used by MCP thread)
            if (hasattr(self, 'detector') and 
                self.detector and 
                hasattr(self.detector, 'detector_type') and
                self.detector.detector_type == 'crafter_api'):
                detectors_to_sync.append(('self.detector', self.detector))
            
            # Synchronize all detectors
            if detectors_to_sync:
                for detector_name, detector in detectors_to_sync:
                    detector.crafter_env = env
                    print(f"✅ Synchronized {detector_name}.crafter_env with GUI environment")
                    
                    # Verify synchronization
                    if detector.crafter_env is env:
                        print(f"✅ Environment synchronization verified: {detector_name} uses same instance")
                    else:
                        print(f"❌ Environment synchronization failed for {detector_name}: different instances detected")
            else:
                print("⚠️ No crafter_api detectors found to synchronize")
            
            # Render a few frames to ensure the environment is ready
            for i in range(3):
                frame = env.render(size=window_size)
                if frame is not None:
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(frame_surface, (0, 0))
                    pygame.display.flip()
                    time.sleep(0.1)
            
            print(f"✅ Parallel GUI started in main thread ({window_size[0]}x{window_size[1]})")
            print("🎮 User: WASD+SPACE, MCP: Auto-decisions, ESC: Exit")
            print("⚡ Press 'M' to toggle MCP mode, 'U' to toggle user priority")
            
            # Now start MCP thread after GUI is ready
            mcp_thread = threading.Thread(target=mcp_decision_worker, daemon=True)
            mcp_thread.start()
            print("🚀 MCP thread started after GUI initialization")
            
            # Main GUI loop - exactly like test_parallel_gui.py
            running = True
            self.gui_running = True  # Ensure GUI is running
            while running and not shared_state['done']:
                # Handle pygame events (user input)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_m:  # Toggle MCP mode
                            control_flags['mcp_enabled'] = not control_flags['mcp_enabled']
                            status = "enabled" if control_flags['mcp_enabled'] else "disabled"
                            print(f"🤖 MCP mode {status}")
                        elif event.key == pygame.K_u:  # Toggle user priority
                            control_flags['user_priority'] = not control_flags['user_priority']
                            priority = "user" if control_flags['user_priority'] else "MCP"
                            print(f"⚡ Priority mode: {priority}")
                        else:
                            # Handle game actions from user
                            user_action = self._handle_keyboard_input(event.key)
                            if user_action is not None:
                                if not action_queue.full():
                                    action_queue.put(('user', user_action, f'key_{event.key}'))
                                    print(f"🎮 User input: {pygame.key.name(event.key)} -> action {user_action}")
                
                # Process actions from queue
                current_action = None
                action_source = None
                action_desc = None
                
                try:
                    if not action_queue.empty():
                        action_source, current_action, action_desc = action_queue.get_nowait()
                        
                        # Execute action
                        obs, reward, done, info = env.step(current_action)
                        shared_state['total_reward'] += reward
                        shared_state['steps'] += 1
                        shared_state['done'] = done
                        
                        print(f"🎯 {action_source.upper()}: {action_desc} -> reward: {reward:.3f}, total: {shared_state['total_reward']:.3f}")
                        
                        if done:
                            print(f"🏁 Episode finished! Total reward: {shared_state['total_reward']:.3f}")
                            running = False
                            
                except queue.Empty:
                    pass
                except Exception as action_error:
                    print(f"⚠️ Action execution error: {action_error}")
                
                # Render the game
                try:
                    frame = env.render(size=window_size)
                    if frame is not None:
                        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                        screen.blit(frame_surface, (0, 0))
                        pygame.display.flip()
                except Exception as render_error:
                    print(f"⚠️ Render error: {render_error}")
                
                clock.tick(30)  # 30 FPS
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            self.gui_running = False
        finally:
            self.gui_running = False
            if gui_thread and gui_thread.is_alive():
                gui_thread.join(timeout=2.0)
            if mcp_thread.is_alive():
                mcp_thread.join(timeout=2.0)
        
        # Collect achievements from environment for Crafter score calculation
        achievements = {}
        try:
            if hasattr(env, '_player') and hasattr(env._player, 'achievements'):
                achievements = dict(env._player.achievements)
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_player'):
                achievements = dict(env.unwrapped._player.achievements)
            elif hasattr(env, '_get_info'):
                info = env._get_info()
                achievements = info.get('achievements', {})
        except Exception as e:
            print(f"⚠️ Could not collect achievements: {e}")
            achievements = {}
        
        episode_stats = {
            'steps': shared_state['steps'],
            'total_reward': shared_state['total_reward'],
            'done': shared_state['done'],
            'achievements': achievements
        }
        
        # Calculate and display Crafter score like crafter_interactive_launcher.py
        self._print_crafter_score_summary([achievements], 1, shared_state['total_reward'], shared_state['steps'])
        
        return episode_stats
    

    def _plan_current_task(self, obs: Dict[str, Any], current_step: int) -> str:
        """
        Dynamically plan the current task based on game state and configuration.
        
        Args:
            obs: Current observation from the environment
            current_step: Current step number
            
        Returns:
            str: The planned task description
        """
        try:
            # Get task planning configuration
            task_config = self.config.get('task_planning', {})
            if not task_config.get('enabled', False):
                # Fallback to basic exploration if task planning is disabled
                return "Explore the environment and collect useful resources"
            
            # Get current achievement progress for context
            achievement_progress = self._get_current_achievement_progress()
            
            # Use adaptive planning based on current context
            if task_config.get('adaptive_planning', {}).get('enabled', False):
                context_task = self._get_adaptive_context_task(obs, achievement_progress, task_config)
                if context_task:
                    return context_task
            
            # Fallback to configured fallback tasks
            fallback_tasks = task_config.get('adaptive_planning', {}).get('fallback_tasks', {})
            return fallback_tasks.get('exploration', "Explore the environment and discover new opportunities")
            
        except Exception as e:
            print(f"⚠️ Task planning error: {e}")
            return "Explore the environment and collect useful resources"
    
    def _get_current_achievement_progress(self) -> set:
        """
        Get the set of currently completed achievements.
        
        Returns:
            set: Set of completed achievement names
        """
        try:
            if hasattr(self.gym_adapter, 'env') and hasattr(self.gym_adapter.env, 'achievements'):
                # Get achievements from Crafter environment
                achievements = self.gym_adapter.env.achievements
                return set(achievements.keys()) if achievements else set()
            return set()
        except Exception as e:
            print(f"⚠️ Error getting achievement progress: {e}")
            return set()
    

    
    def _get_adaptive_context_task(self, obs: Dict, achievement_progress: set, 
                                  task_config: Dict) -> Optional[str]:
        """
        Get an adaptive task based on current game state and context.
        
        Args:
            obs: Current observation
            achievement_progress: Set of completed achievements
            task_config: Task planning configuration
            
        Returns:
            Optional[str]: Context-aware task or None
        """
        try:
            adaptive_config = task_config.get('adaptive_planning', {})
            context_factors = adaptive_config.get('context_factors', [])
            
            # Analyze current context
            context = self._analyze_current_context(obs, achievement_progress, context_factors)
            
            # Get task generation settings
            task_generation = task_config.get('task_generation', {})
            
            # Priority-based task selection based on context
            if context.get('health_low', False) and 'health_status' in context_factors:
                return "Find food or shelter to restore health"
            elif context.get('inventory_full', False) and 'current_inventory' in context_factors:
                return "Use or drop items to make inventory space"
            elif context.get('night_approaching', False) and 'time_of_day' in context_factors:
                return "Prepare for night: find shelter or light source"
            elif context.get('resources_needed', False) and 'current_inventory' in context_factors:
                return "Gather essential resources for survival and crafting"
            elif context.get('exploration_needed', False) and 'visible_objects' in context_factors:
                return "Explore new areas to discover resources and opportunities"
            
            # Generate dynamic task based on achievement progress
            if task_generation.get('dynamic_generation', False):
                return self._generate_dynamic_task(achievement_progress, context, task_generation)
                
            return None
            
        except Exception as e:
            print(f"⚠️ Adaptive task selection error: {e}")
            return None
    
    def _analyze_current_context(self, obs: Dict, achievement_progress: set, 
                                context_factors: List[str]) -> Dict:
        """
        Analyze the current game context.
        
        Args:
            obs: Current observation
            achievement_progress: Set of completed achievements
            context_factors: List of context factors to analyze
            
        Returns:
            Dict: Context analysis results
        """
        context = {}
        
        try:
            # Analyze health status
            if 'health_status' in context_factors:
                health = obs.get('health', 100)
                hunger = obs.get('hunger', 100)
                context['health_low'] = health < 30 or hunger < 30
                
            # Analyze inventory status
            if 'current_inventory' in context_factors:
                inventory = obs.get('inventory', {})
                context['inventory'] = inventory
                max_items = 10  # Typical inventory limit
                current_items = sum(inventory.values()) if inventory else 0
                context['inventory_full'] = current_items >= max_items * 0.8
                
                # Check for resource needs
                basic_resources = ['wood', 'stone', 'food']
                context['resources_needed'] = any(inventory.get(res, 0) < 5 for res in basic_resources)
                
            # Analyze time/day cycle
            if 'time_of_day' in context_factors:
                # This would need game-specific implementation
                context['night_approaching'] = False  # Placeholder
                
            # Analyze visible objects and exploration needs
            if 'visible_objects' in context_factors:
                context['visible_objects'] = obs.get('detected_objects', [])
                # Simple heuristic for exploration need
                context['exploration_needed'] = len(achievement_progress) < 3
            
            if 'achievement_progress' in context_factors:
                context['achievements'] = achievement_progress
            
        except Exception as e:
            print(f"⚠️ Context analysis error: {e}")
        
        return context
    
    def _generate_dynamic_task(self, achievement_progress: set, context: Dict, 
                              task_generation: Dict) -> str:
        """
        Generate a dynamic task based on current progress, context, and screen analysis.
        Uses LLM to intelligently analyze the current game state and generate appropriate tasks.
        
        Args:
            achievement_progress: Set of completed achievements
            context: Current context analysis
            task_generation: Task generation configuration
            
        Returns:
            str: Generated task description
        """
        try:
            # Try intelligent task generation using Brain's LLM capabilities
            if hasattr(self, 'brain') and self.brain is not None:
                try:
                    # Get current observation for screen analysis
                    current_obs = self.get_observation()
                    detected_objects = self.gym_adapter.detect_objects(current_obs['screen'])
                    
                    # Build intelligent task generation prompt
                    task_prompt = self._build_intelligent_task_prompt(
                        achievement_progress, context, detected_objects, current_obs
                    )
                    
                    # Use Brain's LLM to generate task
                    from utils.utils import cv_to_base64
                    screen_b64 = cv_to_base64(current_obs['screen'])
                    
                    response = self.brain.base_model.call_text_images(
                        task_prompt, [screen_b64], []
                    )
                    
                    if response and 'text' in response:
                        generated_task = self._parse_generated_task(response['text'])
                        if generated_task:
                            print(f"🧠 LLM Generated Task: {generated_task}")
                            return generated_task
                            
                except Exception as e:
                    print(f"⚠️ LLM task generation failed: {e}")
            
            # Fallback to enhanced rule-based generation
            return self._generate_rule_based_task(achievement_progress, context, task_generation)
                
        except Exception as e:
            print(f"⚠️ Dynamic task generation error: {e}")
            return "Continue exploring and improving your situation"
    
    def _build_intelligent_task_prompt(self, achievement_progress, context, detected_objects, current_obs):
        """
        Build an intelligent prompt for LLM-based task generation
        """
        # Get inventory and game state info
        inventory_info = current_obs.get('inventory', {})
        game_state = current_obs.get('game_state', {})
        
        # Format detected objects
        objects_desc = ", ".join([obj.get('name', 'unknown') for obj in detected_objects[:10]])
        
        prompt = f"""You are an intelligent game assistant analyzing the current Crafter game state to generate the next optimal task.

Current Game Analysis:
- Completed Achievements: {len(achievement_progress)} total
- Context Factors: {context}
- Visible Objects: {objects_desc}
- Current Inventory: {inventory_info}
- Game State: {game_state}

Based on the current screen and game state, generate ONE specific, actionable task that would be most beneficial right now.

Consider:
1. Immediate survival needs (health, food, shelter)
2. Resource gathering opportunities visible on screen
3. Crafting progression based on current inventory
4. Environmental threats or opportunities
5. Achievement progression potential

Respond with ONLY a concise task description (1-2 sentences), no explanations.

Example responses:
- "Collect the wood visible nearby to build basic tools"
- "Find food immediately - health is critically low"
- "Craft a pickaxe using available wood and stone"
- "Explore the cave entrance to find rare resources"

Task:"""
        
        return prompt
    
    def _parse_generated_task(self, response_text):
        """
        Parse and clean the LLM-generated task response
        """
        try:
            # Clean up the response
            task = response_text.strip()
            
            # Remove common prefixes/suffixes
            prefixes_to_remove = ['Task:', 'Generated task:', 'Next task:', 'Objective:']
            for prefix in prefixes_to_remove:
                if task.lower().startswith(prefix.lower()):
                    task = task[len(prefix):].strip()
            
            # Ensure reasonable length
            if len(task) > 200:
                task = task[:200] + "..."
            
            # Ensure it's not empty and meaningful
            if len(task) > 10 and not task.lower().startswith('i '):
                return task
                
        except Exception as e:
            print(f"Error parsing generated task: {e}")
        
        return None
    
    def _generate_rule_based_task(self, achievement_progress, context, task_generation):
        """
        Enhanced rule-based task generation as fallback
        """
        progress_count = len(achievement_progress)
        
        # Priority-based task selection
        if context.get('health_low', False):
            return "Find food or healing items immediately - health is critically low"
        
        if context.get('night_approaching', False):
            return "Prepare for night: find shelter or create light sources"
        
        if context.get('inventory_full', False):
            return "Use or drop items to make space, then craft useful tools"
        
        if context.get('resources_needed', False):
            return "Gather essential resources: wood, stone, and food"
        
        # Progress-based tasks with more variety
        if progress_count == 0:
            return "Start survival basics: collect wood and stone from the environment"
        elif progress_count < 3:
            return "Build essential tools: craft a pickaxe and sword for better resource gathering"
        elif progress_count < 7:
            return "Establish a base: create a crafting table and furnace for advanced items"
        elif progress_count < 12:
            return "Advance your equipment: craft better tools and explore dangerous areas"
        else:
            return "Master advanced challenges: complete complex achievements and explore fully"
    


    def _print_crafter_score_summary(self, achievements_history, episode_count, total_reward, total_steps):
        """Print Crafter score summary like crafter_interactive_launcher.py"""
        import warnings
        
        # Crafter achievements list (22 total achievements)
        CRAFTER_ACHIEVEMENTS = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        if not achievements_history:
            print("\n📊 No episodes completed yet.")
            return
        
        # Calculate success rates
        success_rates = {}
        for achievement in CRAFTER_ACHIEVEMENTS:
            success_count = sum(1 for episode_achievements in achievements_history 
                              if episode_achievements.get(achievement, 0) > 0)
            success_rates[achievement] = (success_count / episode_count) * 100
        
        # Calculate Crafter score (geometric mean, offset by 1%)
        rates = [success_rates[name] for name in CRAFTER_ACHIEVEMENTS]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Geometric mean: exp(mean(log(1 + rates))) - 1
            log_rates = [np.log(1 + rate) for rate in rates if rate >= 0]
            if log_rates:
                crafter_score = np.exp(np.mean(log_rates)) - 1
            else:
                crafter_score = 0.0
        
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
                left_name = self._format_achievement_name(left[0])
                left_display = f"{left_name} ({left[1]:.0f}%, x{left[2]})"
                
                if i + 1 < len(unlocked_achievements):
                    right = unlocked_achievements[i + 1]
                    right_name = self._format_achievement_name(right[0])
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
                unlock_names = [self._format_achievement_name(name) for name, _ in new_unlocks]
                print(f"🆕 LATEST: {', '.join(unlock_names[:5])}{'...' if len(unlock_names) > 5 else ''}")
            else:
                print("\n🆕 LATEST EPISODE: No achievements unlocked")
        
        print("="*70)
    
    def _format_achievement_name(self, achievement):
        """Format achievement name to readable form"""
        name = achievement.replace('achievement_', '').replace('_', ' ').title()
        return name
    
    def _convert_mcp_to_action(self, operation_description):
        """Convert MCP operation description to Crafter action number"""
        if not operation_description:
            return None
            
        operation_lower = operation_description.lower()
        
        # Map common MCP operations to Crafter actions
        if 'move up' in operation_lower or 'go up' in operation_lower:
            return 3
        elif 'move down' in operation_lower or 'go down' in operation_lower:
            return 4
        elif 'move left' in operation_lower or 'go left' in operation_lower:
            return 1
        elif 'move right' in operation_lower or 'go right' in operation_lower:
            return 2
        elif 'collect' in operation_lower or 'interact' in operation_lower or 'attack' in operation_lower:
            return 5
        elif 'sleep' in operation_lower:
            return 6
        elif 'place stone' in operation_lower:
            return 7
        elif 'place table' in operation_lower:
            return 8
        elif 'place furnace' in operation_lower:
            return 9
        elif 'place plant' in operation_lower:
            return 10
        elif 'wood pickaxe' in operation_lower:
            return 11
        elif 'stone pickaxe' in operation_lower:
            return 12
        elif 'iron pickaxe' in operation_lower:
            return 13
        elif 'wood sword' in operation_lower:
            return 14
        elif 'stone sword' in operation_lower:
            return 15
        elif 'iron sword' in operation_lower:
            return 16
        else:
            # Default to interact/collect if unclear
            return 5
    
    def _convert_mcp_result_to_action(self, mcp_result):
        """
        Convert MCP result from Brain.do_operation_mcp to Crafter action index
        """
        try:
            if not mcp_result:
                return None
                
            action_type = mcp_result.get('action_type', '').lower()
            
            # Handle direct_operation type - get the actual operation from 'operate' field
            if action_type == 'direct_operation':
                operate = mcp_result.get('operate', '').lower()
                print(f"🎯 Converting direct operation: {operate}")
                
                # Map Crafter-specific operations to action indices
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
                
                # Try direct mapping first
                if operate in crafter_operation_mapping:
                    return crafter_operation_mapping[operate]
                
                # Handle variations and partial matches
                if 'move' in operate:
                    if 'up' in operate:
                        return 3
                    elif 'down' in operate:
                        return 4
                    elif 'left' in operate:
                        return 1
                    elif 'right' in operate:
                        return 2
                
                if any(word in operate for word in ['interact', 'collect', 'attack', 'do']):
                    return 5  # Do action
                
                if 'sleep' in operate:
                    return 6
                
                if 'place' in operate:
                    if 'stone' in operate:
                        return 7
                    elif 'table' in operate:
                        return 8
                    elif 'furnace' in operate:
                        return 9
                    elif 'plant' in operate or 'sapling' in operate:
                        return 10
                
                if 'craft' in operate or 'make' in operate:
                    if 'wood' in operate and 'pickaxe' in operate:
                        return 11
                    elif 'stone' in operate and 'pickaxe' in operate:
                        return 12
                    elif 'iron' in operate and 'pickaxe' in operate:
                        return 13
                    elif 'wood' in operate and 'sword' in operate:
                        return 14
                    elif 'stone' in operate and 'sword' in operate:
                        return 15
                    elif 'iron' in operate and 'sword' in operate:
                        return 16
                
                print(f"⚠️ Unknown direct operation: {operate}, defaulting to interact")
                return 5  # Default to interact
            
            # Handle other action types (legacy support)
            action_mapping = {
                'noop': 0,
                'move_left': 1,
                'move_right': 2, 
                'move_up': 3,
                'move_down': 4,
                'do': 5,
                'sleep': 6,
                'place_stone': 7,
                'place_table': 8,
                'place_furnace': 9,
                'place_plant': 10,
                'make_wood_pickaxe': 11,
                'make_stone_pickaxe': 12,
                'make_iron_pickaxe': 13,
                'make_wood_sword': 14,
                'make_stone_sword': 15,
                'make_iron_sword': 16
            }
            
            # Try direct mapping first
            if action_type in action_mapping:
                return action_mapping[action_type]
            
            # Handle common variations
            if 'move' in action_type:
                if 'left' in action_type:
                    return 1
                elif 'right' in action_type:
                    return 2
                elif 'up' in action_type:
                    return 3
                elif 'down' in action_type:
                    return 4
            
            if 'interact' in action_type or 'collect' in action_type or 'do' in action_type:
                return 5  # Do action
            
            if 'craft' in action_type or 'make' in action_type:
                if 'wood_pickaxe' in action_type:
                    return 11
                elif 'wood_sword' in action_type:
                    return 14
                elif 'stone_pickaxe' in action_type:
                    return 12
                elif 'stone_sword' in action_type:
                    return 15
            
            if 'place' in action_type:
                if 'stone' in action_type:
                    return 7
                elif 'table' in action_type:
                    return 8
                elif 'furnace' in action_type:
                    return 9
            
            # Default fallback
            print(f"⚠️ Unknown MCP action type: {action_type}, using interact action")
            return 5  # Default to interact instead of random movement
            
        except Exception as e:
            print(f"❌ Error converting MCP result to action: {e}")
            return 5  # Safe fallback to interact
    
    def _handle_keyboard_input(self, key):
        """Convert pygame key to Crafter action"""
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
    
    # Note: Training and evaluation methods removed as BottomUp Agent
    # uses exploration-based learning rather than traditional RL training
    
    def _perform_skill_evolution(self, step, state):
        """
        Perform skill evolution: remove low-fitness skills and optimize skill clusters
        Inspired by BottomUpAgent's skill_evolution method
        """
        try:
            if not hasattr(self.brain, 'skill_evolution'):
                return
            
            # Perform skill evolution using Brain's method
            evolution_result = self.brain.skill_evolution(
                fitness_threshold=self.skill_fitness_threshold,
                observation_threshold=self.skill_observation_threshold
            )
            
            if evolution_result:
                self.logger.info(f"Step {step}: Skill evolution completed - {evolution_result}")
                
                # Track evolution in history
                self.skill_generation_history.append({
                    'step': step,
                    'type': 'evolution',
                    'result': evolution_result,
                    'timestamp': time.time()
                })
            
        except Exception as e:
            self.logger.error(f"Error during skill evolution at step {step}: {e}")
    
    def _generate_and_save_skill_from_operations(self, operations, context_info, step):
        """
        Generate and save a skill from a sequence of operations
        Inspired by BottomUpAgent's generate_and_save_skill method
        """
        try:
            if not hasattr(self.brain, 'generate_and_save_skill'):
                return None
            
            # Build operation string for skill generation
            operation_str = " -> ".join([str(op) for op in operations[-self.skill_generation_threshold:]])
            
            # Generate skill using Brain's method
            skill_result = self.brain.generate_and_save_skill(
                operations=operations[-self.skill_generation_threshold:],
                context=context_info,
                step=step
            )
            
            if skill_result:
                self.logger.info(f"Step {step}: Generated new skill from operations: {operation_str}")
                
                # Track skill generation in history
                self.skill_generation_history.append({
                    'step': step,
                    'type': 'generation',
                    'operations': operation_str,
                    'skill_result': skill_result,
                    'timestamp': time.time()
                })
                
                return skill_result
            
        except Exception as e:
            self.logger.error(f"Error generating skill from operations at step {step}: {e}")
        
        return None
    
    def _analyze_screen_changes_and_generate_skill(self, step, obs_before, obs_after, operations):
        """
        Analyze screen changes and generate skills based on successful operations
        Inspired by BottomUpAgent's _analyze_screen_changes_and_generate_skill method
        """
        try:
            if not hasattr(self.brain, 'detect_state_changed'):
                return None
            
            # Detect if screen state has changed significantly
            state_changed = self.brain.detect_state_changed(
                obs_before['screen'], obs_after['screen']
            )
            
            if state_changed and len(operations) >= self.skill_generation_threshold:
                print(f"📈 Step {step}: Significant screen change detected, generating skill")
                
                # Create temporary state for skill generation
                temp_state = {'id': f'gym_skill_gen_{step}'}
                temp_node_id = f'gym_node_{step}'
                
                # Generate skill from successful operations
                new_skill = self.brain.generate_and_save_skill(
                    step, obs_after, operations, temp_state['id'], temp_node_id, self
                )
                
                if new_skill:
                    print(f"✅ Successfully generated skill: {new_skill.get('name', 'Unknown')}")
                    
                    # Track in skill generation history
                    self.skill_generation_history.append({
                        'step': step,
                        'type': 'screen_change_generation',
                        'operations': [str(op) for op in operations],
                        'skill_name': new_skill.get('name', 'Unknown'),
                        'timestamp': time.time()
                    })
                    
                    return new_skill
            
        except Exception as e:
            self.logger.error(f"Error analyzing screen changes at step {step}: {e}")
        
        return None
    
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
    parser.add_argument('--mode', choices=['demo', 'interactive', 'crafter_launcher', 'crafter_detection', 'hybrid', 'parallel_interactive'], default='demo', help='Run mode')
    parser.add_argument('--resolution', default='medium', help='GUI resolution for Crafter modes (tiny/small/low/medium/high/ultra)')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps for interactive modes')
    parser.add_argument('--analysis_interval', type=int, default=5, help='Steps between Agent analysis in hybrid mode')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI (background mode)')
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_gym_agent(args.env, args.config)
    
    try:
        if args.mode == 'interactive':
            print("🚀 Starting standard interactive mode...")
            stats = agent.run_interactive(max_steps=args.max_steps)
            print(f"Interactive mode ended: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
        elif args.mode == 'crafter_launcher':
            print("🎮 Starting Crafter interactive game mode...")
            stats = agent.start_crafter_interactive_launcher(
                max_steps=args.max_steps,
                resolution=args.resolution,
                no_gui=args.no_gui
            )
            if stats:
                print(f"Crafter game ended: {stats}")
        elif args.mode == 'crafter_detection':
            print("🔍 Starting Crafter detection analysis mode...")
            stats = agent.start_crafter_interactive_with_detection(max_steps=args.max_steps)
            if stats:
                print(f"Detection mode ended: accuracy {stats['detection_accuracy']:.1f}%, {stats['steps']} steps")
        elif args.mode == 'hybrid':
            print("🤝 Starting hybrid mode (user control + agent suggestions)...")
            stats = agent.start_hybrid_crafter_mode(
                max_steps=args.max_steps,
                analysis_interval=args.analysis_interval
            )
            if stats:
                print(f"Hybrid mode ended: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
        elif args.mode == 'parallel_interactive':
            print("🚀 Starting parallel interactive mode (user + MCP parallel control)...")
            stats = agent.run_parallel_interactive(max_steps=args.max_steps)
            if stats:
                print(f"Parallel interactive mode ended: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
        else:  # demo
            print(f"Running demo for {args.episodes} episodes...")
            for i in range(args.episodes):
                stats = agent.run_episode()
                print(f"Episode {i+1}: {stats['steps']} steps, {stats['total_reward']:.2f} reward")
    
    finally:
        agent.close()