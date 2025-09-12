import time
from .Logger import Logger
from .Eye import Eye
from .Hand import Hand
from .Brain import Brain
from .Detector import Detector
from .Teacher import Teacher
from .Mcts import MCTS
from utils.utils import image_grounding, cv_to_base64, operations_to_str, image_grounding_v3
from .UnifiedOperation import UnifiedOperation
from .pre_knowledge import get_pre_knowledge
import numpy as np
from pynput import keyboard as pkb
from .visualizer import push_data, data_init
from warnings import warn

from concurrent.futures import ThreadPoolExecutor, as_completed

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warn(f"Call to deprecated function {func.__name__}.",
             category=DeprecationWarning,
             stacklevel=2)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

class BottomUpAgent:
    def __init__(self, config):
        self.game_name = config['game_name']

        self.logger = Logger(config['project_name'], config['game_name'] + ' - ' + config['run_name'], backend='wandb')
        self.eye = Eye(config)
        self.hand = Hand(config)
        self.detector = Detector(config)
        self.teacher = Teacher(config)
        self.brain = Brain(config, self.detector, self.logger)
        self.close_explore = config['close_explore']
        self.close_evaluate = config['close_evaluate'] if 'close_evaluate' in config else False
        self.close_reset = config['close_reset'] if 'close_reset' in config else True

        self.operates = config['operates'] if 'operates' in config else ['Click']
        self.max_operation_length = config['max_operation_length'] if 'max_operation_length' in config else 2
        self.is_base = config['is_base'] if 'is_base' in config else False
        self.use_mcp = config['use_mcp'] if 'use_mcp' in config else False
        self.exec_duration = config['exec_duration'] if 'exec_duration' in config else 0.8

        self.suspended_skill_cluster_ids = []
        
        # Add state tracking for intelligent reset
        self.last_reset_step = -1
        self.current_turn_detected = False
        self.game_over_detected = False
        
        # Context management for hint information
        self.context_manager = {
            'current_hint': None,
            'hint_history': [],
            'max_hint_history': 5  # Keep last 5 hints for context
        }

        print(f"GameAgent initialized")
        print(f"game_name: {self.game_name}")
        print(f"MCP mode: {'Enabled' if self.use_mcp else 'Disabled'}")
        print(f"Base mode: {'Enabled' if self.is_base else 'Disabled'}")
        print(f"Exec duration: {self.exec_duration}s")

    def get_observation(self, include_hint=None):
        screen = self.eye.get_screenshot_cv()
        state_feature = self.detector.encode_image(screen)
        observation = {"state_feature": state_feature, "screen": screen}
        
        # Add hint information if provided
        if include_hint:
            observation["hint"] = include_hint
            
        return observation
    
    def store_hint(self, hint_info):
        """Store hint information in context manager"""
        if hint_info:
            self.context_manager['current_hint'] = hint_info
            self.context_manager['hint_history'].append({
                'hint': hint_info,
                'timestamp': time.time()
            })
            
            # Keep only the most recent hints
            if len(self.context_manager['hint_history']) > self.context_manager['max_hint_history']:
                self.context_manager['hint_history'].pop(0)
            
            print(f"Stored hint: {hint_info}")
    
    def get_current_hint(self):
        """Get the current hint information"""
        return self.context_manager.get('current_hint')
    
    def get_hint_context(self):
        """Get hint context for MCP decision making"""
        context = []
        if self.context_manager['current_hint']:
            context.append(f"Current hint: {self.context_manager['current_hint']}")
        
        if self.context_manager['hint_history']:
            recent_hints = self.context_manager['hint_history'][-3:]  # Last 3 hints
            for i, hint_entry in enumerate(recent_hints):
                context.append(f"Previous hint {i+1}: {hint_entry['hint']}")
        
        return "\n".join(context) if context else None
    
    def clear_current_hint(self):
        """Clear current hint after successful action"""
        self.context_manager['current_hint'] = None
    
    def get_potential_operations(self, existed_objects):
        # use for train action
        operations = []
        for operate in self.operates:
            if operate == 'Click':
                for object in existed_objects:
                    operations.append({'operate': operate, 'object_id': object['id'], 'params': {'x': object['center'][0], 'y': object['center'][1]}})
            else:
                print(f"Unsupported operate: {operate}")
                continue
        return operations
    
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
            
        grounding_result, score = image_grounding_v3(state['screen'], object_img)
        if grounding_result:
            operation['params'] = {'x': grounding_result[0], 'y': grounding_result[1]}
            return grounding_result[0], grounding_result[1]
        else:
            print(f"Operation grounding failed: {operation['object_id']} score: {score}")
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


    
    def run_step(self, step, task):
        self.logger.log({"step": step}, step)
        # get screen 
        ob = self.get_observation()

        state = self.brain.long_memory.get_state(ob)
        if state is None:
            print("No state found, create state")
            mcts = MCTS()

            state = {
                'id': None,
                'state_feature': ob['state_feature'],
                'object_ids': [],
                'mcts': mcts,
                'skill_clusters': [],
            }
            state['id'] = self.brain.long_memory.save_state(state)
            

        skill_clusters_ids = state['skill_clusters']
        skill_clusters = self.brain.long_memory.get_skill_clusters_by_ids(skill_clusters_ids)

        if len(skill_clusters) == 0:
            skill_cluster = None
            skills = []
        else:
            skill_cluster_id = self.brain.select_skill_cluster(step, ob, skill_clusters)
            if skill_cluster_id is None:
                return 'Continue'   
            
            skill_cluster = self.brain.long_memory.get_skill_clusters_by_id(skill_cluster_id)
            skills = self.brain.long_memory.get_skills_by_ids(skill_cluster['members'])
            print(f"selected skill_cluster id: {skill_cluster['id']} name: {skill_cluster['name']} description: {skill_cluster['description']}")
            try:
                from .visualizer import push_data
                push_data({'skill_goal': {'id': skill_cluster['id'], "name": skill_cluster['name'], "description": skill_cluster['description']}})
            except ImportError:
                pass  # Visualizer not available

        result = 'Retry'
        suspended_skill_ids = []
        while result == 'Retry':
            skill, suspend_flag = self.brain.select_skill(skills, skill_cluster, suspended_skill_ids, self.close_explore)
            if skill['name'] == 'Explore':
                self.logger.log({"decision": 0, "decision_text": "Explore"}, step)
                try:
                    from .visualizer import push_data
                    push_data({'decision': "Explore"})
                except ImportError:
                    pass  # Visualizer not available

                result = self.explore(step, state, skill, skill_clusters)
                if skill_cluster is not None:
                    self.brain.long_memory.update_skill_cluster_explore_nums(skill_cluster['id'], skill_cluster['explore_nums'] + 1)
                break
            else:
                self.logger.log({"decision": 1, "decision_text": "Exploit"}, step)
                try:
                    from .visualizer import push_data
                    push_data({'decision': "Exploit"})
                except ImportError:
                    pass  # Visualizer not available
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
                self.suspended_skill_cluster_ids.append(skill_cluster['id'])
            elif result == 'Continue':
                self.suspended_skill_cluster_ids.clear()
            self.state_reset()

        self.brain.skill_evolution(step, skills, skill_cluster) 
        
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
        self.state_reset(step)
        
        obs = [self.get_observation()]
        operations = node.operations.copy()
        
        # Execute existing operations from the node
        for operation in operations:
            print(f"operation: {operation}")
            operation_ = self.operate_grounding(operation, obs[-1])
            if operation_ is None:
                return None, True
            self.hand.do_operation(operation_, self.eye.left, self.eye.top)
            print("wait for operations to finish......")
            time.sleep(self.exec_duration)
            obs.append(self.get_observation())

        # Update objects and get potential operations
        existed_object_ids = state['object_ids']
        existed_objects = self.brain.long_memory.get_object_by_ids(existed_object_ids)
        updated_objects = self.detector.update_objects(obs[-1]['screen'], existed_objects)
        print(f"detected objects nums: {len(updated_objects)}")
        updated_objects = self.brain.long_memory.update_objects(state, updated_objects)
        potential_operations = self.get_potential_operations(updated_objects)
        
        return self._process_skill_augmentation(step, state, node, obs, operations, potential_operations)
    
    def _select_operation_with_mcp(self, candidate_operations, step, obs, operations):
        """
        使用MCP模式智能选择操作
        
        Args:
            candidate_operations: candidate operations
            step: current step
            obs: observation history
            operations: curren operation list
            
        Returns:
            selected operations
        """
        # Use MCP mode for intelligent operation selection
        print(f"Using MCP mode for operation selection from {len(candidate_operations)} candidates")
        
        # Get comprehensive detected objects context (similar to exploit_mcp)
        # Get historical objects from recent states for comprehensive context
        historical_objects = []
        try:
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
        except Exception as e:
            print(f"Could not retrieve historical objects: {e}")
        
        # Use comprehensive object detection that includes both new and historical objects
        detected_objects = self.detector.get_detected_objects_with_context(obs[-1]['screen'], historical_objects)
        print(f"Detected {len(detected_objects)} objects for MCP operation selection (including historical context)")
        
        task_context = f"Exploring new operations from node with {len(operations)} existing operations. Choose the best operation to continue exploration from {len(candidate_operations)} candidates."
        mcp_result = self.brain.do_operation_mcp(
            step, task_context, obs[-1], detected_objects
        )
        
        if mcp_result and mcp_result.get('action_type') == 'direct_operation':
            # Convert MCP result back to operation format
            select_operation = {
                'operate': mcp_result['operate'],
                'params': mcp_result['params']
            }
            
            # First priority: use object_id directly from MCP result if available
            if 'object_id' in mcp_result:
                select_operation['object_id'] = mcp_result['object_id']
                print(f"MCP provided object_id: {select_operation['object_id']}")
            # Second priority: handle fallback decision with selected_object info
            elif mcp_result.get('fallback_decision') and 'selected_object' in mcp_result:
                selected_obj = mcp_result['selected_object']
                select_operation['object_id'] = selected_obj.get('id', 'None')
                print(f"MCP fallback selected object ID: {select_operation['object_id']}")
            else:
                # Third priority: try to match with candidate operations to get object_id
                mcp_x, mcp_y = None, None
                if 'coordinate' in mcp_result['params']:
                    mcp_x, mcp_y = mcp_result['params']['coordinate']
                else:
                    mcp_x, mcp_y = mcp_result['params'].get('x'), mcp_result['params'].get('y')
                
                for candidate in candidate_operations:
                    candidate_x = candidate.get('params', {}).get('x')
                    candidate_y = candidate.get('params', {}).get('y')
                    if candidate_x == mcp_x and candidate_y == mcp_y:
                        select_operation['object_id'] = candidate.get('object_id')
                        break
                
                # Ensure object_id is set even if no match found
                if 'object_id' not in select_operation:
                    select_operation['object_id'] = 'None'
            
            print(f"MCP selected operation: {select_operation}")
            return select_operation
        else:
            # Fallback to teacher guidance if MCP fails
            print("MCP operation selection failed, falling back to teacher guidance")
            return self.teacher.get_operation_guidance(candidate_operations)
    
    def _process_skill_augmentation(self, step, state, node, obs, operations, potential_operations):
        """
        Process skill augmentation with screen change detection and object interactivity updates.
        
        This method contains the core logic for:
        1. Filtering candidate operations
        2. Selecting and executing new operations
        3. Detecting screen changes
        4. Updating object interactivity
        5. Generating new skills based on changes
        """

        if len(potential_operations) == 0:
            print("No potential operations found")
            node.is_fixed = True
            return None, False

        # Filter candidate operations
        candidate_operations = []
        existed_children_operations = state['mcts'].get_children_operations(node)    
        for operation in potential_operations:
            existed_flag = False   
            for existed_operation in existed_children_operations:
                # Handle cases where object_id might not exist
                op_object_id = operation.get('object_id', 'None')
                existed_op_object_id = existed_operation.get('object_id', 'None')
                if (operation['operate'] == existed_operation['operate']) and (op_object_id == existed_op_object_id):
                    existed_flag = True
                    break
            if not existed_flag:
                candidate_operations.append(operation)
            
        if len(candidate_operations) == 0:
            print("No candidate operations found after skill")
            node.is_fixed = True
            return None, False

        # Select and execute new operation
        if self.use_mcp:
            select_operation = self._select_operation_with_mcp(candidate_operations, step, obs, operations)
        else:
            # Use traditional teacher guidance
            select_operation = self.teacher.get_operation_guidance(candidate_operations)
        
        print(f"select_operation: {select_operation}")
        existed_children_operations.append(select_operation)
        operation_ = self.operate_grounding(select_operation, obs[-1])
        
        # Check if operation grounding was successful
        if operation_ is None:
            print(f"Operation grounding failed for {select_operation}, skipping execution")
            node.is_fixed = True
            return None, False

        operated_object_id = select_operation.get('object_id')
        self.hand.do_operation(operation_, self.eye.left, self.eye.top)
        print("wait for operations to finish......")
        time.sleep(self.exec_duration)
        obs.append(self.get_observation())
        operations.append(select_operation)
        print(f"operations: {operations}")
        new_mcts_node = state['mcts'].expand(node, 3, operations)
        
        # Unified screen change detection and object interactivity update
        return self._analyze_screen_changes_and_generate_skill(
            step, obs, operations, select_operation, operated_object_id, 
            state, new_mcts_node
        )
    
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

    
    def explore(self, step, state, skill, skill_clusters):
        print(f"begin explore with COMMON mode")
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
        print(f"begin explore skill augment from parent node")
        if (parent_node is not None) and (len(parent_node.operations) < self.max_operation_length):
            while (not parent_node.is_fixed) and (not stop_flag) and (new_skill_num < 3):
                new_skill, stop_flag = self.skill_augment(step, state, parent_node)
                if new_skill is not None:
                    new_skill_num += 1
                    new_skills.append(new_skill)

        print(f"begin explore skill augment from selected node")
        new_skill_num = 0
        if (len(mcts_node.operations) < self.max_operation_length):
            while (not mcts_node.is_fixed) and (not stop_flag) and (new_skill_num < 3):
                new_skill, stop_flag = self.skill_augment(step, state, mcts_node)
                if new_skill is not None:
                    new_skill_num += 1
                    new_skills.append(new_skill)

        if len(new_skills) == 0:
            print("No new skills generated")
            return 'ExploreFail'
        else:
            print(f"New skills generated: {len(new_skills)}")
            self.brain.merge_and_save_skills(step, state, skill_clusters, new_skills)
            print(f"skill_clusters num: {len(skill_clusters)}")
            self.brain.long_memory.update_state(state)
            self.logger.log({f"skills/skills_generate_num": len(new_skills)}, step)
            return 'Continue'
    
    def exploit(self, step, task, skill):
        print(f"begin exploit with COMMON mode")
        self.state_reset(step)
        obs = [self.get_observation()]
        print(f"selected skill id: {skill['id']} name: {skill['name']} description: {skill['description']} \
                   fitness: {skill['fitness']} num: {skill['num']} operations: {skill['operations']}")
        skill_fitness = skill['fitness']
        exec_chain = []
        operations = skill['operations']
        for operation in operations:
            ob = self.get_observation()
            operation_ = self.operate_grounding(operation, ob)
            exec_chain.append({'screen': f'data:image/png;base64,{cv_to_base64(ob["screen"])}', 'operation': operation_})
            push_data({'exec_chain': exec_chain})
            if operation_ is None:
                print("Operation grounding failed")
                push_data({'result': 'grounding failed'})
                return 'Fail'

            self.hand.do_operation(operation_, self.eye.left, self.eye.top)
            print("wait for operations to finish......")
            time.sleep(self.exec_duration)
        obs.append(self.get_observation())
        exec_chain.append({'screen': f'data:image/png;base64,{cv_to_base64(obs[-1]["screen"])}'}) 
        push_data({'exec_chain': exec_chain})
        
        if not self.eye.detect_acted_cv(obs[-2]['screen'], obs[-1]['screen']):
            print("Action not acted")
            self.logger.log({"eval/skill_acted": 0}, step)
            push_data({'result': 'not acted'})
            return 'Fail'

        # Generate skill from MCP-selected operation if screen changed significantly
        if len(operations) > 0:
            # Detect screen changes and generate skill if significant
            screen_change_result = self._detect_screen_changes_and_update_interactivity(
                obs, operations[-1], operated_object_id=operations[-1].get('object_id')
            )
            screen_change_ratio = screen_change_result['screen_change_ratio']
            
            if self._is_significant_screen_change(screen_change_ratio):
                print(f"Significant screen change detected (ratio: {screen_change_ratio:.3f}), generating skill")
                # Create a temporary state and node for skill generation
                temp_state = {'id': f'mcp_temp_{step}'}
                temp_node_id = f'mcp_node_{step}'
                
                new_skill = self.brain.generate_and_save_skill(
                    step, obs, operations, temp_state['id'], temp_node_id, self
                )
                
                if new_skill:
                    print(f"Successfully generated skill from MCP operation: {new_skill.get('name', 'Unknown')}")
                    
                    # Check if the skill is incomplete and needs continuation
                    if new_skill.get('incomplete', False):
                        print(f"Incomplete skill detected: {new_skill['name']}")
                        print(f"Next action hint: {new_skill['next_action_hint']}")
                        
                        # Continue with MCP for follow-up actions
                        print("Triggering MCP continuation for incomplete skill...")
                        try:
                            # Get current observation and detected objects for MCP continuation
                            current_obs = self.get_observation()
                            detected_objects = self.detector.detect(current_obs['screen'])
                            
                            # Call MCP with context about the incomplete operation
                            continuation_result = self.brain.do_operation_mcp(
                                step + 0.5,  # Use fractional step to indicate continuation
                                task + f" (Continue: {new_skill['next_action_hint']})",
                                {'screen': current_obs['screen']},
                                detected_objects,
                                pre_knowledge=f"Previous incomplete action: {new_skill['description']}. Next: {new_skill['next_action_hint']}",
                                max_iterations=2  # Limit iterations for continuation
                            )
                            
                            if continuation_result and continuation_result.get('success'):
                                print(f"MCP continuation successful: {continuation_result.get('operation', 'Unknown')}")
                                # Update the incomplete skill to complete if successful
                                self.brain.long_memory.update_skill(new_skill['id'], 1, 1)  # Mark as successful
                            else:
                                print("MCP continuation failed or no follow-up action taken")
                                
                        except Exception as e:
                            print(f"Error during MCP continuation: {e}")
                else:
                    print("Failed to generate skill from MCP operation")
            else:
                print(f"Screen change not significant (ratio: {screen_change_ratio:.3f}), skipping skill generation")

        # skill_evaluate
        if not self.close_evaluate:
            time0 = time.time()
            is_consistent, is_progressive = self.brain.skill_evaluate(step, task, obs, skill)
            elapsed_time = time.time() - time0
            print(f"skill_evaluate elapsed_time: {elapsed_time}")
            print(f"is_consistent: {is_consistent} is_progressive: {is_progressive}")
            push_data({'result': f"is_consistent: {is_consistent} is_progressive: {is_progressive}"})
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
                
                # skill_fitness = (skill_fitness + skill['fitness'] * skill['num']) / num
                skill['fitness'] = skill_fitness
                skill['num'] = num

                self.brain.long_memory.update_skill(skill['id'], skill_fitness, num)

                if is_consistent and is_progressive:
                    result = 'Continue'
                else:
                    result = 'Fail'
        else:
            result = 'Continue'

        self.logger.log({"eval/skill_acted": 1}, step)
        return result
    
    def exploit_mcp(self, step, task, skill):
        """Enhanced exploit method with MCP support for operation selection"""
        print(f"begin exploit with MCP mode:")
        self.state_reset(step)
        obs = [self.get_observation()]
        print(f"selected skill id: {skill['id']} name: {skill['name']} description: {skill['description']} \
                   fitness: {skill['fitness']} num: {skill['num']} operations: {skill['operations']}")
        skill_fitness = skill['fitness']
        exec_chain = []
        operations = skill['operations']
        
        # If MCP mode is enabled and skill has no operations, use MCP for operation selection
        if not operations or len(operations) == 0:
            print("Using MCP for operation selection")
            # Get historical objects from recent states for comprehensive context
            historical_objects = []
            try:
                # Get objects from the current state's object_ids if available
                current_state = obs[0].get('state', {})
                if 'object_ids' in current_state and current_state['object_ids']:
                    # Get more historical objects for richer MCP context (increased from 20 to 50)
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
            except Exception as e:
                print(f"Could not retrieve historical objects: {e}")
            
            # Use comprehensive object detection that includes both new and historical objects
            detected_objects = self.detector.get_detected_objects_with_context(obs[0]['screen'], historical_objects)
            print(f"Detected {len(detected_objects)} objects for MCP mode (including historical context)")
            
            # Get hint context for MCP decision making
            hint_context = self.get_hint_context()
            base_pre_knowledge = get_pre_knowledge(self.game_name)
            if hint_context:
                base_pre_knowledge += f"\n\nHint Context:\n{hint_context}"
            
            # Enhance pre_knowledge with historical insights
            pre_knowledge = self.brain.enhance_pre_knowledge_with_history(base_pre_knowledge, task)
            
            # Use streaming MCP-style interaction for operation selection (optimized token usage)
            mcp_result = self.brain.do_operation_mcp_streaming(
                step, task, obs[0], detected_objects, 
                pre_knowledge
            )
            
            if mcp_result is None:
                print("MCP operation selection failed")
                return 'Fail'
            
            if mcp_result['action_type'] == 'select_skill':
                # Handle skill selection - recursive call with selected skill
                skill_id = mcp_result['skill_id']
                print(f"MCP selected skill ID: {skill_id}")
                # This would need proper skill retrieval logic
                return 'Continue'  # Placeholder
            elif mcp_result['action_type'] == 'direct_operation':
                # Handle direct operation
                operation = {
                    'operate': mcp_result['operate'],
                    'params': mcp_result['params']
                }
                
                # Include object_id from MCP result if available
                if 'object_id' in mcp_result:
                    operation['object_id'] = mcp_result['object_id']
                    print(f"MCP exploit operation with object_id: {operation['object_id']}")
                else:
                    operation['object_id'] = 'None'
                    print("MCP exploit operation without object_id, setting to None")
                
                operations = [operation]  # Use MCP-selected operation
            else:
                print(f"Unknown MCP action type: {mcp_result['action_type']}")
                return 'Fail'
        
        # Execute operations (either from skill or MCP-selected)
        for operation in operations:
            ob = self.get_observation()
            operation_ = self.operate_grounding(operation, ob)
            exec_chain.append({'screen': f'data:image/png;base64,{cv_to_base64(ob["screen"])}', 'operation': operation_})
            push_data({'exec_chain': exec_chain})
            if operation_ is None:
                print("Operation grounding failed")
                push_data({'result': 'grounding failed'})
                return 'Fail'

            self.hand.do_operation(operation_, self.eye.left, self.eye.top)
            print("wait for operations to finish......")
            time.sleep(self.exec_duration)
        obs.append(self.get_observation())
        exec_chain.append({'screen': f'data:image/png;base64,{cv_to_base64(obs[-1]["screen"])}'})
        push_data({'exec_chain': exec_chain})
        
        if not self.eye.detect_acted_cv(obs[-2]['screen'], obs[-1]['screen']):
            print("Action not acted")
            self.logger.log({"eval/skill_acted": 0}, step)
            push_data({'result': 'not acted'})
            return 'Fail'

        # skill_evaluate
        if not self.close_evaluate:
            time0 = time.time()
            is_consistent, is_progressive = self.brain.skill_evaluate(step, task, obs, skill)
            elapsed_time = time.time() - time0
            print(f"skill_evaluate elapsed_time: {elapsed_time}")
            print(f"is_consistent: {is_consistent} is_progressive: {is_progressive}")
            push_data({'result': f"is_consistent: {is_consistent} is_progressive: {is_progressive}"})
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
                
                # skill_fitness = (skill_fitness + skill['fitness'] * skill['num']) / num
                skill['fitness'] = skill_fitness
                skill['num'] = num

                self.brain.long_memory.update_skill(skill['id'], skill_fitness, num)

                if is_consistent and is_progressive:
                    result = 'Continue'
                else:
                    result = 'Fail'
        else:
            result = 'Continue'

        self.logger.log({"eval/skill_acted": 1}, step)
        return result
        
    @deprecated
    def run_step_base(self, step, task):
        self.state_reset(step)
        self.logger.log({"decision": 1, "decision_text": "Exploit"}, step)
        push_data({'step': step, 'decision': 'Exploit'})
        states = [self.get_state()]
        operation = self.brain.do_operation(step, task, states[0], get_pre_knowledge(self.game_name))
        if operation is None:
            return 'Continue'
        self.hand.do_operation(operation, self.eye.left, self.eye.top)
        print("wait for operations to finish......")

        time.sleep(self.exec_duration)
        states.append(self.get_state())

        if not self.eye.detect_acted_cv(states[-2]['screen'], states[-1]['screen']):
            print("Action not acted")
            self.logger.log({"eval/skill_acted": 0}, step)
            return 'Continue'

        # skill_evaluate
        if not self.close_evaluate:
            time0 = time.time()
            is_progressive = self.brain.skill_evaluate2(step, task, states)
            elapsed_time = time.time() - time0
            print(f"skill_evaluate elapsed_time: {elapsed_time}")
            print(f"is_progressive: {is_progressive}")
            if is_progressive is not None:
                self.logger.log({"eval/skill_progressive": int(is_progressive)}, step)
                accumulated_progressive = self.logger.last_value('eval/accumulated_skill_progressive') if self.logger.last_value('eval/accumulated_skill_progressive') is not None else 0

                if is_progressive:
                    accumulated_progressive += 1

                self.logger.log({"eval/accumulated_skill_progressive": accumulated_progressive}, step)

        self.logger.log({"eval/skill_acted": 1}, step)

        im1 = f'data:image/png;base64,{cv_to_base64(states[0]["screen"])}'
        im2 = f'data:image/png;base64,{cv_to_base64(states[1]["screen"])}'

        push_data({'exec_chain': [{'screen': im1, 'operation': operation}, {'screen': im2}]})
        return 'Continue'
    
    def _execute_skill_by_id(self, skill_id, state):
        """Execute a skill by its ID - placeholder for skill execution logic"""
        # This is a placeholder - you would need to implement skill retrieval and execution
        # based on your existing skill management system
        print(f"Executing skill ID: {skill_id}")
        
        # For now, return a basic click operation as fallback
        # In a real implementation, you would:
        # 1. Retrieve the skill from your skill database/memory
        # 2. Execute the skill's operations
        # 3. Return the appropriate operation
        
        return {
            'operate': 'Click',
            'coordinate': [400, 300]  # Default center click
        }
    
    def run(self, task, max_step=50):
        is_paused = True
        is_continuous = False
        step_requested = False
        should_exit = False

        self.get_observation()

        def toggle_pause():
            nonlocal is_paused
            is_paused = not is_paused   

        def toggle_continuous():
            nonlocal is_continuous
            is_continuous = True

        def request_step():
            nonlocal step_requested
            step_requested = True   

        def request_exit():
            nonlocal should_exit
            should_exit = True

        def on_press(key):
            try:
                if key.char == ' ':
                    toggle_pause()
                elif key.char == ']':
                    toggle_continuous()
                elif key.char == '[':
                    request_step()
                elif key.char == '/':
                    request_exit()
            except AttributeError:
                # prevent shift、ctrl error
                pass

        listener = pkb.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()  # non-blocking listen

        print("Running controls:")
        print("Space: Toggle pause")
        print("]: Toggle continuous execution")
        print("[: Step execution")
        print("/: Exit running")

        finished = False
        step = self.logger.last_value('step') + 1  if self.logger.last_value('step') is not None else 0
        while not finished and not should_exit:
            if is_paused and not step_requested:
                time.sleep(0.1)
                continue

            print(f"Running step: {step}")
            self.logger.log({"step": step}, step)

            if step > max_step:
                break

            if self.is_base:
                result = self.run_step_base(step, task)
            else:
                result = self.run_step(step, task)

            if result == 'Finished':
                finished = True

            if step_requested:
                step_requested = False
                is_paused = True  # Always pause after step execution

            if not is_continuous:
                is_paused = True  # Always pause if not in continuous mode

            time.sleep(0.1)  # Small delay between steps
            step += 1
    
    def state_reset(self, step=None, force_reset=False):
        """Universal state reset that adapts to any application scenario
        
        Args:
            step: Current step number (optional)
            force_reset: Force reset regardless of conditions
        """
        if self.close_reset:
            return
            
        # Force reset if explicitly requested
        if force_reset:
            self._perform_reset()
            if step is not None:
                self.last_reset_step = step
            return
            
        # Skip reset if no step provided (backward compatibility)
        if step is None:
            return
            
        # Skip reset if we just reset in the same step
        if step == self.last_reset_step:
            print(f"Skipping reset - already reset in step {step}")
            return
            
        # Universal reset condition check
        if self._should_reset_universal(step):
            print(f"Performing intelligent reset for step {step}")
            self._perform_reset()
            self.last_reset_step = step
        else:
            print(f"Skipping reset for step {step} - not needed")
                
    def _should_reset_universal(self, step):
        """Universal reset condition check that adapts to any application scenario"""
        try:
            # Get current screen to analyze state
            current_screen = self.eye.get_screenshot_cv()
            
            # Use detector to get text content and UI elements
            detected_objects = self.detector.get_detected_objects(current_screen)
            
            # Extract text content from detected objects
            text_content = []
            for obj in detected_objects:
                content = obj.get('content', '').lower().strip()
                if content:
                    text_content.append(content)
            
            all_text = ' '.join(text_content).lower()
            
            # Universal reset conditions based on common UI patterns:
            
            # 1. State transition indicators (common across applications)
            state_transition_keywords = [
                "restart", "reset", "start over", "begin", "new", "continue", 
                "next", "proceed", "finish", "complete", "done", "end",
                "menu", "home", "back", "return", "exit", "close"
            ]
            
            # 2. Error or completion indicators
            completion_keywords = [
                "error", "failed", "success", "completed", "finished", 
                "victory", "defeat", "game over", "task complete", "mission accomplished"
            ]
            
            # 3. Navigation or mode change indicators
            navigation_keywords = [
                "loading", "please wait", "processing", "connecting", 
                "login", "logout", "sign in", "sign out", "switch", "change"
            ]
            
            # Check for state transitions that typically require reset
            if any(keyword in all_text for keyword in state_transition_keywords):
                if step - self.last_reset_step > 3:  # Avoid too frequent resets
                    print(f"Reset condition: State transition detected - {step - self.last_reset_step} steps since last reset")
                    return True
            
            # Check for completion or error states
            if any(keyword in all_text for keyword in completion_keywords):
                print("Reset condition: Completion/error state detected")
                return True
            
            # Check for navigation changes
            if any(keyword in all_text for keyword in navigation_keywords):
                if step - self.last_reset_step > 5:  # Less frequent for navigation
                    print(f"Reset condition: Navigation change detected - {step - self.last_reset_step} steps since last reset")
                    return True
            
            # Safety mechanism: reset after extended period without reset
            max_steps_without_reset = getattr(self, 'max_steps_without_reset', 25)
            if step - self.last_reset_step > max_steps_without_reset:
                print(f"Reset condition: Safety reset after {step - self.last_reset_step} steps")
                return True
            
            # Screen change based reset (if significant visual changes detected)
            if hasattr(self, 'last_screen_state'):
                # This could be enhanced with actual screen comparison logic
                # For now, we rely on text-based detection
                pass
                
            return False
            
        except Exception as e:
            print(f"Error in universal reset condition check: {e}")
            # Fallback to reset on error to maintain system stability
            return True
            
    def _perform_reset(self):
        """Universal reset operation that adapts to any application scenario"""
        try:
            # Universal reset approach: right-click at a safe position
            # This works for most applications as it often brings up context menus
            # or cancels current operations
            x = self.eye.left + 50
            y = self.eye.top + 50   
            self.hand.right_single_click(x, y)
            
            # Allow time for the reset operation to complete
            reset_wait_time = getattr(self, 'reset_wait_time', 0.2)
            print(f"Performing universal reset operation, waiting {reset_wait_time}s...")
            time.sleep(reset_wait_time)
            
            # Clear hint context to prevent stale hints from affecting next steps
            self.clear_current_hint()
            print("Cleared current hint during reset")
            
            # Clear any application-specific state flags
            if hasattr(self, 'game_over_detected'):
                self.game_over_detected = False
            if hasattr(self, 'current_turn_detected'):
                self.current_turn_detected = False
                
        except Exception as e:
            print(f"Error performing reset operation: {e}")
            # Even if reset fails, we should continue to avoid getting stuck
