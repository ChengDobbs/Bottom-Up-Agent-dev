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

        # Initialize Eye first to check window availability early
        self.eye = Eye(config)
        
        # Initialize other components after confirming window exists
        self.logger = Logger(config['project_name'], config['game_name'] + ' - ' + config['run_name'], backend='wandb')
        self.hand = Hand(config)
        self.detector = Detector(config)
        self.teacher = Teacher(config)
        self.brain = Brain(config, self.detector, self.logger)
        self.close_explore = config['close_explore']
        self.close_evaluate = config['close_evaluate'] if 'close_evaluate' in config else False
        self.close_reset = config['close_reset'] if 'close_reset' in config else True
        self.encode_image = self.detector.encode_image

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
    
    def get_screen(self):
        """Get current screen without detector encoding - lightweight version"""
        return self.eye.get_screenshot_cv()

    def get_observation(self, include_hint=None):
        screen = self.get_screen()
        state_feature = self.encode_image(screen)
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
        """Get hint context for single/multi-turn decision making"""
        context = []
        if self.context_manager['current_hint']:
            current_hint = self.context_manager['current_hint']
            print(f"[HINT] Current hint: {current_hint}")
            # Handle structured hint format
            if isinstance(current_hint, dict):
                context.append("🎯 CURRENT ACTION HINT (High Priority):")
                context.append(f"  Action Type: {current_hint.get('action_type', 'N/A')}")
                
                if 'target_coordinates' in current_hint:
                    coords = current_hint['target_coordinates']
                    context.append(f"  Target Coordinates: [{coords[0]}, {coords[1]}]")
                
                if 'target_object_id' in current_hint:
                    context.append(f"  Target Object ID: {current_hint['target_object_id']}")
                
                if 'keyboard_key' in current_hint:
                    context.append(f"  Keyboard Key: {current_hint['keyboard_key']}")
                
                context.append(f"  Reasoning: {current_hint.get('reasoning', 'N/A')}")
                context.append(f"  Expected Outcome: {current_hint.get('expected_outcome', 'N/A')}")
                
                # Add actionable instruction
                action_type = current_hint.get('action_type', '')
                if action_type == 'Click' and 'target_coordinates' in current_hint:
                    coords = current_hint['target_coordinates']
                    context.append(f"  ⚡ EXECUTE: Click({coords[0]}, {coords[1]})")
                elif action_type == 'Key' and 'keyboard_key' in current_hint:
                    key = current_hint['keyboard_key']
                    context.append(f"  ⚡ EXECUTE: Key('{key}')")
                
            else:
                # Handle legacy string format
                context.append(f"Current hint: {current_hint}")
        
        if self.context_manager['hint_history']:
            recent_hints = self.context_manager['hint_history'][-2:]  # Last 2 hints (excluding current)
            if recent_hints:
                context.append("\n📋 Recent Hint History:")
                for i, hint_entry in enumerate(recent_hints):
                    hint = hint_entry['hint']
                    if isinstance(hint, dict):
                        context.append(f"  {i+1}. {hint.get('action_type', 'N/A')}: {hint.get('reasoning', 'N/A')}")
                    else:
                        context.append(f"  {i+1}. {hint}")
        
        return "\n".join(context) if context else None
    
    def _execute_hint_directly(self, step, task, hint):
        """
        Directly execute a hint without going through the normal exploration/exploitation flow
        """
        print(f"🎯 Executing hint directly: {hint}")
        
        # Get current screen directly without full observation to avoid detector calls
        screen = self.get_screen()
      
        # Convert hint to operation format
        operation = self._convert_hint_to_operation(hint)
        if operation is None:
            print("❌ Failed to convert hint to operation")
            self.clear_current_hint()
            return 'Fail'
        
        print(f"🎯 Converted hint to operation: {operation}")
        
        # Create minimal observation for grounding
        ob = {
            'screen': screen,
            'detected_objects': []
        }
        
        # Ground the operation
        grounded_operation = self.operate_grounding(operation, ob)
        
        if grounded_operation is None:
            print("❌ Failed to ground hint operation")
            self.clear_current_hint()
            return 'Fail'
        
        print(f"🎯 Grounded operation: {grounded_operation}")
        
        # Execute the operation
        try:
            self.hand.do_operation(grounded_operation, self.eye.left, self.eye.top)
            print("🎯 Hint operation executed successfully")
            
            # Wait for operation to complete
            time.sleep(self.exec_duration)
            
            # Clear the hint after successful execution
            self.clear_current_hint()
            
            return 'Continue'
            
        except Exception as e:
            print(f"❌ Error executing hint operation: {e}")
            self.clear_current_hint()
            return 'Fail'
    
    def _convert_hint_to_operation(self, hint):
        """
        Convert hint information to operation format
        """
        if not isinstance(hint, dict):
            print(f"❌ Invalid hint format: {hint}")
            return None
        
        action_type = hint.get('action_type', '').lower()
        
        if action_type == 'click':
            target_coords = hint.get('target_coordinates')
            if target_coords and len(target_coords) >= 2:
                return {
                    'operate': 'Click',
                    'params': {'x': target_coords[0], 'y': target_coords[1]},
                    'object_id': hint.get('target_object_id', 'None')
                }
        elif action_type == 'key':
            keyboard_key = hint.get('keyboard_key')
            if keyboard_key:
                return {
                    'operate': 'Key',
                    'params': {'key': keyboard_key}
                }
        elif action_type in ['interact', 'move_up', 'move_down', 'move_left', 'move_right', 'sleep', 'attack']:
            return {
                'operate': action_type,
                'params': {}
            }
        
        print(f"❌ Unsupported hint action type: {action_type}")
        return None
    
    def clear_current_hint(self):
        """Clear current hint after successful action"""
        self.context_manager['current_hint'] = None
    
    def get_click_trigger_operations(self, existed_objects):
        operations = []
        for object in existed_objects:
            if object.get('id') is None:
                print(f"Skipping object without ID: {object.get('content', 'Unknown')[:30]}...")
                continue
            operations.append({
                'operate': 'Click',
                'object_id': object['id'],
                'params': {'x': object['center'][0], 'y': object['center'][1]}
            })
        return operations

    def get_potential_operations(self, existed_objects):
        operations = []
        
        # For other games, use configured operations
        for operate in self.operates:
            if operate == 'Click':
                operations.extend(self.get_click_trigger_operations(existed_objects))
            else:
                print(f"Unsupported operate: {operate}")
                continue
        return operations
    
    def explore_hover_effects(self, existed_objects, state, hover_threshold):
        """
        Test hover effects on all detected objects.
        Always re-test to handle object replacements/updates.
        Does NOT create skill nodes.
        """
        import random
        
        print(f"\n[HOVER] Starting hover exploration")
        
        # Filter valid objects (all objects with ID, no filtering by is_hover_change)
        valid_objects = [obj for obj in existed_objects if obj.get('id') is not None]
        
        if not valid_objects:
            print("[HOVER] No valid objects to explore")
            return {'explored': 0, 'hoverable': 0}
        
        # Shuffle all objects for random order testing
        sample_size = len(valid_objects)
        shuffled_objects = random.sample(valid_objects, sample_size)
        
        print(f"[HOVER] Will test {sample_size} objects")
        
        hoverable_count = 0
        for i, obj in enumerate(shuffled_objects, 1):
            object_id = obj['id']
            
            try:
                has_change = self.regional_hover_detect(obj, state, hover_threshold)
                if has_change:
                    hoverable_count += 1
                    print(f"[HOVER] {i}/{sample_size} Object {object_id}: HAS effect ✓")
                else:
                    print(f"[HOVER] {i}/{sample_size} Object {object_id}: NO effect ✗")
            except Exception as e:
                print(f"[HOVER] {i}/{sample_size} Object {object_id}: Error - {e}")
        
        print(f"[HOVER] Complete: {hoverable_count}/{sample_size} hoverable\n")
        return {'explored': sample_size, 'hoverable': hoverable_count}
    
    def regional_hover_detect(self, obj, state, hover_threshold):
        import cv2
        import numpy as np
        
        x, y = obj['center'][0], obj['center'][1]
        
        # Get screen before hover
        screen_before = self.get_observation()['screen'].copy()
        
        # Execute hover
        hover_op = {'operate': 'Hover', 'params': {'x': x, 'y': y}}
        operation_ = self.operate_grounding(hover_op, state)
        if operation_ is None:
            return False
            
        self.hand.do_operation(operation_, self.eye.left, self.eye.top)
        time.sleep(0.5)  # Wait for tooltip
        
        # Get screen after hover
        screen_after = self.get_observation()['screen']
        
        # Calculate 1/3 screen region around hover position
        screen_h, screen_w = screen_before.shape[:2]
        region_w = screen_w // 3
        region_h = screen_h // 3
        x1 = max(0, x - region_w // 2)
        y1 = max(0, y - region_h // 2)
        x2 = min(screen_w, x1 + region_w)
        y2 = min(screen_h, y1 + region_h)
        
        # Check visual change in region
        region_before = screen_before[y1:y2, x1:x2]
        region_after = screen_after[y1:y2, x1:x2]
        
        diff = cv2.absdiff(region_before, region_after)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        change_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        change_percentage = (change_pixels / total_pixels) * 100
        
        # Only proceed if visual change meets threshold
        if change_percentage <= hover_threshold:
            return False
        
        # Detect new text objects in the region
        objects_before = self.detector.extract_objects_omni(screen_before)
        objects_after = self.detector.extract_objects_omni(screen_after)
        
        # Get content from objects before hover in the region
        before_contents = set()
        for obj_before in objects_before:
            obj_x, obj_y = obj_before.get('center', [0, 0])
            if (x1 <= obj_x <= x2 and y1 <= obj_y <= y2 and 
                (obj_before.get('type') == 'text' or (obj_before.get('type') == 'icon' and obj_before.get('area') > 3500))):
                before_contents.add(obj_before.get('content'))
        
        # Find new text objects after hover in the region
        new_text_objects = []
        print(f"region after hover: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        for obj_after in objects_after:
            obj_x, obj_y = obj_after.get('center', [0, 0])
            if (x1 <= obj_x <= x2 and y1 <= obj_y <= y2 and 
                (obj_after.get('type') == 'text' or (obj_after.get('type') == 'icon' and obj_after.get('area') > 3500))):
                content = obj_after.get('content')
                if content and len(content) > 3 and content not in before_contents:
                    new_text_objects.append(obj_after)
        # Only consider it a hover change if new text objects are found
        has_hover_change = len(new_text_objects) > 0
        tooltip_text = None
        
        if has_hover_change:
            tooltip_text = ' \\ '.join([obj.get('content', '') for obj in new_text_objects])
        
        # Update database with hover info
        self.brain.long_memory.update_object_hover_info(
            object_id=obj['id'],
            is_hover_change=has_hover_change,
            hover_tooltip=tooltip_text
        )
        
        return has_hover_change
    
    def _is_center_in_bbox(self, center, bbox):
        """Check if center point is inside bbox. Bbox format: [x, y, w, h] (xywh)."""
        if not bbox or len(bbox) != 4:
            return False
        x, y, w, h = bbox
        cx, cy = center
        return x <= cx <= x + w and y <= cy <= y + h
    
    def _has_significant_bbox_overlap(self, bbox, bbox_list, overlap_threshold=0.3):
        """Check if bbox has significant overlap with any bbox in the list. Bbox format: [x, y, w, h] (xywh)."""
        if not bbox or len(bbox) != 4:
            return False
        
        for other_bbox in bbox_list:
            if not other_bbox or len(other_bbox) != 4:
                continue
                
            # Convert xywh to xyxy for intersection calculation
            x1, y1, w1, h1 = bbox
            x2, y2, w2, h2 = other_bbox
            
            # Calculate intersection area
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                bbox_area = w1 * h1
                other_area = w2 * h2
                
                # Use smaller area as denominator for overlap ratio
                min_area = min(bbox_area, other_area)
                if min_area > 0:
                    overlap_ratio = intersection_area / min_area
                    if overlap_ratio > overlap_threshold:
                        return True
        
        return False
    
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
            'LeftDouble': UnifiedOperation.LeftDouble,
            'Hover': UnifiedOperation.Hover
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
        
        # Priority check: If we have a pending hint, execute it directly
        current_hint = self.get_current_hint()
        if current_hint:
            print(f"[HINT] Found pending hint, executing directly: {current_hint.get('action_type', 'Unknown')}")
            return self._execute_hint_directly(step, task, current_hint)
        
        # get screen only when no hint is available
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
        
        if 'Hover' in self.operates:
            hover_stats = self.explore_hover_effects(updated_objects, obs[-1], hover_threshold=5)
            print(f"[INFO] Hover exploration completed: {hover_stats['hoverable']}/{hover_stats['explored']} objects have hover effects")
        
        # Generate potential operations (Click operations that will create skill nodes)
        potential_operations = self.get_potential_operations(updated_objects)
        
        return self._process_skill_augmentation(step, state, node, obs, operations, potential_operations)
    
    def _select_operation_with_mcp(self, candidate_operations, step, obs, operations):
        """
        Use MCP mode to intelligently select operations
        
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
    
    def _update_object_interactivity_from_feedback(self, object_id, screen_change_ratio, is_state_changed):
        try:
            is_click_change = self._is_significant_screen_change(screen_change_ratio)
            
            self.brain.long_memory.update_object_clickability_by_id(object_id, is_click_change)
            
            print(f"[OBJ LIBRARY] Object {object_id} is_click_change: {is_click_change}")
            
        except Exception as e:
            print(f"[ERROR] Failed to update object {object_id} interactivity: {e}")
    
    def _detect_screen_changes(self, states, operation, operated_object_id=None):
        """
        Screen change detection.
        
        Args:
            states: List of observation states (before and after operation)
            operation: The operation that was executed
            operated_object_id: ID of the operated object (optional)
            
        Returns:
            dict: Contains screen_change_ratio, is_state_changed info
        """
        # Detect screen changes
        screen_change_ratio = self.eye.detect_acted_cv(states[-2]['screen'], states[-1]['screen'])
        is_state_changed, sim_2_states = self.brain.detect_state_changed(states[-2], states[-1])
        print(f"screen_change_ratio: {screen_change_ratio} Sim between 2 states: {sim_2_states}")
        
        return {
            'screen_change_ratio': screen_change_ratio,
            'is_state_changed': is_state_changed,
            'sim_2_states': sim_2_states
        }
    
    def _analyze_screen_changes_and_generate_skill(self, step, obs, operations, select_operation, 
                                                 operated_object_id, state, new_mcts_node):
        """
        Unified method for screen change analysis and skill generation.
        This tool can be reused across different execution modes.
        """
        # Use unified screen change detection
        screen_change_result = self._detect_screen_changes(
            obs, select_operation, operated_object_id=operated_object_id
        )
        
        screen_change_ratio = screen_change_result['screen_change_ratio']
        is_state_changed = screen_change_result['is_state_changed']
        
        # Generate new skill based on screen changes
        if self._is_significant_screen_change(screen_change_ratio):
            new_skill = self.brain.generate_and_save_skill(
                step, obs, operations, state['id'], new_mcts_node.node_id, self
            )
            # TODO: add object interactivity update
            self._update_object_interactivity_from_feedback(
                operated_object_id, screen_change_ratio, is_state_changed
            )
            # if self.brain.detect_state_changed(obs[0], obs[-1])[0]:
            # Use the already calculated is_state_changed from screen_change_result
            # to avoid redundant State similarity calculation
            if is_state_changed:
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
                    
                    # Check if the skill is incomplete and needs immediate continuation
                    if new_skill.get('incomplete', False):
                        print(f"[COMPLETENESS] INCOMPLETE SKILL in explore: {new_skill['name']}")                        
                        # Execute hint action immediately to complete the skill
                        hint_info = new_skill['next_action_hint']
                        try:
                            # Build hint operation from hint_info
                            hint_operation = {
                                'operate': hint_info.get('action_type', 'Click'),
                                'params': {
                                    'x': int(hint_info['target_coordinates'][0]),
                                    'y': int(hint_info['target_coordinates'][1])
                                } if 'target_coordinates' in hint_info else {}
                            }
                                                        
                            # Get current observation
                            current_obs = self.get_observation()
                            
                            # Execute the hint operation
                            operation_ = self.operate_grounding(hint_operation, current_obs)
                            if operation_ is not None:
                                self.hand.do_operation(operation_, self.eye.left, self.eye.top)
                                print("[HINT] Waiting for hint operation to complete...")
                                time.sleep(self.exec_duration)
                                
                                # Get new observation after hint execution
                                new_obs = self.get_observation()
                                
                                # Check if hint execution was successful
                                screen_change = self.eye.detect_acted_cv(current_obs['screen'], new_obs['screen'])
                                print(f"[HINT] Hint operation executed, screen change: {screen_change:.3f}")
                                
                                # Update the incomplete skill to complete if successful
                                if screen_change > 0.01:
                                    self.brain.long_memory.update_skill(new_skill['id'], 1, 1)
                                    print(f"[HINT] Skill completed: {new_skill['name']}")
                            else:
                                print(f"[HINT] Hint operation grounding failed")
                                
                        except Exception as e:
                            print(f"[HINT] Error executing hint action: {e}")

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
            screen_change_result = self._detect_screen_changes(
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
                    print(f"Successfully generated skill in exploitation: {new_skill.get('name', 'Unknown')}")
                    
                    # Check if the skill is incomplete and needs continuation
                    if new_skill.get('incomplete', False):
                        print(f"[COMPLETENESS] INCOMPLETE SKILL in exploit: {new_skill['name']}")
                        print(f"[HINT] Next Action Hint: {new_skill['next_action_hint']}")
                        
                        # Execute hint action immediately to complete the skill (direct execution, not MCP)
                        hint_info = new_skill['next_action_hint']
                        try:
                            # Build hint operation from hint_info
                            hint_operation = {
                                'operate': hint_info.get('action_type', 'Click'),
                                'params': {
                                    'x': int(hint_info['target_coordinates'][0]),
                                    'y': int(hint_info['target_coordinates'][1])
                                } if 'target_coordinates' in hint_info else {}
                            }
                            
                            print(f"[HINT] Executing Hint Operation Directly: {hint_operation}")
                            
                            # Get current observation
                            current_obs = self.get_observation()
                            
                            # Execute the hint operation directly
                            operation_ = self.operate_grounding(hint_operation, current_obs)
                            if operation_ is not None:
                                self.hand.do_operation(operation_, self.eye.left, self.eye.top)
                                time.sleep(self.exec_duration)
                                
                                # Get new observation after hint execution
                                new_obs = self.get_observation()
                                
                                # Check if hint execution was successful
                                screen_change = self.eye.detect_acted_cv(current_obs['screen'], new_obs['screen'])
                                print(f"[HINT] Hint operation executed, screen change: {screen_change:.3f}")
                                
                                # Update the incomplete skill to complete if successful
                                if screen_change > 0.01:
                                    self.brain.long_memory.update_skill(new_skill['id'], 1, 1)
                                    print(f"[HINT] Skill completed: {new_skill['name']}")
                            else:
                                print(f"[HINT] Hint operation grounding failed")
                                
                        except Exception as e:
                            print(f"[HINT] Error executing hint action: {e}")
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
            current_screen = self.get_screen()
            
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
