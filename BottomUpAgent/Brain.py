import sqlite3
import time

from base_model import creat_base_model
from . import Prompt
from . import FunctionCalls
from utils.utils import cv_to_base64
from .LongMemory import LongMemory
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

class Brain:
    def __init__(self, config, detector, logger):
        self.game_name = config['game_name']
        self.model_name = config['brain']['base_model']
        self.config = config  # Store config for dynamic tool loading
        # Set timeout for API calls to prevent hanging
        api_timeout = config.get('api_timeout', 60)
        self.base_model = creat_base_model(config['brain']['base_model'], timeout=api_timeout)
        self.evaluate_model = creat_base_model(config['brain']['evaluate_model'], timeout=api_timeout)
        self.long_memory = LongMemory(config)
        self.logger = logger

        self.generate_skill_tools = FunctionCalls.generate_skill_tools(config['brain']['base_model'])
        self.select_skill_tools = FunctionCalls.select_skill_tools(config['brain']['base_model'])
        self.do_operation_tools = FunctionCalls.do_operation_tools(config['brain']['base_model'])
        self.cluster_skills_tool = FunctionCalls.cluster_skills_tool(config['brain']['base_model'])
        self.merge_skills_tool = FunctionCalls.merge_skills_tool(config['brain']['base_model'])

        self.skill_evaluate_tools = FunctionCalls.skill_evaluate_tools(config['brain']['evaluate_model'])
        self.skill_evaluate2_tools = FunctionCalls.skill_evaluate2_tools(config['brain']['evaluate_model'])
        
        # MCP-style interaction tools with dynamic configuration
        self.environment_query_tools = FunctionCalls.environment_query_tools(config['brain']['base_model'])
        # Initialize with full_mode by default, can be changed dynamically
        self.mcp_interaction_tools = FunctionCalls.mcp_interaction_tools(config['brain']['base_model'], mode='full_mode')
        self.current_mcp_mode = 'full_mode'  # Track current mode

        self.uct_c = config['brain']['uct_c']
        self.uct_threshold = config['brain']['uct_threshold']
        self.max_mcp_iter = config.get('mcp', {}).get('max_iter', 6)
        
        # 🧠 MCP Historical Context Management
        self.mcp_conversation_history = []  # Persistent conversation history across MCP calls
        self.max_history_length = config.get('mcp', {}).get('max_history_length', 20)  # Keep last 20 interactions
        self.context_retention_threshold = config.get('mcp', {}).get('context_retention_threshold', 0.7)  # Similarity threshold for context reuse
    
    def switch_mcp_mode(self, mode):
        """
        Dynamically switch MCP tool mode based on game state or phase
        Available modes: combat_mode, map_mode, reward_mode, shop_mode, exploration_mode, skill_learning_mode, full_mode
        """
        if mode != self.current_mcp_mode:
            self.logger.info(f"Switching MCP mode from '{self.current_mcp_mode}' to '{mode}'")
            self.mcp_interaction_tools = FunctionCalls.mcp_interaction_tools(self.model_name, mode=mode)
            self.current_mcp_mode = mode
            return True
        return False
    
    def get_available_mcp_modes(self):
        """
        Get list of available MCP modes from configuration
        """
        mcp_config = FunctionCalls.load_mcp_tools_config()
        if mcp_config:
            # Return all mode keys that end with '_mode'
            return [key for key in mcp_config.keys() if key.endswith('_mode')]
        return ['full_mode']  # Default fallback

    def select_skill_cluster(self, step, ob, skills):
        # select id name description from skills
        skills_info = ''
        for skill in skills:
            skills_info += f"id: {skill['id']}, name: {skill['name']}, description: {skill['description']}\n"

        prompt = Prompt.select_skill_prompt(skills_info)
        response = self.base_model.call_text_images(prompt, [cv_to_base64(ob['screen'])], self.select_skill_tools)
        print(response)
        self.logger.log({"eval/tokens/select_skill/input": response['usage']['input'], 
                         "eval/tokens/select_skill/output": response['usage']['output'],
                         "eval/tokens/select_skill/total": response['usage']['total']}, step)
        
        if response['function'] is not None:
            return int(response['function']['input']['id'])
        else:
            return None
            
    def temperature(self, N, tau0=1, k=0.1):
        return max(0.1, tau0 * np.exp(-k * N))

                
    def select_skill(self, skills, skill_cluster, suspended_skill_ids, close_exploration=False):
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
                push_data({'candidate_skills': skills_info, 'selected_skill_id': 'Explore'})
                return {'name': 'Explore', 'mcts_node_id': mcts_node_id}, True

            ucts = []
            for skill in candidate_skills:
                # Add safety check to prevent division by zero and NaN values
                skill_num = max(skill['num'], 1)  # Ensure num is at least 1
                if num_total <= 0:
                    num_total = 1  # Ensure num_total is positive
                uct = skill['fitness'] + self.uct_c * np.sqrt(np.log(num_total) / skill_num)
                # Check for NaN or infinite values
                if np.isnan(uct) or np.isinf(uct):
                    uct = skill['fitness']  # Fallback to fitness value only
                ucts.append(uct)

            if not close_exploration:
                # Add safety check for exploration UCT calculation
                explore_nums = max(skill_cluster['explore_nums'], 1)  # Ensure explore_nums is at least 1
                explore_uct = self.uct_threshold + self.uct_c * np.sqrt(np.log(num_total) / explore_nums)
                # Check for NaN or infinite values
                if np.isnan(explore_uct) or np.isinf(explore_uct):
                    explore_uct = self.uct_threshold  # Fallback to threshold only
                ucts.append(explore_uct)
                candidate_skills.append({'id': 'Explore', 'name': 'Explore', 'num': skill_cluster['explore_nums'], 
                                         'fitness': self.uct_threshold, 'mcts_node_id': mcts_node_id})
            
            ucts = np.array(ucts)
            # Additional safety checks for UCT array
            ucts = np.nan_to_num(ucts, nan=0.0, posinf=1e6, neginf=-1e6)
            
            temperature = max(self.temperature(num_total), 1e-2)
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

            try:
                from .visualizer import push_data
                push_data({'candidate_skills': skills_info, 'temperature': round(temperature,2), 'selected_skill_id': selected_skill['id']})
            except ImportError:
                pass  # Visualizer not available

            if probs[-1] > 0.9:
                return selected_skill, True
            else:
                return selected_skill, False
                            

    def skill_evaluate(self, step, task, obs, skill):
        skill_info = ''
        if skill is not None:
            skill_info += f"skill: {skill['name']}, description: {skill['description']}\n"
        prompt = Prompt.skill_evaluate_prompt(task, skill_info)
        imgs_64 = [cv_to_base64(ob['screen']) for ob in obs]
        response = self.evaluate_model.call_text_images(prompt, imgs_64, self.skill_evaluate_tools)
        print(response)
        self.logger.log({"eval/tokens/skill_evaluate/input": response['usage']['input'], 
                         "eval/tokens/skill_evaluate/output": response['usage']['output'],
                         "eval/tokens/skill_evaluate/total": response['usage']['total']}, step)

        if response['function'] is not None:
            return response['function']['input']['is_consistent'], response['function']['input']['is_progressive']
        else:
            return None, None
        
    def skill_evaluate2(self, step, task, obs):
        prompt = Prompt.skill_evaluate2_prompt(task)
        imgs_64 = [cv_to_base64(ob['screen']) for ob in obs]
        response = self.evaluate_model.call_text_images(prompt, imgs_64, self.skill_evaluate2_tools)
        print(response)

        self.logger.log({"eval/tokens/skill_evaluate/input": response['usage']['input'],
                         "eval/tokens/skill_evaluate/output": response['usage']['output'],
                         "eval/tokens/skill_evaluate/total": response['usage']['total']}, step)

        if response['function'] is not None:
            return response['function']['input']['is_progressive']
        else:
            return None

    

    def skill_evolution(self, step, skills, skill_cluster, observation_num = 4, fitness_threshold = 2):
        print("begin skill evolution")
        delete_ids = []
        for skill in skills:
            if skill['num'] > observation_num and skill['fitness'] < fitness_threshold:
                print(f"delete skill: {skill['id']} fitness: {skill['fitness']} num: {skill['num']} \
                      operations: {skill['operations']}")
                self.long_memory.delete_skill(skill, skill_cluster)
                delete_ids.append(skill['id'])
        print("end skill evolution")
        try:
            from .visualizer import push_data
            push_data({'delete_ids': delete_ids})
        except ImportError:
            pass  # Visualizer not available
        self.logger.log({f"skills/skills_delete_num": len(delete_ids)}, step)

    def skill_log(self, logger, step):
        skills = self.long_memory.get_skills()
        logger.log({"skills/skills_num": len(skills)}, step)
        for skill in skills:
            if skill['num'] > 1:
                logger.log({f"skills/skill_num_{skill['id']}": skill['num'], f"skills/skill_fitness_{skill['id']}": skill['fitness']}, step)

    
    def do_operation(self, step, task, state, pre_knowledge = None):
        if self.model_name == "UI_TARS":
            prompt = Prompt.do_operation_prompt_v2(task)
        else:
            prompt = Prompt.do_operation_prompt(task)

        imgs_64 = [cv_to_base64(state['screen'])]
        response = self.base_model.call_text_images(prompt, imgs_64, self.do_operation_tools, pre_knowledge=pre_knowledge)
        
        print(response)

        self.logger.log({"eval/tokens/do_operation/input": response['usage']['input'], 
                         "eval/tokens/do_operation/output": response['usage']['output'],
                         "eval/tokens/do_operation/total": response['usage']['total']}, step)

        if response['function'] is not None:
            return {
                "operate": response['function']['name'],
                "params": response['function']['input']
            }
        else:
            return None
        

    def cluster_skills(self, step, skills):
        prompt = Prompt.cluster_skills_prompt(skills)

        response = self.base_model.call_text(prompt, [self.cluster_skills_tool])

        print(response)
        self.logger.log({"eval/tokens/merge_skills/input": response['usage']['input'], 
                         "eval/tokens/merge_skills/output": response['usage']['output'],
                         "eval/tokens/merge_skills/total": response['usage']['total']}, step)
        clusters = []

        if response['function'] is not None:
            if 'clusters' in response['function']['input']:
                clusters = response['function']['input']['clusters']

        return clusters
    
    def merge_skills_to_clusters(self, step, existing_skill_clusters, new_skills):
        prompt = Prompt.merge_skills_prompt(existing_skill_clusters, new_skills)

        response = self.base_model.call_text(prompt, [self.merge_skills_tool])

        print(response)
        self.logger.log({"eval/tokens/merge_skills/input": response['usage']['input'], 
                         "eval/tokens/merge_skills/output": response['usage']['output'],
                         "eval/tokens/merge_skills/total": response['usage']['total']}, step)
        clusters = []

        if response['function'] is not None:
            if 'clusters' in response['function']['input']:
                clusters = response['function']['input']['clusters']

        return clusters
        

    def detect_state_changed(self, state1, state2, threshold=0.85):
        state_embedding1 = state1['state_feature']
        state_embedding2 = state2['state_feature']

        # Ensure embeddings are 2D arrays for cosine_similarity
        if len(state_embedding1.shape) == 1:
            state_embedding1 = state_embedding1.reshape(1, -1)
        if len(state_embedding2.shape) == 1:
            state_embedding2 = state_embedding2.reshape(1, -1)

        similarity = cosine_similarity(state_embedding1, state_embedding2)[0][0]

        print(f"State similarity: {similarity}, threshold: {threshold}")

        return similarity < threshold, similarity
    
    def merge_and_save_skills(self, step, state, skill_clusters, new_skills):
        if len(new_skills) == 0:
            return
        
        if len(skill_clusters) == 0:
            clusters = self.cluster_skills(step, new_skills)
            for cluster in clusters:
                cluster_id = self.long_memory.save_skill_cluster(state['state_feature'], cluster['name'], cluster['description'], cluster['members'])
                state['skill_clusters'].append(cluster_id)
        else:
            clusters = self.merge_skills_to_clusters(step, skill_clusters, new_skills)
            for cluster in clusters:
                if cluster['id'] == -1:
                    cluster_id = self.long_memory.save_skill_cluster(state['state_feature'], cluster['name'], cluster['description'], cluster['members'])
                    state['skill_clusters'].append(cluster_id)
                else:
                    exist_cluter = self.long_memory.get_skill_clusters_by_id(cluster['id'])
                    for id in cluster['members']:
                        if id not in exist_cluter['members']:
                            print(f"add skill {id} to cluster {exist_cluter['name']}")
                            exist_cluter['members'].append(id)
                    self.long_memory.update_skill_cluster(cluster['id'], state['state_feature'], cluster['name'], cluster['description'], exist_cluter['members'])

    def generate_and_save_skill(self, step, obs, operations, state_id, mcst_node_id, agent=None):
        operations_str =  ''
        for idx, operation in enumerate(operations):
            if operation['operate'] in ['Click', 'RightSingle', 'LeftDouble']:
                # Handle both coordinate and x,y formats
                if 'coordinate' in operation['params']:
                    x, y = operation['params']['coordinate']
                else:
                    x, y = operation['params']['x'], operation['params']['y']
                cords = f"({x}, {y})"
                operations_str += 'operation' + str(idx) + ': ' + operation['operate'] + ' ' + cords + '\n'
            elif operation['operate'] == 'Drag' and 'params' in operation:
                # Handle Drag operations with coordinates
                params = operation['params']
                if all(key in params for key in ['x1', 'y1', 'x2', 'y2']):
                    cords = f"({params['x1']}, {params['y1']}) to ({params['x2']}, {params['y2']})"
                    operations_str += 'operation' + str(idx) + ': ' + operation['operate'] + ' ' + cords + '\n'
                else:
                    operations_str += 'operation' + str(idx) + ': ' + operation['operate'] + '\n'
            else:
                # Handle other operation types without coordinates
                operations_str += 'operation' + str(idx) + ': ' + operation['operate'] + '\n'
        # Create a copy of operations for saving to preserve params
        operations_to_save = [op.copy() for op in operations]
        prompt = Prompt.generate_skill_prompt(operations_str)
        imgs_64 = [cv_to_base64(ob['screen']) for ob in obs]
        response = self.base_model.call_text_images(prompt, imgs_64, self.generate_skill_tools)
        print(response)
        self.logger.log({"eval/tokens/generate_skill/input": response['usage']['input'], 
                         "eval/tokens/generate_skill/output": response['usage']['output'],
                         "eval/tokens/generate_skill/total": response['usage']['total']}, step)
        
        if response['function'] is not None:
            if response['function']['name'] == "save_skill":
                id = self.long_memory.save_skill(response['function']['input']['name'], response['function']['input']['description'], 
                                            operations_to_save, 0, 1, state_id, mcst_node_id, obs[0]['screen'], obs[-1]['screen'])
                print(f"save skill: {response['function']['input']['name']}, operations: {operations_str}, fitness: {0}")
                return {"id": id, "name": response['function']['input']['name'], "description": response['function']['input']['description']}
            elif response['function']['name'] == "incomplete_skill":
                # Handle incomplete skill - save as temporary skill and signal for MCP continuation
                print(f"incomplete skill detected: {response['function']['input']['name']}")
                print(f"description: {response['function']['input']['description']}")
                print(f"next action hint: {response['function']['input']['next_action_hint']}")
                
                # Store hint information in agent's context manager if agent is provided
                if agent is not None:
                    agent.store_hint(response['function']['input']['next_action_hint'])
                
                # Save as temporary incomplete skill with special fitness marker
                id = self.long_memory.save_skill(
                    response['function']['input']['name'], 
                    response['function']['input']['description'], 
                    operations_to_save, -1, 0, state_id, mcst_node_id, 
                    obs[0]['screen'], obs[-1]['screen']
                )
                
                return {
                    "id": id, 
                    "name": response['function']['input']['name'], 
                    "description": response['function']['input']['description'],
                    "incomplete": True,
                    "next_action_hint": response['function']['input']['next_action_hint']
                }
            elif response['function']['name'] == "no_meaning_skill":
                print("no meaning skill detected")
                return None
            else:
                print("Unknown function: "+response['function']['name'])
                return None
        else:
            print("No function call!")
            return None
    
    def execute_environment_query(self, query_type, detected_objects, object_id=None, context_query=None):
        """Execute enhanced environment query with game-specific intelligence"""
        if query_type == "all_objects":
            return self._format_all_objects_enhanced(detected_objects)
        elif query_type == "historical_context":
            return self._query_historical_context(context_query)
        elif query_type == "text_content":
            return self._format_text_content_enhanced(detected_objects)
        elif query_type == "specific_object" and object_id:
            return self._format_specific_object(detected_objects, object_id)
        elif query_type == "object_summary":
            return self._format_object_summary_enhanced(detected_objects)
        elif query_type == "game_state":
            return self._analyze_game_state(detected_objects)
        elif query_type == "scene_summary":
            return self._generate_scene_summary(detected_objects)
        else:
            return "Invalid query type or missing object_id for specific_object query"
    
    def _format_all_objects(self, detected_objects):
        """Format all detected objects information"""
        if not detected_objects:
            return "No objects detected in current screen."
        
        result = f"Detected {len(detected_objects)} objects:\n"
        for i, obj in enumerate(detected_objects):
            result += f"{i+1}. ID: {obj.get('id', 'N/A')}, Type: {obj.get('type', 'unknown')}, "
            result += f"Center: {obj.get('center', 'N/A')}, Content: '{obj.get('content', '')[:50]}...', "
            result += f"Interactivity: {obj.get('interactivity', 'unknown')}\n"
        return result
    
    def _generate_scene_summary(self, detected_objects, inventory=None, game_state=None):
        """Generate general scene summary for MCP tools - delegates to game-specific methods when available"""
        if not detected_objects:
            return {
                "scene_type": "empty",
                "total_objects": 0,
                "message": "No objects detected in current view."
            }
        
        # Check if this is Crafter game and use specialized format
        if hasattr(self, 'config') and self.config.get('game_name') == 'Crafter':
            return self._generate_crafter_scene_summary(detected_objects, inventory, game_state)
        
        # General scene summary for other games (placeholder for future expansion)
        object_counts = {}
        detailed_objects = []
        
        for obj in detected_objects:
            obj_type = obj.get('type', 'unknown')
            obj_name = obj.get('name', obj.get('content', 'N/A'))
            center = obj.get('center', [0, 0])
            
            # Create detailed object entry
            detailed_obj = {
                "id": obj.get('id'),
                "type": obj_type,
                "name": obj_name,
                "center": center,
                "interactivity": obj.get('interactivity', 'unknown'),
                "content": obj.get('content', '')
            }
            detailed_objects.append(detailed_obj)
            
            # Count object types
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Build general scene summary
        scene_data = {
            "scene_type": "general_environment",
            "total_objects": len(detected_objects),
            "detailed_objects": detailed_objects,
            "object_counts": object_counts
        }
        
        # Add inventory information if available
        if inventory:
            scene_data["inventory"] = inventory
        
        # Add game state information if available
        if game_state:
            scene_data["game_state"] = game_state
        
        return scene_data
    
    def _generate_crafter_scene_summary(self, detected_objects, inventory=None, game_state=None):
        """Generate simplified scene summary specifically for Crafter game"""
        # Use grid_position from detected objects if available, otherwise convert coordinates
        def get_grid_position(obj):
            """Get grid position from object, preferring grid_position field over coordinate conversion"""
            # First try to use grid_position from detector (more accurate)
            if 'grid_position' in obj:
                grid_pos = obj['grid_position']
                # grid_position is [row, col], return as tuple
                return (grid_pos[0], grid_pos[1])
            
            # Fallback to coordinate conversion if grid_position not available
            center = obj.get('center', [0, 0])
            # Simple conversion assuming standard Crafter grid layout
            grid_x = max(0, min(6, int(center[0] / 64)))  # Assuming ~64px per grid cell
            grid_y = max(0, min(8, int(center[1] / 64)))  # Assuming ~64px per grid cell
            return (grid_x, grid_y)
        
        # Simplified object categorization for Crafter
        object_counts = {}
        player_info = None
        grid_map = {}  # grid_pos -> [object_types]
        
        # Crafter's three-layer classification (simplified)
        materials = []  # Terrain layer: grass, stone, tree, water, sand, path, coal, iron, diamond, lava
        objects = []    # Item layer: fence, plant, arrow, table, furnace
        creatures = []  # Entity layer: cow, zombie, skeleton, player
        
        for obj in detected_objects:
            obj_type = obj.get('type', 'unknown')
            grid_pos = get_grid_position(obj)
            
            # Initialize grid position if not exists
            if grid_pos not in grid_map:
                grid_map[grid_pos] = []
            
            # Avoid duplicate entries in the same grid position
            if obj_type not in grid_map[grid_pos]:
                grid_map[grid_pos].append(obj_type)
            
            # Count object types
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            # Find player position and direction
            if 'player' in obj_type.lower():
                direction = obj_type.split('-')[-1] if '-' in obj_type else 'unknown'
                player_info = {
                    'position': grid_pos,
                    'direction': direction
                }
                creatures.append(f"{obj_type}@{grid_pos}")
            
            # Categorize by Crafter's three-layer system
            elif obj_type in ['grass', 'stone', 'tree', 'water', 'sand', 'path', 'coal', 'iron', 'diamond', 'lava']:
                materials.append(f"{obj_type}@{grid_pos}")
            elif obj_type in ['fence', 'plant', 'arrow', 'table', 'furnace']:
                objects.append(f"{obj_type}@{grid_pos}")
            elif obj_type in ['cow', 'zombie', 'skeleton']:
                creatures.append(f"{obj_type}@{grid_pos}")
        
        # Build simplified JSON scene summary for Crafter
        scene_data = {
            "scene_type": "crafter_environment",
            "coordinate_system": "IMPORTANT: All positions use (row, col) format where (0,0) is top-left. Movement: UP=row-1, DOWN=row+1, LEFT=col-1, RIGHT=col+1",
            "total_objects": len(detected_objects),
            "grid_size": "7x9",
            "player_info": player_info,
            "object_counts": object_counts,
            # "grid_map": {f"{pos[0]},{pos[1]}": objects for pos, objects in grid_map.items()},
            "layers": {
                "materials": {
                    "count": len(materials),
                    "items": materials
                },
                "objects": {
                    "count": len(objects),
                    "items": objects
                },
                "creatures": {
                    "count": len(creatures),
                    "items": creatures
                }
            }
        }
        
        # Add inventory information (simplified)
        if inventory:
            inv_items = {k: v for k, v in inventory.items() if v > 0}
            scene_data["inventory"] = inv_items
        
        # Add game state information (only key stats)
        if game_state:
            key_stats = {}
            for key, value in game_state.items():
                if key in ['health', 'food', 'water', 'energy']:
                    key_stats[key] = value
            if key_stats:
                scene_data["stats"] = key_stats
        
        # Add important objects summary (exclude grass/empty)
        important_objects = {k: v for k, v in object_counts.items() if k not in ['grass', 'empty']}
        if important_objects:
            scene_data["important_objects"] = important_objects
        
        # Add nearby objects around player (within 1 grid distance)
        if player_info:
            player_pos = player_info['position']
            nearby_objects = []
            for pos, objs in grid_map.items():
                # Check if position is within 1 grid distance from player
                if abs(pos[0] - player_pos[0]) <= 1 and abs(pos[1] - player_pos[1]) <= 1:
                    for obj_type in objs:
                        if obj_type not in ['grass', 'empty']:
                            nearby_objects.append(f"{obj_type}@{pos}")
            if nearby_objects:
                scene_data["nearby_objects"] = nearby_objects
        
        return scene_data
    
    def _format_all_objects_enhanced(self, detected_objects):
        """Enhanced all objects formatting with intelligent categorization"""
        if not detected_objects:
            return "No objects detected in current screen."
        
        # Game-specific filtering for Crafter
        if self.game_name.lower() in ['crafter']:
            return self._format_crafter_objects(detected_objects)
        
        # Categorize objects by likely function for other games
        cards = []
        buttons = []
        ui_elements = []
        text_info = []
        
        for obj in detected_objects:
            content = obj.get('content', '').strip()
            obj_type = obj.get('type', 'unknown')
            center = obj.get('center', [0, 0])
            
            # Categorize based on content and position
            if any(keyword in content.lower() for keyword in ['attack', 'defend', 'block', 'energy', 'damage']):
                cards.append(obj)
            elif any(keyword in content.lower() for keyword in ['end', 'turn', 'skip', 'confirm', 'cancel']):
                buttons.append(obj)
            elif content and len(content) > 0:
                if content.isdigit() or '/' in content:
                    text_info.append(obj)
                else:
                    ui_elements.append(obj)
            else:
                ui_elements.append(obj)
        
        result = f"🎮 ENHANCED OBJECT ANALYSIS ({len(detected_objects)} total):\n\n"
        
        if cards:
            result += f"🃏 CARDS/ACTIONS ({len(cards)}): "
            for card in cards:
                result += f"[{card.get('content', '')[:20]} @{card.get('center')}] "
            result += "\n\n"
        
        if buttons:
            result += f"🔘 BUTTONS ({len(buttons)}): "
            for btn in buttons:
                result += f"[{btn.get('content', '')[:15]} @{btn.get('center')}] "
            result += "\n\n"
        
        if text_info:
            result += f"📊 INFO/STATS ({len(text_info)}): "
            for info in text_info:
                result += f"[{info.get('content', '')[:10]} @{info.get('center')}] "
            result += "\n\n"
        
        if ui_elements:
            result += f"🔧 UI ELEMENTS ({len(ui_elements)}): "
            for elem in ui_elements[:3]:  # Limit to first 3
                result += f"[ID:{elem.get('id')} @{elem.get('center')}] "
            if len(ui_elements) > 3:
                result += f"... +{len(ui_elements)-3} more"
            result += "\n\n"
        
        result += "💡 RECOMMENDATION: Focus on cards/actions for gameplay, buttons for navigation."
        return result
    
    def _format_crafter_objects(self, detected_objects):
        """Format objects specifically for Crafter game, filtering out irrelevant UI elements"""
        if not detected_objects:
            return "No objects detected in current screen."
        
        # Filter objects relevant to Crafter gameplay
        grid_objects = []  # High priority: actual game world objects
        resources = []
        creatures = []
        structures = []
        inventory_items = []
        other_objects = []  # Lower priority: UI elements, etc.
        
        for obj in detected_objects:
            content = obj.get('content', '').strip().lower()
            obj_type = obj.get('type', 'unknown').lower()
            source = obj.get('source', '')
            
            # Skip irrelevant UI elements like buttons, menus, etc.
            if any(skip_word in content for skip_word in ['button', 'menu', 'ui', 'interface', 'panel']):
                continue
            
            # PRIORITY 1: Grid-based objects from direct API (highest priority)
            if source == 'direct_api' or obj.get('grid_position') is not None:
                # Categorize by object type (from Crafter API)
                if obj_type in ['grass', 'path', 'sand', 'lava', 'water']:
                    # Terrain - lower priority within grid objects
                    if obj_type != 'grass':  # Skip grass as it's common
                        grid_objects.append(obj)
                elif obj_type in ['tree', 'stone', 'coal', 'iron', 'diamond']:
                    resources.append(obj)
                elif obj_type in ['cow', 'pig', 'zombie', 'skeleton', 'spider']:
                    creatures.append(obj)
                elif obj_type in ['furnace', 'table', 'plant']:
                    structures.append(obj)
                elif obj_type not in ['empty', 'out_of_bounds']:
                    # Any other non-empty grid object
                    grid_objects.append(obj)
            
            # PRIORITY 2: Content-based categorization (for non-grid objects)
            elif any(resource in content for resource in ['wood', 'stone', 'iron', 'coal', 'diamond', 'water', 'food']):
                resources.append(obj)
            elif any(creature in content for creature in ['cow', 'pig', 'zombie', 'skeleton', 'spider']):
                creatures.append(obj)
            elif any(structure in content for structure in ['tree', 'rock', 'furnace', 'table', 'plant']):
                structures.append(obj)
            elif any(tool in content for tool in ['pickaxe', 'sword', 'axe']):
                inventory_items.append(obj)
            else:
                # Include other objects with valid positions
                if obj.get('center'):
                    other_objects.append(obj)
        
        total_relevant = len(grid_objects + resources + creatures + structures + inventory_items)
        result = f"🎮 CRAFTER WORLD ANALYSIS ({len(detected_objects)} total, {total_relevant} relevant):\n"
        result += "📍 COORDINATE SYSTEM: All positions use (row, col) format where (0,0) is top-left. Movement: UP=row-1, DOWN=row+1, LEFT=col-1, RIGHT=col+1\n\n"
        
        # HIGH PRIORITY: Resources (most important for gameplay)
        if resources:
            result += f"🪨 RESOURCES ({len(resources)}) [HIGH PRIORITY]: "
            for res in resources[:8]:  # Show more resources
                pos = res.get('grid_position', res.get('center', 'unknown'))
                obj_type = res.get('type', res.get('content', 'resource'))
                result += f"[{obj_type} @{pos}] "
            if len(resources) > 8:
                result += f"... +{len(resources)-8} more"
            result += "\n\n"
        
        # HIGH PRIORITY: Creatures (important for interaction)
        if creatures:
            result += f"🐄 CREATURES ({len(creatures)}) [HIGH PRIORITY]: "
            for creature in creatures[:5]:
                pos = creature.get('grid_position', creature.get('center', 'unknown'))
                obj_type = creature.get('type', creature.get('content', 'creature'))
                result += f"[{obj_type} @{pos}] "
            if len(creatures) > 5:
                result += f"... +{len(creatures)-5} more"
            result += "\n\n"
        
        # MEDIUM PRIORITY: Structures
        if structures:
            result += f"🏗️ STRUCTURES ({len(structures)}) [MEDIUM PRIORITY]: "
            for struct in structures[:6]:
                pos = struct.get('grid_position', struct.get('center', 'unknown'))
                obj_type = struct.get('type', struct.get('content', 'structure'))
                result += f"[{obj_type} @{pos}] "
            if len(structures) > 6:
                result += f"... +{len(structures)-6} more"
            result += "\n\n"
        
        # MEDIUM PRIORITY: Grid objects (terrain, etc.)
        if grid_objects:
            result += f"🌍 GRID OBJECTS ({len(grid_objects)}) [MEDIUM PRIORITY]: "
            for obj in grid_objects[:6]:  # Show more grid objects
                pos = obj.get('grid_position', obj.get('center', 'unknown'))
                obj_type = obj.get('type', 'unknown')
                result += f"[{obj_type} @{pos}] "
            if len(grid_objects) > 6:
                result += f"... +{len(grid_objects)-6} more"
            result += "\n\n"
        
        # LOW PRIORITY: Inventory items
        if inventory_items:
            result += f"🔧 TOOLS/ITEMS ({len(inventory_items)}): "
            for item in inventory_items:
                result += f"[{item.get('content', item.get('type', 'item'))}] "
            result += "\n\n"
        
        # LOWEST PRIORITY: Other objects
        if other_objects:
            result += f"🎯 OTHER OBJECTS ({len(other_objects)}): "
            for obj in other_objects[:2]:  # Limit to first 2
                pos = obj.get('grid_position', obj.get('center', 'unknown'))
                result += f"[ID:{obj.get('id')} @{pos}] "
            if len(other_objects) > 2:
                result += f"... +{len(other_objects)-2} more"
            result += "\n\n"
        
        result += "💡 CRAFTER TIP: Focus on RESOURCES (highest priority) for crafting, CREATURES for food/combat, STRUCTURES for interaction. Grid objects show exact world state."
        return result

    def _query_historical_context(self, context_query=None):
        """Query historical MCP conversation context for relevant insights"""
        if not self.mcp_conversation_history:
            return "📚 No historical context available. This is a fresh start - explore the environment to build context."
        
        # If no specific query, return general historical summary
        if not context_query:
            return self._format_general_historical_context()
        
        # Search for relevant historical context based on query
        relevant_contexts = []
        query_lower = context_query.lower()
        
        for session in self.mcp_conversation_history:
            relevance_score = 0
            
            # Check task relevance
            if query_lower in session['task'].lower():
                relevance_score += 3
            
            # Check findings relevance
            for finding in session['key_findings']:
                if query_lower in finding['result_preview'].lower():
                    relevance_score += 2
                if query_lower in finding['query_type'].lower():
                    relevance_score += 1
            
            # Check decision relevance
            if session['final_decision']:
                decision_str = str(session['final_decision']).lower()
                if query_lower in decision_str:
                    relevance_score += 2
            
            if relevance_score > 0:
                relevant_contexts.append((relevance_score, session))
        
        if not relevant_contexts:
            return f"📚 No historical context found for '{context_query}'. Consider exploring this area or using a different query."
        
        # Sort by relevance and format results
        relevant_contexts.sort(key=lambda x: x[0], reverse=True)
        
        result = f"📚 HISTORICAL CONTEXT for '{context_query}' ({len(relevant_contexts)} matches):\n\n"
        
        for i, (score, session) in enumerate(relevant_contexts[:3], 1):  # Show top 3
            result += f"{i}. Task: '{session['task']}' (Score: {score})\n"
            result += f"   - Queries: {session['query_count']}, Decision: {session['final_decision'].get('action_type', 'unknown')}\n"
            
            # Show relevant findings
            relevant_findings = [f for f in session['key_findings'] 
                               if context_query.lower() in f['result_preview'].lower() or 
                                  context_query.lower() in f['query_type'].lower()]
            
            for finding in relevant_findings[:2]:  # Show top 2 relevant findings
                result += f"   - {finding['query_type']}: {finding['result_preview'][:300]}...\n"
            
            result += "\n"
        
        result += "💡 Use these insights to make informed decisions and avoid repeating unsuccessful approaches."
        return result
    
    def _format_general_historical_context(self):
        """Format general historical context summary"""
        if not self.mcp_conversation_history:
            return "📚 No historical context available."
        
        recent_sessions = self.mcp_conversation_history[-3:]  # Last 3 sessions
        
        result = f"📚 RECENT HISTORICAL CONTEXT ({len(recent_sessions)} sessions):\n\n"
        
        for i, session in enumerate(recent_sessions, 1):
            result += f"{i}. '{session['task']}'\n"
            result += f"   - Explored {session['query_count']} aspects\n"
            result += f"   - Decision: {session['final_decision'].get('action_type', 'unknown')}\n"
            
            # Show environment state
            if session['environment_state']:
                env_info = []
                if session['environment_state'].get('objects_detected'):
                    env_info.append('objects detected')
                if session['environment_state'].get('clickable_available'):
                    env_info.append('interactive elements found')
                if env_info:
                    result += f"   - Environment: {', '.join(env_info)}\n"
            
            # Show key findings
            if session['key_findings']:
                top_finding = session['key_findings'][0]
                result += f"   - Key Finding: {top_finding['result_preview'][:80]}...\n"
            
            result += "\n"
        
        result += "💡 PATTERNS: Look for repeated queries or decisions to avoid inefficient exploration."
        return result

    def _format_clickable_objects_enhanced(self, detected_objects):
        """Enhanced clickable objects with action priority"""
        # More intelligent clickable detection
        clickable_objects = []
        for obj in detected_objects:
            interactivity = obj.get('interactivity', 'unknown')
            content = obj.get('content', '').strip()
            obj_type = obj.get('type', 'unknown')
            
            # Enhanced clickable detection logic
            is_clickable = (
                interactivity in ['clickable', 'button'] or
                obj_type in ['button', 'link', 'input', 'clickable'] or
                any(keyword in content.lower() for keyword in ['end', 'turn', 'play', 'use', 'confirm', 'cancel']) or
                (content and len(content) > 0 and interactivity != 'no_effect')
            )
            
            if is_clickable:
                clickable_objects.append(obj)
        
        if not clickable_objects:
            return "❌ No clearly clickable objects detected. Consider exploring with general click or checking all objects."
        
        # Prioritize by likely importance
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for obj in clickable_objects:
            content = obj.get('content', '').lower()
            if any(keyword in content for keyword in ['end', 'turn', 'play', 'attack', 'defend']):
                high_priority.append(obj)
            elif any(keyword in content for keyword in ['confirm', 'cancel', 'skip']):
                medium_priority.append(obj)
            else:
                low_priority.append(obj)
        
        result = f"🎯 CLICKABLE OBJECTS ANALYSIS ({len(clickable_objects)} found):\n\n"
        
        if high_priority:
            result += "🔥 HIGH PRIORITY (Game Actions):\n"
            for obj in high_priority:
                result += f"  • '{obj.get('content', '')[:30]}' @{obj.get('center')} (ID:{obj.get('id')})\n"
            result += "\n"
        
        if medium_priority:
            result += "⚡ MEDIUM PRIORITY (UI Controls):\n"
            for obj in medium_priority:
                result += f"  • '{obj.get('content', '')[:30]}' @{obj.get('center')} (ID:{obj.get('id')})\n"
            result += "\n"
        
        if low_priority:
            result += "📋 OTHER OPTIONS:\n"
            for obj in low_priority[:3]:  # Limit to 3
                result += f"  • '{obj.get('content', '')[:20]}' @{obj.get('center')} (ID:{obj.get('id')})\n"
            if len(low_priority) > 3:
                result += f"  ... +{len(low_priority)-3} more options\n"
        
        result += "\n💡 SUGGESTION: Start with HIGH PRIORITY actions for game progress."
        return result
    
    def _format_text_content_enhanced(self, detected_objects):
        """Enhanced text content with semantic analysis"""
        text_objects = [obj for obj in detected_objects if obj.get('content') and obj.get('content').strip()]
        
        if not text_objects:
            return "No readable text content detected."
        
        # Categorize text by type
        numbers = []
        game_terms = []
        ui_text = []
        
        for obj in text_objects:
            content = obj.get('content', '').strip()
            if content.isdigit() or '/' in content or any(c.isdigit() for c in content):
                numbers.append(obj)
            elif any(term in content.lower() for term in ['attack', 'defend', 'energy', 'block', 'damage', 'health']):
                game_terms.append(obj)
            else:
                ui_text.append(obj)
        
        result = f"📝 TEXT CONTENT ANALYSIS ({len(text_objects)} items):\n\n"
        
        if numbers:
            result += "🔢 NUMBERS/STATS:\n"
            for obj in numbers:
                result += f"  • '{obj.get('content')}' @{obj.get('center')}\n"
            result += "\n"
        
        if game_terms:
            result += "⚔️ GAME TERMS:\n"
            for obj in game_terms:
                result += f"  • '{obj.get('content')}' @{obj.get('center')}\n"
            result += "\n"
        
        if ui_text:
            result += "🖥️ UI TEXT:\n"
            for obj in ui_text[:5]:  # Limit to 5
                result += f"  • '{obj.get('content')[:30]}' @{obj.get('center')}\n"
            if len(ui_text) > 5:
                result += f"  ... +{len(ui_text)-5} more\n"
        
        return result
    
    def _format_object_summary_enhanced(self, detected_objects):
        """Enhanced object summary with game state insights"""
        if not detected_objects:
            return "No objects detected."
        
        # Enhanced categorization
        categories = {
            'cards': 0,
            'buttons': 0,
            'numbers': 0,
            'text': 0,
            'unknown': 0
        }
        
        interactivity_counts = {}
        total_text_chars = 0
        
        for obj in detected_objects:
            content = obj.get('content', '').strip()
            obj_type = obj.get('type', 'unknown')
            interactivity = obj.get('interactivity', 'unknown')
            
            # Enhanced categorization logic
            if any(keyword in content.lower() for keyword in ['attack', 'defend', 'block', 'energy', 'damage']):
                categories['cards'] += 1
            elif any(keyword in content.lower() for keyword in ['end', 'turn', 'skip', 'confirm', 'cancel']):
                categories['buttons'] += 1
            elif content and (content.isdigit() or '/' in content):
                categories['numbers'] += 1
            elif content:
                categories['text'] += 1
            else:
                categories['unknown'] += 1
            
            interactivity_counts[interactivity] = interactivity_counts.get(interactivity, 0) + 1
            total_text_chars += len(content)
        
        result = f"📊 ENHANCED SCREEN SUMMARY:\n"
        result += f"Total objects: {len(detected_objects)}\n"
        result += f"Categories: {dict(categories)}\n"
        result += f"Interactivity: {dict(interactivity_counts)}\n"
        result += f"Text density: {total_text_chars} characters\n\n"
        
        # Game state insights
        if categories['cards'] > 0:
            result += f"🃏 {categories['cards']} card(s)/action(s) available\n"
        if categories['buttons'] > 0:
            result += f"🔘 {categories['buttons']} interactive button(s)\n"
        if categories['numbers'] > 0:
            result += f"📊 {categories['numbers']} stat(s)/counter(s)\n"
        
        # Actionability assessment
        clickable_count = sum(1 for obj in detected_objects 
                            if obj.get('interactivity') not in ['no_effect', 'unknown'])
        
        if clickable_count > 0:
            result += f"\n✅ {clickable_count} potentially actionable objects detected"
        else:
            result += f"\n⚠️ No clearly actionable objects - may need exploration"
        
        return result
    
    def _analyze_game_state(self, detected_objects):
        """Analyze current game state with Crafter-style evaluation system"""
        if not detected_objects:
            return "Cannot analyze game state - no objects detected."
        
        # Initialize Crafter-style analysis
        analysis = {
            'inventory_items': 0,
            'available_resources': [],
            'crafting_opportunities': [],
            'survival_status': 'stable',
            'exploration_potential': 0,
            'achievement_progress': 0
        }
        
        # Analyze detected objects for Crafter-specific elements
        resource_types = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']
        creature_types = ['cow', 'zombie', 'skeleton']
        structure_types = ['table', 'furnace', 'plant']
        
        for obj in detected_objects:
            obj_class = obj.get('class', '').lower()
            content = obj.get('content', '').lower()
            
            # Count available resources
            for resource in resource_types:
                if resource in obj_class or resource in content:
                    if resource not in analysis['available_resources']:
                        analysis['available_resources'].append(resource)
            
            # Detect crafting opportunities
            if any(structure in obj_class for structure in structure_types):
                analysis['crafting_opportunities'].append(obj_class)
            
            # Count exploration potential (new areas, unexplored objects)
            if obj.get('confidence', 0) > 0.8:  # High confidence objects
                analysis['exploration_potential'] += 1
        
        # Analyze inventory if available
        inventory = getattr(detected_objects, 'inventory', {}) if hasattr(detected_objects, 'inventory') else {}
        if isinstance(detected_objects, dict) and 'inventory' in detected_objects:
            inventory = detected_objects['inventory']
        elif isinstance(detected_objects, list) and len(detected_objects) > 0:
            # Try to extract inventory from first object if it's a dict
            first_obj = detected_objects[0]
            if isinstance(first_obj, dict) and 'inventory' in first_obj:
                inventory = first_obj['inventory']
        
        if inventory:
            analysis['inventory_items'] = sum(1 for item, count in inventory.items() if count > 0)
            
            # Calculate survival status based on health/food/drink
            health = inventory.get('health', 9)
            food = inventory.get('food', 9)
            drink = inventory.get('drink', 9)
            
            if health < 3 or food < 3 or drink < 3:
                analysis['survival_status'] = 'critical'
            elif health < 6 or food < 6 or drink < 6:
                analysis['survival_status'] = 'warning'
            else:
                analysis['survival_status'] = 'stable'
            
            # Estimate achievement progress based on inventory diversity
            tool_items = ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
            resource_items = ['wood', 'stone', 'coal', 'iron', 'diamond']
            
            tools_owned = sum(1 for tool in tool_items if inventory.get(tool, 0) > 0)
            resources_owned = sum(1 for resource in resource_items if inventory.get(resource, 0) > 0)
            
            # Simple achievement progress estimation (0-100%)
            analysis['achievement_progress'] = min(100, (tools_owned * 15) + (resources_owned * 10) + (analysis['inventory_items'] * 5))
        
        # Generate Crafter-style evaluation report
        result = f"🎮 CRAFTER EVALUATION:\n\n"
        result += f"📦 Inventory Items: {analysis['inventory_items']}\n"
        result += f"🌲 Available Resources: {len(analysis['available_resources'])} types\n"
        result += f"🔨 Crafting Opportunities: {len(analysis['crafting_opportunities'])}\n"
        result += f"❤️ Survival Status: {analysis['survival_status'].upper()}\n"
        result += f"🗺️ Exploration Potential: {analysis['exploration_potential']} objects\n"
        result += f"🏆 Achievement Progress: {analysis['achievement_progress']:.0f}%\n\n"
        
        # Show available resources
        if analysis['available_resources']:
            result += "🌲 NEARBY RESOURCES:\n"
            for resource in analysis['available_resources'][:5]:  # Show top 5
                result += f"  • {resource.title()}\n"
            result += "\n"
        
        # Show crafting opportunities
        if analysis['crafting_opportunities']:
            result += "🔨 CRAFTING STATIONS:\n"
            for opportunity in set(analysis['crafting_opportunities'][:3]):  # Show unique top 3
                result += f"  • {opportunity.title()}\n"
            result += "\n"
        
        # Strategic recommendations based on Crafter gameplay
        if analysis['survival_status'] == 'critical':
            result += "⚠️ PRIORITY: Find food/water or sleep to restore health!"
        elif analysis['achievement_progress'] < 20:
            result += "🎯 FOCUS: Collect basic resources (wood, stone) and craft tools."
        elif len(analysis['available_resources']) > 3:
            result += "✨ OPPORTUNITY: Rich resource area - consider extended collection."
        elif analysis['inventory_items'] > 8:
            result += "🏗️ STRATEGY: Inventory getting full - consider crafting or placing items."
        else:
            result += "🎮 CONTINUE: Maintain balanced exploration and resource collection."
        
        return result
    
    def _format_clickable_objects(self, detected_objects):
        """Format only clickable objects information"""
        clickable_objects = [obj for obj in detected_objects 
                           if obj.get('interactivity') not in ['no_effect', 'unknown'] or 
                           obj.get('type') in ['button', 'link', 'input', 'clickable']]
        
        if not clickable_objects:
            return "No clearly clickable objects detected."
        
        result = f"Found {len(clickable_objects)} potentially clickable objects:\n"
        for i, obj in enumerate(clickable_objects):
            result += f"{i+1}. ID: {obj.get('id', 'N/A')}, Type: {obj.get('type', 'unknown')}, "
            result += f"Center: {obj.get('center', 'N/A')}, Content: '{obj.get('content', '')[:30]}', "
            result += f"Interactivity: {obj.get('interactivity', 'unknown')}\n"
        return result
    
    def _format_text_content(self, detected_objects):
        """Format text content from detected objects"""
        text_objects = [obj for obj in detected_objects if obj.get('content') and obj.get('content').strip()]
        
        if not text_objects:
            return "No text content detected."
        
        result = f"Text content from {len(text_objects)} objects:\n"
        for i, obj in enumerate(text_objects):
            content = obj.get('content', '').strip()
            if content:
                result += f"{i+1}. '{content}' (Type: {obj.get('type', 'unknown')}, Center: {obj.get('center', 'N/A')})\n"
        return result
    
    def _format_specific_object(self, detected_objects, object_id):
        """Format specific object information"""
        target_obj = None
        for obj in detected_objects:
            if str(obj.get('id')) == str(object_id):
                target_obj = obj
                break
        
        if not target_obj:
            return f"Object with ID '{object_id}' not found."
        
        result = f"Object ID {object_id} details:\n"
        result += f"- Type: {target_obj.get('type', 'unknown')}\n"
        result += f"- Content: '{target_obj.get('content', '')}'\n"
        result += f"- Center coordinates: {target_obj.get('center', 'N/A')}\n"
        result += f"- Bounding box: {target_obj.get('bbox', 'N/A')}\n"
        result += f"- Area: {target_obj.get('area', 'N/A')}\n"
        result += f"- Interactivity: {target_obj.get('interactivity', 'unknown')}\n"
        return result
    
    def _format_object_summary(self, detected_objects):
        """Format a summary of detected objects"""
        if not detected_objects:
            return "No objects detected."
        
        # Count by type
        type_counts = {}
        interactivity_counts = {}
        total_text_chars = 0
        
        for obj in detected_objects:
            obj_type = obj.get('type', 'unknown')
            interactivity = obj.get('interactivity', 'unknown')
            content = obj.get('content', '')
            
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            interactivity_counts[interactivity] = interactivity_counts.get(interactivity, 0) + 1
            total_text_chars += len(content)
        
        result = f"Screen Summary: {len(detected_objects)} total objects\n"
        result += f"Types: {dict(type_counts)}\n"
        result += f"Interactivity: {dict(interactivity_counts)}\n"
        result += f"Total text characters: {total_text_chars}\n"
        return result
    
    def interactive_decision(self, step, task, state, detected_objects, pre_knowledge=None):
        """Interactive decision making - wait for user input instead of AI decision"""
        print(f"\n=== Interactive Decision Mode ===")
        print(f"Task: {task}")
        print(f"Step: {step}")
        
        # Display available objects for user selection
        if detected_objects:
            print("\nAvailable objects to interact with:")
            for i, obj in enumerate(detected_objects):
                print(f"{i+1}. ID: {obj.get('id', 'N/A')}, Type: {obj.get('type', 'unknown')}, ")
                print(f"   Content: '{obj.get('content', '')[:50]}', Center: {obj.get('center', 'N/A')}")
        else:
            print("\nNo objects detected in current state.")
        
        # Return a special marker indicating user interaction is needed
        return {
            "action_type": "user_interaction_required",
            "available_objects": detected_objects,
            "task": task,
            "step": step,
            "state": state
        }
    
    def do_operation_mcp(self, step, task, state, detected_objects, pre_knowledge=None, max_iterations=None):
        """MCP-style operation with multi-turn interaction"""
        # Get max_iterations from config if not provided
        if max_iterations is None:
            max_iterations = self.max_mcp_iter
        
        # Initialize conversation_context with relevant historical context
        conversation_context = []
        
        # Get relevant historical context from previous MCP sessions
        historical_contexts = self._get_relevant_historical_context(task)
        if historical_contexts:
            # Add historical context as initial context
            historical_summary = self._build_historical_context_summary(historical_contexts)
            conversation_context.append({
                'iteration': 0,
                'query': {'query_type': 'historical_context', 'task': task},
                'result': historical_summary
            })
            print(f"Initialized MCP with {len(historical_contexts)} relevant historical contexts")
        
        for iteration in range(max_iterations):
            print(f"MCP Iteration {iteration + 1}/{max_iterations}")
            
            # Update prompt with current iteration context
            current_prompt = self._build_mcp_prompt(task, conversation_context, iteration, max_iterations, pre_knowledge)
            
            # Debug: Print simplified MCP iteration info (detailed prompt removed to reduce noise)
            print(f"🔄 MCP Iteration {iteration + 1}/{max_iterations}: Building prompt and calling LLM...")
            
            # Call LLM with MCP tools with timeout protection
            # Only pass image on first iteration or when specifically needed for visual queries
            # This improves efficiency by relying on structured JSON data from environment queries
            if iteration == 0 or any('scene_summary' in str(ctx.get('query', {})) for ctx in conversation_context[-2:]):
                imgs_64 = [cv_to_base64(state['screen'])]
            else:
                imgs_64 = []
            
            try:
                response = self.base_model.call_text_images(
                    current_prompt, 
                    imgs_64, 
                    self.mcp_interaction_tools, 
                    pre_knowledge=pre_knowledge
                )
                
            except Exception as api_error:
                print(f"❌ MCP API call failed: {api_error}")
                print(f"⚠️ Falling back to random exploration due to API error")
                
                # Fallback to random exploration when API fails
                clickable_objects = [obj for obj in detected_objects if obj.get('interactivity') == 'clickable']
                if not clickable_objects:
                    clickable_objects = detected_objects
                
                if clickable_objects:
                    import random
                    selected_obj = random.choice(clickable_objects)
                    operations = ['Click', 'RightSingle', 'LeftDouble']
                    selected_operation = random.choice(operations)
                    
                    print(f"🎲 API Fallback: {selected_operation} on object {selected_obj.get('id', 'unknown')} at {selected_obj['center']}")
                    
                    return {
                        "action_type": "direct_operation",
                        "operate": selected_operation,
                        "params": {'x': selected_obj['center'][0], 'y': selected_obj['center'][1]},
                        "object_id": selected_obj.get('id'),
                        "conversation_context": conversation_context,
                        "fallback_reason": "api_error"
                    }
                else:
                    print("❌ No objects available for fallback")
                    return None
            
            print(f"MCP Response: {response}")
            
            # Log token usage if available
            if response and 'usage' in response:
                self.logger.log({
                    f"eval/tokens/mcp_iteration_{iteration+1}/input": response['usage']['input'],
                    f"eval/tokens/mcp_iteration_{iteration+1}/output": response['usage']['output'],
                    f"eval/tokens/mcp_iteration_{iteration+1}/total": response['usage']['total']
                }, step)
            
            if response['function'] is None:
                print("No function call in MCP response")
                return None
            
            function_name = response['function']['name']
            function_input = response['function']['input']
            
            if function_name == 'query_environment':
                # Execute environment query
                query_result = self.execute_environment_query(
                    function_input['query_type'],
                    detected_objects,
                    function_input.get('object_id'),
                    function_input.get('context_query')
                )
                
                # Add to conversation context
                conversation_context.append({
                    'iteration': iteration + 1,
                    'query': function_input,
                    'result': query_result
                })
                
                # Continue to next iteration (prompt will be updated in the loop)
                continue
                
            elif function_name == 'select_skill':
                # Execute skill selection
                print(f"MCP selected skill ID: {function_input['id']}")
                
                # Save conversation to history before returning
                final_decision = {
                    "action_type": "select_skill",
                    "skill_id": int(function_input['id'])
                }
                self._save_conversation_to_history(conversation_context, task, final_decision)
                
                return {
                    "action_type": "select_skill",
                    "skill_id": int(function_input['id']),
                    "conversation_context": conversation_context
                }
                
            elif function_name in ['Click', 'RightSingle', 'LeftDouble', 'Type', 'Drag', 'Finished'] or \
                 function_name in ['move_up', 'move_down', 'move_left', 'move_right', 'interact', 'sleep', 
                                   'place_table', 'place_stone', 'place_furnace', 'plant_sapling',
                                   'craft_wood_pickaxe', 'craft_stone_pickaxe', 'craft_iron_pickaxe',
                                   'craft_wood_sword', 'craft_stone_sword', 'craft_iron_sword']:
                # Execute direct operation (including Crafter keyboard actions)
                print(f"MCP selected operation: {function_name}")
                
                # CRITICAL: Validate tool requirements for interact action in Crafter
                if function_name == 'interact' and state and 'inventory' in state:
                    inventory = state['inventory']
                    
                    # Check if trying to interact with resources that require tools
                    nearby_objects = [obj for obj in detected_objects if obj.get('class', '').lower() in ['stone', 'coal', 'iron', 'diamond']]
                    
                    if nearby_objects:
                        resource_type = nearby_objects[0].get('class', '').lower()
                        required_tool = None
                        
                        if resource_type in ['stone', 'coal']:
                            required_tool = 'wood_pickaxe'
                        elif resource_type == 'iron':
                            required_tool = 'stone_pickaxe'
                        elif resource_type == 'diamond':
                            required_tool = 'iron_pickaxe'
                        
                        if required_tool and inventory.get(required_tool, 0) == 0:
                            print(f"❌ TOOL CHECK FAILED: Cannot collect {resource_type} without {required_tool}")
                            print(f"📦 Current inventory: {inventory}")
                            
                            # Add warning to conversation context and continue to next iteration
                            conversation_context.append({
                                'iteration': iteration + 1,
                                'query': {'query_type': 'tool_validation', 'action': function_name, 'target': resource_type},
                                'result': f"VALIDATION FAILED: Cannot collect {resource_type} without {required_tool}. Current inventory: {inventory}. You must craft the required tool first or find a different target."
                            })
                            continue
                
                # Try to find the corresponding object_id based on coordinates
                selected_object_id = None
                if 'x' in function_input and 'y' in function_input:
                    target_x, target_y = function_input['x'], function_input['y']
                    # Find the closest object to the selected coordinates
                    min_distance = float('inf')
                    closest_object = None
                    
                    for obj in detected_objects:
                        if 'center' in obj and obj['center']:
                            obj_x, obj_y = obj['center']
                            distance = ((target_x - obj_x) ** 2 + (target_y - obj_y) ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                closest_object = obj
                    
                    # If we found a close object (within 50 pixels), use its ID
                    if closest_object and min_distance <= 50:
                        selected_object_id = closest_object.get('id')
                        print(f"MCP matched operation to object ID: {selected_object_id} (distance: {min_distance:.1f})")
                    else:
                        print(f"MCP operation at ({target_x}, {target_y}) - no close object found (min distance: {min_distance:.1f})")
                
                # Save conversation to history before returning
                final_decision = {
                    "action_type": "direct_operation",
                    "operate": function_name,
                    "params": function_input,
                    "object_id": selected_object_id
                }
                self._save_conversation_to_history(conversation_context, task, final_decision)
                
                return {
                    "action_type": "direct_operation",
                    "operate": function_name,
                    "params": function_input,
                    "object_id": selected_object_id,
                    "conversation_context": conversation_context
                }
            
            else:
                print(f"Unknown function: {function_name}")
                return None
        
        print(f"MCP max iterations ({max_iterations}) reached without final decision. Falling back to intelligent decision.")
        
        # Get clickable objects for random selection
        clickable_objects = [obj for obj in detected_objects if obj.get('interactivity') == 'clickable']
        
        # If no clickable objects, use all objects
        if not clickable_objects:
            clickable_objects = detected_objects
        
        if clickable_objects:
            # Randomly select an object
            selected_obj = random.choice(clickable_objects)
            
            # Randomly select an operation type
            operations = ['Click', 'RightSingle', 'LeftDouble']
            selected_operation = random.choice(operations)
            
            print(f"Random exploration: {selected_operation} on object {selected_obj.get('id', 'unknown')} at {selected_obj['center']}")
            
            # Save conversation to history before returning (fallback case)
            final_decision = {
                "action_type": "direct_operation",
                "operate": selected_operation,
                "params": {"coordinate": selected_obj['center']},
                "object_id": selected_obj.get('id'),
                "fallback_decision": True
            }
            self._save_conversation_to_history(conversation_context, task, final_decision)
            
            return {
                "action_type": "direct_operation",
                "operate": selected_operation,
                "params": {"coordinate": selected_obj['center']},
                "object_id": selected_obj.get('id'),
                "conversation_context": conversation_context,
                "fallback_decision": True,
                "selected_object": selected_obj
            }
        else:
            # Ultimate fallback if no objects detected
            print("No objects detected, using center screen click as last resort")
            
            # Save conversation to history before returning (ultimate fallback)
            final_decision = {
                "action_type": "direct_operation",
                "operate": "Click",
                "params": {"coordinate": [640, 360]},
                "object_id": None,
                "fallback_decision": True
            }
            self._save_conversation_to_history(conversation_context, task, final_decision)
            
            return {
                "action_type": "direct_operation",
                "operate": "Click",
                "params": {"coordinate": [640, 360]},
                "object_id": None,
                "conversation_context": conversation_context,
                "fallback_decision": True,
                "fallback_reason": "no_objects_detected"
            }
    
    def _build_mcp_prompt(self, task, conversation_context, iteration_count, max_iterations, pre_knowledge=None):
        """Build enhanced MCP interaction prompt with intelligent context analysis and game-specific knowledge"""
        remaining_iterations = max_iterations - iteration_count
        
        # Analyze conversation context for intelligent decision making
        context_analysis = self._analyze_conversation_context(conversation_context)
        
        base_prompt = f"""You are an intelligent agent playing '{self.game_name}' and your task is: '{task}'.

🎯 CURRENT SITUATION:
- Iteration: {iteration_count + 1}/{max_iterations} (⚠️ {remaining_iterations} remaining)
- Previous queries: {len(conversation_context)}
{context_analysis['status_summary']}

🔧 AVAILABLE TOOLS:
1. query_environment - Get real-time UI information (use strategically, avoid repetition)
2. select_skill - Choose from learned skills when applicable
3. Direct actions:
{self._get_game_specific_direct_actions()}

💡 OPERATION GUIDANCE:
{self._get_game_specific_operation_guidance()}

🧠 DECISION STRATEGY:
{context_analysis['decision_guidance']}

⚡ EFFICIENCY RULES:
- AVOID repeating identical queries (especially 'all_objects')
- If you've seen the same objects multiple times, MAKE A DECISION
- Use specific queries (clickable_objects, text_content) over general ones
- With {remaining_iterations} iterations left, prioritize ACTION over exploration
- Don't hesitate to take direct action when you have sufficient information
- MANDATORY: You MUST select a concrete action (skill or direct operation) within {max_iterations} iterations
- If uncertain after {max_iterations-1} iterations, choose the most reasonable action based on available information
- ⚠️ CRITICAL FOR CRAFTER: When you see trees, stones, or resources nearby, use 'interact' action immediately
- ⚠️ STOP QUERYING if you've already seen the environment - take action instead!"""
        
        # Add game-specific knowledge if available
        if pre_knowledge:
            base_prompt += f"\n\n🎮 GAME KNOWLEDGE:\n{pre_knowledge}"
        
        base_prompt += "\n\nAnalyze the current situation and choose the most appropriate action to complete the task."
        
        # Add condensed context history if available
        if conversation_context:
            base_prompt += "\n\n📋 CONTEXT SUMMARY:\n"
            base_prompt += context_analysis['context_summary']
        
        return base_prompt
    
    def _get_game_specific_operation_guidance(self):
        """Get operation guidance specific to the current game"""
        if self.game_name.lower() in ['slay the spire', 'sts']:
            return """- For CARD GAMES: Use Drag to play cards on specific targets (enemies, self)
- For UI ELEMENTS: Use Click for buttons, menus, selections
- PREFER Drag over Click when moving objects to specific destinations
- Drag is more efficient than Click+Click sequences for card targeting"""
        elif self.game_name.lower() in ['crafter']:
            return """- For KEYBOARD GAMES: Use keyboard actions for movement and interaction
- WASD: Movement controls with coordinate system understanding:
  * W (up): Decreases row coordinate (row-1) - moves toward smaller row numbers
  * A (left): Decreases column coordinate (col-1) - moves toward smaller column numbers
  * S (down): Increases row coordinate (row+1) - moves toward larger row numbers
  * D (right): Increases column coordinate (col+1) - moves toward larger column numbers
- Player facing direction follows same coordinate logic
- SPACE/INTERACT: Primary interaction (collect, attack, eat)
  * Use 'interact' action when facing trees, stones, creatures, or any collectible resources
  * CRITICAL: When you see trees, stones, or other materials and you're adjacent to them, use 'interact' to collect
  * This is the ONLY way to gather resources in Crafter - you must use interact/SPACE when facing materials
  * RESOURCE COLLECTION REQUIREMENTS:
    - Trees → wood (no tool required)
    - Stone/Coal → requires wood pickaxe (craft first: 1 wood + table)
    - Iron → requires stone pickaxe (craft first: 1 wood + 1 stone + table)
    - Diamond → requires iron pickaxe (craft first: 1 wood + 1 coal + 1 iron + table + furnace)
  * NEVER attempt to collect stone/coal without wood pickaxe in inventory
  * NEVER attempt to collect iron without stone pickaxe in inventory
  * NEVER attempt to collect diamond without iron pickaxe in inventory
- TAB: Sleep to restore energy
- Number keys 1-6: Craft tools and weapons
- Letter keys T,R,F,P: Place structures (table, stone, furnace, plant)
- CRITICAL GAME MECHANICS:
  * Player character is ALWAYS at the center of the screen
  * Objects/targets move toward the player through player movement, not the other way around
  * When you want to reach an object, move in the direction that brings the object closer to the center
  * Short-term goals should be achieved by making targets approach the player's central position
  * UP=row-1, DOWN=row+1, LEFT=col-1, RIGHT=col+1
  * RESOURCE COLLECTION: Always use 'interact' when adjacent to trees, stones, or other collectibles"""
        else:
            return """- For UI ELEMENTS: Use Click for buttons, menus, selections
- Use Drag when moving objects to specific destinations
- Use Type for text input when needed
- Choose the most appropriate action based on the game interface"""
    
    def _get_game_specific_direct_actions(self):
        """Get direct actions available for the current game"""
        if self.game_name.lower() in ['slay the spire', 'sts']:
            return """   - Click: Select/activate UI elements, buttons, or single-target actions
   - Drag: Move cards to targets (enemies/self), drag items between locations
   - RightSingle, LeftDouble: Alternative click interactions
   - Type: Input text when needed"""
        elif self.game_name.lower() in ['crafter']:
            return """   - KeyboardAction: Use WASD for movement, SPACE for interaction, TAB for sleep
   - CraftAction: Use number keys 1-6 for crafting tools and weapons
   - PlaceAction: Use T,R,F,P for placing structures (table, stone, furnace, plant)"""
        else:
            return """   - Click: Select/activate UI elements, buttons, or single-target actions
   - Drag: Move objects to specific destinations
   - Type: Input text when needed
   - RightSingle, LeftDouble: Alternative interactions"""
    
    def _save_conversation_to_history(self, conversation_context, task, final_decision):
        """Save valuable conversation context to persistent history"""
        if not conversation_context:
            return
            
        # Create a summary of this conversation session
        session_summary = {
            'task': task,
            'timestamp': time.time(),
            'query_count': len(conversation_context),
            'final_decision': final_decision,
            'key_findings': [],
            'environment_state': {}
        }
        
        # Extract key findings from conversation
        for ctx in conversation_context:
            query_type = ctx['query'].get('query_type', 'unknown')
            result = ctx['result']
            if isinstance(result, dict):
                result_str = str(result)
                result_preview = result_str[:200] if len(result_str) > 200 else result_str
            else:
                result_preview = result[:200] if len(result) > 200 else result
            
            session_summary['key_findings'].append({
                'query_type': query_type,
                'result_preview': result_preview,
                'iteration': ctx['iteration']
            })
            
            # Store environment state information
            if query_type == 'all_objects' and 'objects' in ctx['result']:
                session_summary['environment_state']['objects_detected'] = True
            elif query_type == 'clickable_objects':
                session_summary['environment_state']['clickable_available'] = True
        
        # Add to history and maintain size limit
        self.mcp_conversation_history.append(session_summary)
        if len(self.mcp_conversation_history) > self.max_history_length:
            self.mcp_conversation_history.pop(0)  # Remove oldest entry
        
        print(f"💾 Saved MCP session to history: {len(conversation_context)} interactions, decision: {final_decision.get('action_type', 'unknown')}")
    
    def _get_relevant_historical_context(self, current_task, max_contexts=3):
        """Retrieve relevant historical context for current task"""
        if not self.mcp_conversation_history:
            return []
        
        # Simple relevance scoring based on task similarity and recency
        scored_contexts = []
        current_time = time.time()
        
        for i, session in enumerate(self.mcp_conversation_history):
            # Recency score (more recent = higher score)
            recency_score = 1.0 / (1.0 + (current_time - session['timestamp']) / 3600)  # Decay over hours
            
            # Task similarity score (simple keyword matching)
            task_similarity = 0.5  # Default moderate similarity
            if session['task'].lower() in current_task.lower() or current_task.lower() in session['task'].lower():
                task_similarity = 1.0
            
            # Quality score based on number of findings
            quality_score = min(1.0, len(session['key_findings']) / 5.0)  # Normalize to 0-1
            
            total_score = (recency_score * 0.4 + task_similarity * 0.4 + quality_score * 0.2)
            scored_contexts.append((total_score, session, i))
        
        # Sort by score and return top contexts
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        return [ctx[1] for ctx in scored_contexts[:max_contexts]]
    
    def _build_historical_context_summary(self, historical_contexts):
        """Build a concise summary from historical contexts"""
        if not historical_contexts:
            return ""
        
        summary = "\n🕒 RELEVANT HISTORICAL CONTEXT:\n"
        
        for i, ctx in enumerate(historical_contexts, 1):
            summary += f"\n{i}. Previous Task: '{ctx['task']}'\n"
            summary += f"   - Explored {ctx['query_count']} aspects of environment\n"
            
            if ctx['final_decision']:
                action_type = ctx['final_decision'].get('action_type', 'unknown')
                summary += f"   - Final Decision: {action_type}\n"
            
            # Add key environment findings
            if ctx['environment_state']:
                env_info = []
                if ctx['environment_state'].get('objects_detected'):
                    env_info.append('objects detected')
                if ctx['environment_state'].get('clickable_available'):
                    env_info.append('clickable elements found')
                if env_info:
                    summary += f"   - Environment: {', '.join(env_info)}\n"
            
            # Add most relevant findings
            if ctx['key_findings']:
                top_findings = ctx['key_findings'][:2]  # Show top 2 findings
                for finding in top_findings:
                    summary += f"   - Found: {finding['result_preview'][:300]}...\n"
        
        summary += "\n💡 Use this context to avoid redundant exploration and make informed decisions.\n"
        return summary
    
    def enhance_pre_knowledge_with_history(self, base_pre_knowledge, task):
        """Enhance pre_knowledge with relevant historical MCP findings"""
        enhanced_knowledge = base_pre_knowledge
        
        # Get relevant historical contexts
        historical_contexts = self._get_relevant_historical_context(task, max_contexts=2)
        
        if historical_contexts:
            enhanced_knowledge += "\n\n=== HISTORICAL INSIGHTS ==="
            
            # Extract key findings from historical contexts
            key_findings = set()
            environment_patterns = set()
            
            for context in historical_contexts:
                # Extract key findings
                if 'key_findings' in context and context['key_findings']:
                    for finding in context['key_findings']:
                        if len(finding) > 10:  # Filter out very short findings
                            key_findings.add(finding)
                
                # Extract environment patterns
                if 'environment_state' in context and context['environment_state']:
                    env_state = context['environment_state']
                    if isinstance(env_state, str) and len(env_state) > 20:
                        environment_patterns.add(env_state[:200])  # Limit length
            
            # Add key findings to enhanced knowledge
            if key_findings:
                enhanced_knowledge += "\n\nKey Findings from Previous Sessions:"
                for i, finding in enumerate(list(key_findings)[:3], 1):  # Limit to top 3
                    enhanced_knowledge += f"\n{i}. {finding}"
            
            # Add environment patterns
            if environment_patterns:
                enhanced_knowledge += "\n\nEnvironment Patterns Observed:"
                for i, pattern in enumerate(list(environment_patterns)[:2], 1):  # Limit to top 2
                    enhanced_knowledge += f"\n{i}. {pattern}"
            
            enhanced_knowledge += "\n\nNote: Use these insights to make more informed decisions and avoid repeating unsuccessful approaches."
        
        return enhanced_knowledge
    
    def _analyze_conversation_context(self, conversation_context):
        """Analyze conversation context to provide intelligent guidance"""
        if not conversation_context:
            return {
                'status_summary': '- Fresh start, no previous context',
                'decision_guidance': 'Start with a strategic environment query to understand available options.',
                'context_summary': ''
            }
        
        # Check if we have historical context as the first entry
        has_historical_context = (len(conversation_context) > 0 and 
                                conversation_context[0]['query'].get('query_type') == 'historical_context')
        
        # Filter out historical context for pattern analysis (focus on current session queries)
        current_session_context = conversation_context[1:] if has_historical_context else conversation_context
        
        # If only historical context exists, provide enhanced guidance with increased character limit
        if has_historical_context and not current_session_context:
            # Handle both string and dict results
            result = conversation_context[0]['result']
            if isinstance(result, dict):
                context_text = str(result)[:600] + '...' if len(str(result)) > 600 else str(result)  # Increased from 300 to 600 chars
            else:
                context_text = result[:600] + '...' if len(result) > 600 else result  # Increased from 300 to 600 chars
            
            return {
                'status_summary': '- Initialized with historical insights from previous sessions',
                'decision_guidance': 'You have valuable historical context. Use these insights to make informed decisions and avoid repeating unsuccessful approaches. Start with targeted queries based on historical patterns.',
                'context_summary': context_text
            }
        
        # Analyze query patterns (use current session context for pattern analysis)
        analysis_context = current_session_context if current_session_context else conversation_context
        query_types = [ctx['query'].get('query_type', 'unknown') for ctx in analysis_context]
        query_counts = {}
        for qt in query_types:
            query_counts[qt] = query_counts.get(qt, 0) + 1
        
        # Detect repetitive behavior
        repeated_queries = [qt for qt, count in query_counts.items() if count > 2]
        last_3_queries = query_types[-3:] if len(query_types) >= 3 else query_types
        
        # Generate status summary
        status_summary = ""
        if has_historical_context:
            status_summary += "- Enhanced with historical insights\n"
        status_summary += f"- Current session queries: {dict(query_counts)}"
        if repeated_queries:
            status_summary += f"\n- ⚠️ REPETITIVE: {repeated_queries} (stop repeating!)"
        
        # Generate decision guidance
        decision_guidance = ""
        if 'all_objects' in repeated_queries:
            decision_guidance += "🚨 You've queried 'all_objects' multiple times. You know what's on screen - TAKE ACTION!\n"
        
        if len(set(last_3_queries)) == 1:
            decision_guidance += "🚨 Last 3 queries were identical. Break the loop - try a different approach or make a decision!\n"
        
        # Adjust thresholds based on whether we have historical context
        exploration_threshold = 3 if has_historical_context else 4
        if len(analysis_context) > exploration_threshold:
            decision_guidance += "⏰ You've explored enough. Time to make a decision based on available information.\n"
        
        # Add historical context guidance
        if has_historical_context and not decision_guidance:
            decision_guidance = "Leverage historical insights to make informed decisions. Focus on targeted exploration based on past learnings."
        elif not decision_guidance:
            decision_guidance = "Continue strategic exploration, but be ready to act when you have enough information."
        
        # Generate condensed context summary
        context_summary = ""
        
        # Include historical context summary if available with increased character limit
        if has_historical_context:
            result = conversation_context[0]['result']
            if isinstance(result, dict):
                historical_summary = str(result)[:500]  # Increased from 200 to 500 chars
            else:
                historical_summary = result[:500]  # Increased from 200 to 500 chars
            context_summary += f"📚 Historical Context: {historical_summary}...\n\n"
        
        # Add current session context with increased character limits
        unique_results = set()
        context_to_summarize = analysis_context[-4:] if analysis_context else []  # Last 4 current session interactions
        for ctx in context_to_summarize:
            result = ctx['result']
            if isinstance(result, dict):
                result_str = str(result)
                result_key = result_str[:300]  # Increased from 200 to 300 chars as key
                result_display = result_str[:800] + '...' if len(result_str) > 800 else result_str  # Increased from 400 to 800 chars
            else:
                result_key = result[:300]  # Increased from 200 to 300 chars as key
                result_display = result[:3000] + '...' if len(result) > 800 else result  # Increased from 400 to 800 chars
            
            if result_key not in unique_results:
                unique_results.add(result_key)
                context_summary += f"• {ctx['query']} → {result_display}\n"
        
        return {
            'status_summary': status_summary,
            'decision_guidance': decision_guidance.strip(),
            'context_summary': context_summary
        }
    
    def _extract_fallback_action(self, response, state):
        """Extract a reasonable fallback action when MCP doesn't provide a clear decision"""
        try:
            # Try to extract any action mentioned in the response text
            response_text = response.get('message', '').lower()
            
            # Look for common action keywords
            if 'click' in response_text or 'select' in response_text:
                return {
                    "action_type": "direct_operation",
                    "operate": "Click",
                    "params": {"coordinate": [400, 300]},  # Center of screen
                    "fallback_reason": "Extracted click action from response"
                }
            elif 'move' in response_text:
                # Default to move_right for Crafter
                return {
                    "action_type": "direct_operation", 
                    "operate": "move_right",
                    "params": {},
                    "fallback_reason": "Extracted movement action from response"
                }
            elif 'interact' in response_text:
                return {
                    "action_type": "direct_operation",
                    "operate": "interact", 
                    "params": {},
                    "fallback_reason": "Extracted interact action from response"
                }
            else:
                # Default safe action - just observe/sleep
                return {
                    "action_type": "direct_operation",
                    "operate": "sleep",
                    "params": {},
                    "fallback_reason": "No clear action found, defaulting to safe action"
                }
        except Exception as e:
            print(f"Error in fallback action extraction: {e}")
            return {
                "action_type": "direct_operation",
                "operate": "sleep",
                "params": {},
                "fallback_reason": "Error in extraction, using safe default"
            }