import sqlite3

from base_model import creat_base_model
from . import Prompt
from . import FunctionCalls
from utils.utils import cv_to_base64
from .LongMemory import LongMemory
from .visualizer import push_data
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

class Brain:
    def __init__(self, config, detector, logger):
        self.game_name = config['game_name']
        self.model_name = config['brain']['base_model']
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
        
        # MCP-style interaction tools
        self.environment_query_tools = FunctionCalls.environment_query_tools(config['brain']['base_model'])
        self.mcp_interaction_tools = FunctionCalls.mcp_interaction_tools(config['brain']['base_model'])

        self.uct_c = config['brain']['uct_c']
        self.uct_threshold = config['brain']['uct_threshold']
        self.max_mcp_iter = config.get('mcp', {}).get('max_iter', 8)

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
                uct = skill['fitness'] + self.uct_c * np.sqrt(np.log(num_total) / skill['num'])
                ucts.append(uct)

            if not close_exploration:
                ucts.append(self.uct_threshold + self.uct_c * np.sqrt(np.log(num_total) / skill_cluster['explore_nums']))
                candidate_skills.append({'id': 'Explore', 'name': 'Explore', 'num': skill_cluster['explore_nums'], 
                                         'fitness': self.uct_threshold, 'mcts_node_id': mcts_node_id})
            
            ucts = np.array(ucts)
            temperature = max(self.temperature(num_total), 1e-2)
            scaled_ucts = ucts / temperature
            scaled_ucts -= np.max(scaled_ucts)  
            exp_ucts = np.exp(scaled_ucts)
            exp_ucts = np.clip(exp_ucts, 1e-10, None)
            probs = exp_ucts / np.sum(exp_ucts)

            print(f"ucts: {ucts}, exp_ucts: {exp_ucts}, num: {num_total}, temp: {temperature}")
            
            probs = exp_ucts / np.sum(exp_ucts)
            print(f"probs: {probs}")

            
            for i, skill in enumerate(candidate_skills):
                skills_info.append({'id': skill['id'], 'name': skill['name'], 
                                         'num': skill['num'], 'fitness': skill['fitness'], 'prob': round(probs[i],2)})
                  

            selected_skill = np.random.choice(candidate_skills, p=probs)

            push_data({'candidate_skills': skills_info, 'temperature': round(temperature,2), 'selected_skill_id': selected_skill['id']})

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
        push_data({'delete_ids': delete_ids})
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
            else:
                # Handle other operation types without coordinates
                operations_str += 'operation' + str(idx) + ': ' + operation['operate'] + '\n'
        operations[-1].pop('params', None)
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
                                            operations, 0, 1, state_id, mcst_node_id, obs[0]['screen'], obs[-1]['screen'])
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
                    operations, -1, 0, state_id, mcst_node_id, 
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
    
    def execute_environment_query(self, query_type, detected_objects, object_id=None):
        """Execute enhanced environment query with game-specific intelligence"""
        if query_type == "all_objects":
            return self._format_all_objects_enhanced(detected_objects)
        elif query_type == "clickable_objects":
            return self._format_clickable_objects_enhanced(detected_objects)
        elif query_type == "text_content":
            return self._format_text_content_enhanced(detected_objects)
        elif query_type == "specific_object" and object_id:
            return self._format_specific_object(detected_objects, object_id)
        elif query_type == "object_summary":
            return self._format_object_summary_enhanced(detected_objects)
        elif query_type == "game_state":
            return self._analyze_game_state(detected_objects)
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
    
    def _format_all_objects_enhanced(self, detected_objects):
        """Enhanced all objects formatting with intelligent categorization"""
        if not detected_objects:
            return "No objects detected in current screen."
        
        # Categorize objects by likely function
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
        """Analyze current game state for strategic decision making"""
        if not detected_objects:
            return "Cannot analyze game state - no objects detected."
        
        analysis = {
            'phase': 'unknown',
            'available_actions': [],
            'resources': {},
            'threats': [],
            'opportunities': []
        }
        
        # Analyze objects for game state clues
        for obj in detected_objects:
            content = obj.get('content', '').strip().lower()
            center = obj.get('center', [0, 0])
            
            # Detect game phase
            if 'end turn' in content:
                analysis['phase'] = 'player_turn'
            elif 'enemy turn' in content or 'opponent' in content:
                analysis['phase'] = 'enemy_turn'
            
            # Detect available actions
            if any(action in content for action in ['attack', 'defend', 'play', 'use']):
                analysis['available_actions'].append({
                    'action': content[:20],
                    'position': center,
                    'id': obj.get('id')
                })
            
            # Detect resources (energy, health, etc.)
            if '/' in content and any(c.isdigit() for c in content):
                analysis['resources'][f'resource_{len(analysis["resources"])}'] = content
            
            # Detect threats/opportunities
            if any(threat in content for threat in ['damage', 'attack', 'vulnerable']):
                analysis['threats'].append(content[:30])
            elif any(opp in content for opp in ['heal', 'block', 'energy', 'draw']):
                analysis['opportunities'].append(content[:30])
        
        # Generate strategic summary
        result = f"🎮 GAME STATE ANALYSIS:\n\n"
        result += f"Phase: {analysis['phase'].upper()}\n"
        result += f"Available Actions: {len(analysis['available_actions'])}\n"
        result += f"Resources Detected: {len(analysis['resources'])}\n"
        result += f"Threats: {len(analysis['threats'])}\n"
        result += f"Opportunities: {len(analysis['opportunities'])}\n\n"
        
        if analysis['available_actions']:
            result += "⚔️ AVAILABLE ACTIONS:\n"
            for action in analysis['available_actions'][:3]:
                result += f"  • {action['action']} @{action['position']} (ID:{action['id']})\n"
            result += "\n"
        
        if analysis['resources']:
            result += "💎 RESOURCES:\n"
            for res_name, res_value in analysis['resources'].items():
                result += f"  • {res_value}\n"
            result += "\n"
        
        # Strategic recommendation
        if analysis['phase'] == 'player_turn' and analysis['available_actions']:
            result += "💡 STRATEGIC RECOMMENDATION: It's your turn - consider taking an action!"
        elif analysis['threats']:
            result += "⚠️ STRATEGIC RECOMMENDATION: Threats detected - consider defensive actions."
        elif analysis['opportunities']:
            result += "✨ STRATEGIC RECOMMENDATION: Opportunities available - consider capitalizing!"
        else:
            result += "🤔 STRATEGIC RECOMMENDATION: Analyze available options and make a strategic choice."
        
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
    
    def do_operation_mcp(self, step, task, state, detected_objects, pre_knowledge=None, max_iterations=None):
        """MCP-style operation with multi-turn interaction"""
        # Get max_iterations from config if not provided
        if max_iterations is None:
            max_iterations = self.max_mcp_iter
        
        conversation_context = []
        
        for iteration in range(max_iterations):
            print(f"MCP Iteration {iteration + 1}/{max_iterations}")
            
            # Update prompt with current iteration context
            current_prompt = self._build_mcp_prompt(task, conversation_context, iteration, max_iterations)
            
            # Call LLM with MCP tools
            imgs_64 = [cv_to_base64(state['screen'])]
            response = self.base_model.call_text_images(
                current_prompt, 
                imgs_64, 
                self.mcp_interaction_tools, 
                pre_knowledge=pre_knowledge
            )
            
            print(f"MCP Response: {response}")
            
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
                    function_input.get('object_id')
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
                return {
                    "action_type": "select_skill",
                    "skill_id": int(function_input['id']),
                    "conversation_context": conversation_context
                }
                
            elif function_name in ['Click', 'RightSingle', 'LeftDouble', 'Type', 'Drag', 'Finished']:
                # Execute direct operation
                print(f"MCP selected operation: {function_name}")
                return {
                    "action_type": "direct_operation",
                    "operate": function_name,
                    "params": function_input,
                    "conversation_context": conversation_context
                }
            
            else:
                print(f"Unknown function: {function_name}")
                return None
        
        print(f"MCP max iterations ({max_iterations}) reached without final decision. Falling back to random exploration.")
        
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
            
            return {
                "action_type": "direct_operation",
                "operate": selected_operation,
                "params": {"coordinate": selected_obj['center']},
                "conversation_context": conversation_context,
                "fallback_decision": True,
                "selected_object": selected_obj
            }
        else:
            # Ultimate fallback if no objects detected
            print("No objects detected, using center screen click as last resort")
            return {
                "action_type": "direct_operation",
                "operate": "Click",
                "params": {"coordinate": [640, 360]},
                "conversation_context": conversation_context,
                "fallback_decision": True,
                "fallback_reason": "no_objects_detected"
            }
    
    def _build_mcp_prompt(self, task, conversation_context, iteration_count, max_iterations):
        """Build enhanced MCP interaction prompt with intelligent context analysis"""
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
   - Click: Select/activate UI elements, buttons, or single-target actions
   - Drag: Move cards to targets (enemies/self), drag items between locations
   - RightSingle, LeftDouble: Alternative click interactions
   - Type: Input text when needed
   - Finished: Complete the task

💡 OPERATION GUIDANCE:
- For CARD GAMES: Use Drag to play cards on specific targets (enemies, self)
- For UI ELEMENTS: Use Click for buttons, menus, selections
- PREFER Drag over Click when moving objects to specific destinations
- Drag is more efficient than Click+Click sequences for card targeting

🧠 DECISION STRATEGY:
{context_analysis['decision_guidance']}

⚡ EFFICIENCY RULES:
- AVOID repeating identical queries (especially 'all_objects')
- If you've seen the same objects multiple times, MAKE A DECISION
- Use specific queries (clickable_objects, text_content) over general ones
- With {remaining_iterations} iterations left, prioritize ACTION over exploration
- Don't hesitate to take direct action when you have sufficient information

Analyze the current situation and choose the most appropriate action to complete the task."""
        
        # Add condensed context history if available
        if conversation_context:
            base_prompt += "\n\n📋 CONTEXT SUMMARY:\n"
            base_prompt += context_analysis['context_summary']
        
        return base_prompt
    
    def _analyze_conversation_context(self, conversation_context):
        """Analyze conversation context to provide intelligent guidance"""
        if not conversation_context:
            return {
                'status_summary': '- Fresh start, no previous context',
                'decision_guidance': 'Start with a strategic environment query to understand available options.',
                'context_summary': ''
            }
        
        # Analyze query patterns
        query_types = [ctx['query'].get('query_type', 'unknown') for ctx in conversation_context]
        query_counts = {}
        for qt in query_types:
            query_counts[qt] = query_counts.get(qt, 0) + 1
        
        # Detect repetitive behavior
        repeated_queries = [qt for qt, count in query_counts.items() if count > 2]
        last_3_queries = query_types[-3:] if len(query_types) >= 3 else query_types
        
        # Generate status summary
        status_summary = f"- Query pattern: {dict(query_counts)}"
        if repeated_queries:
            status_summary += f"\n- ⚠️ REPETITIVE: {repeated_queries} (stop repeating!)"
        
        # Generate decision guidance
        decision_guidance = ""
        if 'all_objects' in repeated_queries:
            decision_guidance += "🚨 You've queried 'all_objects' multiple times. You know what's on screen - TAKE ACTION!\n"
        
        if len(set(last_3_queries)) == 1:
            decision_guidance += "🚨 Last 3 queries were identical. Break the loop - try a different approach or make a decision!\n"
        
        if len(conversation_context) > 4:
            decision_guidance += "⏰ You've explored enough. Time to make a decision based on available information.\n"
        
        if not decision_guidance:
            decision_guidance = "Continue strategic exploration, but be ready to act when you have enough information."
        
        # Generate condensed context summary
        context_summary = ""
        unique_results = set()
        for ctx in conversation_context[-5:]:  # Last 5 interactions only
            result_key = ctx['result'][:100]  # First 100 chars as key
            if result_key not in unique_results:
                unique_results.add(result_key)
                context_summary += f"• {ctx['query']} → {ctx['result'][:150]}...\n"
        
        return {
            'status_summary': status_summary,
            'decision_guidance': decision_guidance.strip(),
            'context_summary': context_summary
        }