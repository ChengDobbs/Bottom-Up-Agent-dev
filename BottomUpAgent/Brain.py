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
        self.base_model = creat_base_model(config['brain']['base_model'])
        self.evaluate_model = creat_base_model(config['brain']['evaluate_model'])
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

    def generate_and_save_skill(self, step, obs, operations, state_id, mcst_node_id):
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
            else:
                print("Unknown function: "+response['function']['name'])
                return None
        else:
            print("No function call!")
            return None
    
    def execute_environment_query(self, query_type, detected_objects, object_id=None):
        """Execute environment query and return formatted results"""
        if query_type == "all_objects":
            return self._format_all_objects(detected_objects)
        elif query_type == "clickable_objects":
            return self._format_clickable_objects(detected_objects)
        elif query_type == "text_content":
            return self._format_text_content(detected_objects)
        elif query_type == "specific_object" and object_id:
            return self._format_specific_object(detected_objects, object_id)
        elif query_type == "object_summary":
            return self._format_object_summary(detected_objects)
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
        
        print("MCP max iterations reached without final decision")
        # Fallback to intelligent random exploration from detected objects
        print("Falling back to random exploration from detected objects")
        
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
        """Build MCP interaction prompt with conversation context and dynamic tool discovery"""
        num_queries = len(conversation_context)
        remaining_iterations = max_iterations - iteration_count
        # TODO: cold start every calling, 
        # should reflect&retrospect the histories,
        # should summarize the previous (father node) states.
        base_prompt = f"""You are an intelligent agent playing a game and your task is: '{task}'.

You have access to various tools that you can discover and use as needed. The available tools will be provided to you through the function calling interface.

Approach:
- Explore the available tools and use them strategically to understand the environment
- Learn from past experiences through skill-related tools when applicable
- Take direct actions when you have sufficient information and confidence
- You have {remaining_iterations} iterations remaining to complete this task
- Do not hesitate to make your final desicion before falling back to naive brute-force random search
- Be efficient but thorough in your decision-making process

Analyze the current situation and choose the most appropriate tool to help you progress toward completing the task.
"""
        
        if conversation_context:
            base_prompt += "\n\nPrevious interactions in this session:\n"
            for ctx in conversation_context:
                base_prompt += f"Iteration {ctx['iteration']}: {ctx['query']} -> {ctx['result'][:200]}...\n"
        
        return base_prompt