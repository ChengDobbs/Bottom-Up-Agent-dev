import sqlite3

from base_model import creat_base_model
from . import Prompt
from . import FunctionCalls
from utils.utils import cv_to_base64
from .LongMemory import LongMemory
import numpy as np
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

        self.uct_c = config['brain']['uct_c']
        self.uct_threshold = config['brain']['uct_threshold']


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
                try:
                    from .visualizer import push_data
                    push_data({'candidate_skills': skills_info, 'selected_skill_id': 'Explore'})
                except ImportError:
                    pass  # Visualizer not available
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
            if operation['operate'] == 'Click':
                cords = f"({operation['params']['x']}, {operation['params']['y']})"
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
                # New skills start with default skill_type=0 (PROGRESS_CONTRIBUTING)
                id = self.long_memory.save_skill(response['function']['input']['name'], response['function']['input']['description'], 
                                            operations_to_save, 0, 1, state_id, mcst_node_id, obs[0]['screen'], obs[-1]['screen'], skill_type=0)
                print(f"save skill: {response['function']['input']['name']}, operations: {operations_str}, fitness: {0}, skill_type: PROGRESS_CONTRIBUTING")
                return {"id": id, "name": response['function']['input']['name'], "description": response['function']['input']['description'], "skill_type": 0}
            elif response['function']['name'] == "incomplete_skill":
                print(f"incomplete skill detected: {response['function']['input']['name']}")
                print(f"description: {response['function']['input']['description']}")
                
                # Extract structured hint information
                hint_data = response['function']['input']['next_action_hint']
                print(f"[HINT] next action hint: {hint_data}")
                
                # Store hint information in agent's context manager if agent is provided
                if agent is not None:
                    agent.store_hint(hint_data)
                
                # Save as temporary incomplete skill with special fitness marker
                id = self.long_memory.save_skill(
                    response['function']['input']['name'], 
                    response['function']['input']['description'], 
                    operations_to_save, -1, 0, state_id, mcst_node_id, 
                    obs[0]['screen'], obs[-1]['screen'], skill_type=1
                )
                
                return {
                    "id": id, 
                    "name": response['function']['input']['name'], 
                    "description": response['function']['input']['description'],
                    "skill_type": 1,  # NAVIGATION_ONLY for incomplete skills
                    "incomplete": True,
                    "next_action_hint": hint_data
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