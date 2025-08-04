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

        return similarity < threshold
    
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
            if operation['operate'] == 'Click':
                cords = f"({operation['params']['x']}, {operation['params']['y']})"
                operations_str += 'operation' + str(idx) + ': ' + operation['operate'] + ' ' + cords + '\n'
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