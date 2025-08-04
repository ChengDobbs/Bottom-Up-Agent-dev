import sqlite3
import json
import pickle
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from .Mcts import MCTS

class LongMemory:
    def __init__(self, config):
        self.name = config['game_name']
        self.longmemory = sqlite3.connect(self.name+'.db')
        self.sim_threshold = config['long_memory']['sim_threshold']

        if not self.is_initialized():
            self.initialize()

        # self.objects = self.get_objects() 

    def is_initialized(self):
        # detect database whether has the table named init
        cursor = self.longmemory.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

        if cursor.fetchone():
            return True
        else:
            return False
    
    def initialize(self): 
        print("Initializing LongMomery")
        #create table
        cursor = self.longmemory.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY, state_feature BLOB, mcts TEXT, object_ids TEXT, skill_clusters TEXT)")

        cursor.execute("CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, name TEXT , image BLOB, hash BLOB, area INTEGER)")

        cursor.execute("CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, name TEXT, description TEXT, operations TEXT, " \
        "fitness REAL, num INTEGER, state_id INTEGER, mcts_node_id INTEGER, image1 BLOB, image2 BLOB)")

        cursor.execute("CREATE TABLE IF NOT EXISTS skill_clusters (id INTEGER PRIMARY KEY, state_feature BLOB, name TEXT, description TEXT, members TEXT, explore_nums INTEGER)")

        self.longmemory.commit()


    """   Objects   """
        
    def get_object_by_ids(self, ids):
        objects = []
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, name, image, hash, area FROM objects WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for id, name, image_blob, hash_blob, area in records:
            image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
            hash = pickle.loads(hash_blob)
            objects.append({"id": id, "name": name, "image": image, "hash": hash, "area": area})

        return objects
    
    def update_objects(self, state, objects):
        cursor = self.longmemory.cursor()
        updated_objects_nums = 0
        for obj in objects:
            if obj['id'] is None:
                # New object
                _, image_blob = cv2.imencode('.png', obj['image'])
                image_blob = image_blob.tobytes()
                hash_blob = pickle.dumps(obj['hash'])
                cursor.execute("INSERT INTO objects (image, hash, area) VALUES (?, ?, ?)", (image_blob, hash_blob, obj['area']))
                obj['id'] = cursor.lastrowid
                state['object_ids'].append(obj['id'])
                updated_objects_nums += 1
        
        cursor.execute("UPDATE states SET object_ids = ? WHERE id = ?", (json.dumps(state['object_ids']), state['id']))
        print(f"Updated objects nums: {updated_objects_nums}")
        self.longmemory.commit()
        return objects

    def get_object_image_by_id(self, id):
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT image FROM objects WHERE id = ?', (id,))
        record = cursor.fetchone()

        if record is None:
            return None

        image_blob = record[0]
        image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
        return image
    
    """   States   """
    def save_state(self, state):
        state_feature_blob = pickle.dumps(state['state_feature'])
        mcts_str = json.dumps(state['mcts'].to_dict())
        objects_ids_str = json.dumps(state['object_ids'])
        skill_clusters_str = json.dumps(state['skill_clusters'])

        cursor = self.longmemory.cursor()
        cursor.execute("INSERT INTO states (state_feature, mcts, object_ids, skill_clusters) VALUES (?, ?, ?, ?)", 
                       (state_feature_blob, mcts_str, objects_ids_str, skill_clusters_str))
        self.longmemory.commit()

        state_id = cursor.lastrowid
        return state_id

    def get_state(self, ob, sim_threshold=0.85):
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, state_feature, mcts, object_ids, skill_clusters FROM states')
        records = cursor.fetchall()

        max_sim = -1
        best_idx = None

        state_feature = ob['state_feature'].reshape(1, -1)    
        for idx in range(len(records)):
            feat_blob = records[idx][1]
            feat = pickle.loads(feat_blob).reshape(1, -1)
            sim = cosine_similarity(state_feature, feat)[0][0]
            if sim > max_sim:
                max_sim = sim
                best_idx = idx

        if max_sim > sim_threshold:
            record = records[best_idx]
            state = {
                "id": record[0],
                "state_feature": pickle.loads(record[1]),
                "mcts": MCTS.from_dict(json.loads(record[2])),
                "object_ids": json.loads(record[3]),
                "skill_clusters": json.loads(record[4])
            }
            return state
        else:
            return None
            
    def update_state(self, state):
        mcts_str = json.dumps(state['mcts'].to_dict())
        objects_ids_str = json.dumps(state['object_ids'])
        skill_clusters_str = json.dumps(state['skill_clusters'])

        cursor = self.longmemory.cursor()
        cursor.execute("UPDATE states SET mcts = ?, object_ids = ?, skill_clusters = ? WHERE id = ?", 
                       (mcts_str, objects_ids_str, skill_clusters_str, state['id']))
        self.longmemory.commit()

    """   skill clusters   """
    
    def get_skill_clusters_by_id(self, id):
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, name, description, members, explore_nums FROM skill_clusters WHERE id = ?', (id,))
        record = cursor.fetchone()

        if record is None:
            return None

        skill_cluster = {
            "id": record[0],
            "name": record[1],
            "description": record[2],
            "members": json.loads(record[3]),
            "explore_nums": record[4]
        }
        return skill_cluster
        
    def save_skill_cluster(self, state_feature, name, description, members, explore_nums=1):
        state_feature_blob = pickle.dumps(state_feature)

        cursor = self.longmemory.cursor()
        cursor.execute("INSERT INTO skill_clusters(state_feature, name, description, members, explore_nums) VALUES (?, ?, ?, ?, ?)", \
                       (state_feature_blob, name, description, json.dumps(members), explore_nums))
        
        skill_cluster_id = cursor.lastrowid
        self.longmemory.commit()

        return skill_cluster_id
    
    def get_skill_clusters_by_ids(self, ids):
        skill_clusters = []
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, name, description, members, explore_nums FROM skill_clusters WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for record in records:
            skill_cluster = {"id": record[0], "name": record[1], "description": record[2], "members": json.loads(record[3]), "explore_nums": record[4]}
            skill_clusters.append(skill_cluster)

        return skill_clusters
    
    def update_skill_cluster(self, id, state_feature, name, description, members):

        cursor = self.longmemory.cursor()
        cursor.execute("UPDATE skill_clusters SET name = ?, description = ?, members = ? WHERE id = ?", \
                       (name, description, json.dumps(members), id))
        self.longmemory.commit()

    def update_skill_cluster_explore_nums(self, id, explore_nums):
        cursor = self.longmemory.cursor()
        cursor.execute("UPDATE skill_clusters SET explore_nums = ? WHERE id = ?", (explore_nums, id))
        self.longmemory.commit()


    """   skills   """
    
    def save_skill(self, name, description, operations, fitness, num, state_id, mcts_node_id, image1, image2):
        _, image1_blob = cv2.imencode('.png', image1)
        image1_blob = image1_blob.tobytes()
        _, image2_blob = cv2.imencode('.png', image2)
        image2_blob = image2_blob.tobytes()

        cursor = self.longmemory.cursor()
        cursor.execute("INSERT INTO skills (name, description, operations, fitness, num, state_id, mcts_node_id, image1, image2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", \
                       (name, description, json.dumps(operations), fitness, num, state_id, mcts_node_id, image1_blob, image2_blob))
        self.longmemory.commit()

        skill_id = cursor.lastrowid
        return skill_id

    def update_skill(self, id, fitness, num):
        cursor = self.longmemory.cursor()
        cursor.execute("UPDATE skills SET fitness = ?, num = ? WHERE id = ?", (fitness, num, id))
        self.longmemory.commit()

    
    def get_skills_by_ids(self, ids):
        skills = []
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, name, description, operations, fitness, num, state_id, mcts_node_id FROM skills WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for record in records:
            skill = {
                "id": record[0],
                "name": record[1],
                "description": record[2],
                "operations": json.loads(record[3]),
                "fitness": record[4],
                "num": record[5],
                "state_id": record[6],
                "mcts_node_id": record[7]
            }
            skills.append(skill)

        return skills

    def delete_skill(self, skill, skill_cluster):
        cursor = self.longmemory.cursor()
        
        # delete skill from skill cluster
        if skill['id'] in skill_cluster['members']:
            skill_cluster['members'].remove(skill['id'])
            if len(skill_cluster['members']) == 0:
                cursor.execute("DELETE FROM skill_clusters WHERE id = ?", (skill_cluster['id'],))
            else:
                cursor.execute("UPDATE skill_clusters SET members = ? WHERE id = ?", (json.dumps(skill_cluster['members']), skill_cluster['id']))

        # delete skill from skills
        id = skill['id']
        cursor.execute("DELETE FROM skills WHERE id = ?", (id,))
        self.longmemory.commit()

    def get_skills(self):
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, name, description, operations, fitness, num, state_id, mcts_node_id FROM skills')
        records = cursor.fetchall()

        skills = []
        for record in records:
            skill = {
                "id": record[0],
                "name": record[1],
                "description": record[2],
                "operations": json.loads(record[3]),
                "fitness": record[4],
                "num": record[5],
                "state_id": record[6],
                "mcts_node_id": record[7]
            }
            skills.append(skill)

        return skills
        




    

