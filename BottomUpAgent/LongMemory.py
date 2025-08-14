import sqlite3
import json
import pickle
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from .Mcts import MCTS
from .VectorMemory import VectorMemory

class LongMemory:
    def __init__(self, config):
        self.name = config['game_name']
        # 创建标准化的数据库路径
        sql_db_dir = './data/sql'
        os.makedirs(sql_db_dir, exist_ok=True)
        db_path = os.path.join(sql_db_dir, f'{self.name.lower().replace(" ", "_")}.db')
        self.longmemory = sqlite3.connect(db_path)
        self.sim_threshold = config['long_memory']['sim_threshold']
        
        # 初始化向量数据库
        self.vector_memory = VectorMemory(config)

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

        cursor.execute("CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, content TEXT , image BLOB, hash BLOB, area INTEGER, vector_id TEXT, bbox TEXT, center TEXT)")

        cursor.execute("CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, name TEXT, description TEXT, operations TEXT, " \
        "fitness REAL, num INTEGER, state_id INTEGER, mcts_node_id INTEGER, image1 BLOB, image2 BLOB)")

        cursor.execute("CREATE TABLE IF NOT EXISTS skill_clusters (id INTEGER PRIMARY KEY, state_feature BLOB, name TEXT, description TEXT, members TEXT, explore_nums INTEGER)")

        self.longmemory.commit()


    """   Objects   """
        
    def get_object_by_ids(self, ids):
        objects = []
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, content, image, hash, area, vector_id, bbox, center FROM objects WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for record in records:
            id, content, image_blob, hash_blob, area, vector_id, bbox_str, center_str = record
            image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
            hash = pickle.loads(hash_blob)
            
            # 解析JSON字段
            bbox = json.loads(bbox_str) if bbox_str else []
            center = json.loads(center_str) if center_str else [0, 0]
            
            objects.append({
                "id": id, 
                "content": content, 
                "image": image, 
                "hash": hash, 
                "area": area,
                "vector_id": vector_id,
                "bbox": bbox,
                "center": center
            })

        return objects
    
    def update_objects(self, state, objects, text_weight=0.3, image_weight=0.5, bbox_weight=0.2, bbox_scale=1000.0):
        """
        更新对象到长期记忆中
        
        Args:
            state: 当前状态
            objects: 对象列表
            text_weight: 文本相似度权重
            image_weight: 图像相似度权重
            bbox_weight: 位置相似度权重
            bbox_scale: 位置距离缩放因子
        """
        cursor = self.longmemory.cursor()
        updated_objects_nums = 0
        
        # 使用向量数据库进行智能去重和存储（使用mixed方式）
        processed_objects = self.vector_memory.update_object_with_vector_storage(
            objects, text_weight=text_weight, image_weight=image_weight, 
            bbox_weight=bbox_weight, similarity_threshold=0.85
        )
        
        for obj in processed_objects:
            if obj['id'] is None:
                # New object - 存储到SQLite
                _, image_blob = cv2.imencode('.png', obj['image'])
                image_blob = image_blob.tobytes()
                hash_blob = pickle.dumps(obj['hash'])
                
                cursor.execute(
                    "INSERT INTO objects (content, image, hash, area, vector_id, bbox, center) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                    (
                        obj['content'], 
                        image_blob, 
                        hash_blob, 
                        obj['area'],
                        obj.get('vector_id', ''),
                        json.dumps(obj['bbox']),
                        json.dumps(obj['center'])
                    )
                )
                obj['id'] = cursor.lastrowid
                state['object_ids'].append(obj['id'])
                updated_objects_nums += 1
                
                print(f"新对象存储: ID={obj['id']}, Content='{obj['content'][:20]}...', Vector_ID={obj.get('vector_id', 'N/A')}")
            else:
                # 已存在的对象，可能需要更新向量ID
                if 'vector_id' in obj:
                    cursor.execute(
                        "UPDATE objects SET vector_id = ? WHERE id = ?", 
                        (obj['vector_id'], obj['id'])
                    )
        
        cursor.execute("UPDATE states SET object_ids = ? WHERE id = ?", (json.dumps(state['object_ids']), state['id']))
        print(f"Updated objects nums: {updated_objects_nums}")
        print(f"Vector database stats: {self.vector_memory.get_collection_stats()}")
        self.longmemory.commit()
        return processed_objects

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

    """   Vector Memory Integration   """
    
    def search_similar_objects(self, query_obj: dict, threshold: float = 0.8, top_k: int = 5):
        """
        使用向量数据库搜索相似对象
        
        Args:
            query_obj: 查询对象，包含image字段
            threshold: 相似度阈值
            top_k: 返回的最大结果数
            
        Returns:
            相似对象列表
        """
        return self.vector_memory.find_similar_objects(query_obj, threshold, top_k)
    
    def search_objects_by_content(self, content_query: str, top_k: int = 10):
        """
        根据内容搜索对象
        
        Args:
            content_query: 内容查询字符串
            top_k: 返回的最大结果数
            
        Returns:
            匹配的对象列表
        """
        return self.vector_memory.search_objects_by_content(content_query, top_k)
    
    def get_vector_db_stats(self):
        """获取向量数据库统计信息"""
        return self.vector_memory.get_collection_stats()
    
    def clear_vector_database(self):
        """清空向量数据库"""
        self.vector_memory.clear_database()
    
    def find_objects_in_region(self, bbox_region: list, expand_ratio: float = 0.1):
        """
        查找指定区域内的对象
        
        Args:
            bbox_region: [x, y, w, h] 区域坐标
            expand_ratio: 区域扩展比例
            
        Returns:
            区域内的对象列表
        """
        cursor = self.longmemory.cursor()
        cursor.execute('SELECT id, content, bbox, center, vector_id FROM objects WHERE bbox IS NOT NULL')
        records = cursor.fetchall()
        
        region_objects = []
        x_min, y_min, w, h = bbox_region
        x_max, y_max = x_min + w, y_min + h
        
        # 扩展搜索区域
        expand_w, expand_h = w * expand_ratio, h * expand_ratio
        search_x_min = max(0, x_min - expand_w)
        search_y_min = max(0, y_min - expand_h)
        search_x_max = x_max + expand_w
        search_y_max = y_max + expand_h
        
        for record in records:
            obj_id, content, bbox_str, center_str, vector_id = record
            if bbox_str:
                obj_bbox = json.loads(bbox_str)
                obj_x, obj_y, obj_w, obj_h = obj_bbox
                obj_center_x = obj_x + obj_w / 2
                obj_center_y = obj_y + obj_h / 2
                
                # 检查对象是否在搜索区域内
                if (search_x_min <= obj_center_x <= search_x_max and 
                    search_y_min <= obj_center_y <= search_y_max):
                    region_objects.append({
                        'id': obj_id,
                        'content': content,
                        'bbox': obj_bbox,
                        'center': json.loads(center_str) if center_str else [obj_center_x, obj_center_y],
                        'vector_id': vector_id
                    })
        
        return region_objects
    
    def get_object_interaction_history(self, object_id: int, limit: int = 10):
        """
        获取对象的交互历史（基于技能记录）
        
        Args:
            object_id: 对象ID
            limit: 返回的最大记录数
            
        Returns:
            交互历史列表
        """
        cursor = self.longmemory.cursor()
        
        # 查找涉及该对象的技能
        cursor.execute('''
            SELECT s.id, s.name, s.description, s.operations, s.fitness, s.num
            FROM skills s
            JOIN states st ON s.state_id = st.id
            WHERE st.object_ids LIKE ?
            ORDER BY s.id DESC
            LIMIT ?
        ''', (f'%{object_id}%', limit))
        
        records = cursor.fetchall()
        interactions = []
        
        for record in records:
            skill_id, name, description, operations_str, fitness, num = record
            interactions.append({
                'skill_id': skill_id,
                'name': name,
                'description': description,
                'operations': json.loads(operations_str),
                'fitness': fitness,
                'usage_count': num
            })
        
        return interactions
        




    

