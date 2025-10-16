import sqlite3
import json
import pickle
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import threading
from .Mcts import MCTS
from .VectorMemory import VectorMemory

class LongMemory:
    def __init__(self, config):
        self.name = config['game_name']
        sql_db_dir = './data/sql'
        os.makedirs(sql_db_dir, exist_ok=True)
        self.db_path = os.path.join(sql_db_dir, f'{self.name.lower().replace(" ", "_")}.db')
        self.sim_threshold = config['long_memory']['sim_threshold']
        
        self.vector_memory = VectorMemory(config)
        
        self._local = threading.local()

        if not self.is_initialized():
            self.initialize()

        # self.objects = self.get_objects() 

    def get_connection(self):
        """Get thread-safe database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.connection

    def is_initialized(self):
        # detect database whether has the table named init
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

        if cursor.fetchone():
            return True
        else:
            return False
    
    def initialize(self): 
        print("Initializing LongMomery")
        #create table
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY, state_feature BLOB, mcts TEXT, object_ids TEXT, skill_clusters TEXT)")

        cursor.execute("CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, content TEXT , image BLOB, hash BLOB, area INTEGER, bbox TEXT, center TEXT, history_centers TEXT, vector_id TEXT, linked_skill_id INTEGER, is_click_change INTEGER DEFAULT 0, is_hover_change INTEGER DEFAULT 0, hover_tooltip TEXT)")

        cursor.execute("CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, name TEXT, description TEXT, operations TEXT, skill_type INTEGER DEFAULT 0, " \
        "fitness REAL, num INTEGER, state_id INTEGER, mcts_node_id INTEGER, image1 BLOB, image2 BLOB)")

        cursor.execute("CREATE TABLE IF NOT EXISTS skill_clusters (id INTEGER PRIMARY KEY, state_feature BLOB, name TEXT, description TEXT, members TEXT, explore_nums INTEGER)")

        conn.commit()

    def _normalize_image_color_format(self, image, for_storage=True):
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            # Not a color image, return as-is
            return image
        
        if for_storage:
            # Convert RGB to BGR for OpenCV storage consistency
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # Convert BGR to RGB for application usage consistency
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _encode_image_blob(self, image):
        normalized_image = self._normalize_image_color_format(image, for_storage=True)
        
        # Encode to PNG format
        success, image_blob = cv2.imencode('.png', normalized_image)
        if not success:
            raise ValueError("Failed to encode image to PNG format")
        
        return image_blob.tobytes()

    def _decode_image_blob(self, image_blob):
        image_bgr = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode image blob")
        
        # Normalize color format for application usage
        return self._normalize_image_color_format(image_bgr, for_storage=False)

    """   Objects   """
    def get_object_by_ids(self, ids):
        objects = []
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, image, hash, area, bbox, center, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip FROM objects WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for record in records:
            id, content, image_blob, hash_blob, area, bbox_str, center_str, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip = record
            image = self._decode_image_blob(image_blob)
            hash = pickle.loads(hash_blob)
            
            bbox = json.loads(bbox_str)
            center = json.loads(center_str)
            history_centers_list = json.loads(history_centers) if history_centers else []
            
            objects.append({
                "id": id, 
                "content": content, 
                "image": image, 
                "hash": hash, 
                "area": area,
                "vector_id": vector_id,
                "bbox": bbox,
                "center": center,
                "history_centers": history_centers_list,
                "linked_skill_id": linked_skill_id,
                "is_click_change": bool(is_click_change) if is_click_change is not None else False,
                "is_hover_change": bool(is_hover_change) if is_hover_change is not None else False,
                "hover_tooltip": hover_tooltip
            })

        return objects

    def get_object_by_vector_id(self, vector_id):
        """Get object by vector_id from the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, image, hash, area, bbox, center, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip FROM objects WHERE vector_id = ?', (vector_id,))
        record = cursor.fetchone()
        
        if record is None:
            return None
            
        id, content, image_blob, hash_blob, area, bbox_str, center_str, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip = record
        image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
        hash = pickle.loads(hash_blob)
        
        bbox = json.loads(bbox_str)
        center = json.loads(center_str)
        history_centers = json.loads(history_centers) if history_centers else []
        
        return {
            "id": id, 
            "content": content, 
            "image": image, 
            "hash": hash, 
            "area": area,
            "vector_id": vector_id,
            "bbox": bbox,
            "center": center,
            "history_centers": history_centers,
            "is_click_change": bool(is_click_change) if is_click_change is not None else False,
            "linked_skill_id": linked_skill_id,
            "is_hover_change": bool(is_hover_change) if is_hover_change is not None else False,
            "hover_tooltip": hover_tooltip
        }

    def get_recent_objects(self, limit=50):
        """Get recent objects from the database for MCP context"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, image, hash, area, bbox, center, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip
            FROM objects 
            ORDER BY id DESC 
            LIMIT ?
        ''', (limit,))
        records = cursor.fetchall()
        
        objects = []
        for record in records:
            id, content, image_blob, hash_blob, area, bbox_str, center_str, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip = record
            image = self._decode_image_blob(image_blob)
            hash = pickle.loads(hash_blob)
            
            bbox = json.loads(bbox_str)
            center = json.loads(center_str)
            history_centers_list = json.loads(history_centers) if history_centers else []
            
            objects.append({
                "id": id, 
                "content": content, 
                "image": image, 
                "hash": hash, 
                "area": area,
                "vector_id": vector_id,
                "bbox": bbox,
                "center": center,
                "is_click_change": bool(is_click_change) if is_click_change is not None else False,
                "linked_skill_id": linked_skill_id,
                "history_centers": history_centers_list,
                "is_hover_change": bool(is_hover_change) if is_hover_change is not None else False,
                "hover_tooltip": hover_tooltip
            })
        
        return objects

    def update_objects(self, state, objects, text_weight=0.3, image_weight=0.5, bbox_weight=0.2, bbox_scale=1000.0):
        conn = self.get_connection()
        cursor = conn.cursor()
        updated_objects_nums = 0
        
        # Use VecDB to deduplicate and store objs (mixed)
        processed_objects = self.vector_memory.update_object_with_vector_storage(
            objects, text_weight=text_weight, image_weight=image_weight, 
            bbox_weight=bbox_weight, similarity_threshold=0.85
        )
        
        # Fix objects with None ID by querying SQL database using vector_id
        for obj in processed_objects:
            if obj['id'] is None and obj.get('vector_id'):
                sql_obj = self.get_object_by_vector_id(obj['vector_id'])
                if sql_obj:
                    obj['id'] = sql_obj['id']
                    print(f"🔧 FIXED MISSING ID: Vector_ID={obj['vector_id']} -> SQL ID={obj['id']}")
                else:
                    print(f"⚠️ VECTOR_ID NOT FOUND IN SQL: {obj['vector_id']} for content '{obj['content'][:20]}...'")
        
        # Collect vector_id and sql_id pairs for batch metadata update
        vector_sql_id_pairs = []
        
        for obj in processed_objects:
            if obj['id'] is None:
                # FIRST TIME SAVE: New object detected, initialize with current position
                image_blob = self._encode_image_blob(obj['image'])
                hash_blob = pickle.dumps(obj['hash'])
                
                # Initialize history_centers with current center position
                initial_history = [obj['center']]
                
                cursor.execute(
                    "INSERT INTO objects (content, image, hash, area, bbox, center, history_centers, vector_id, linked_skill_id, is_click_change, is_hover_change, hover_tooltip) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (
                        obj['content'], 
                        image_blob, 
                        hash_blob, 
                        obj['area'],
                        json.dumps(obj['bbox']),
                        json.dumps(obj['center']),
                        json.dumps(initial_history),  # history_centers initialized with current position
                        obj.get('vector_id', ''),
                        None,  # linked_skill_id default
                        0,  # is_click_change default
                        0,  # is_hover_change default
                        None  # hover_tooltip default
                    )
                )
                obj['id'] = cursor.lastrowid
                state['object_ids'].append(obj['id'])
                updated_objects_nums += 1
                
                # Collect vector_id and sql_id for batch metadata update
                if obj.get('vector_id'):
                    vector_sql_id_pairs.append((obj['vector_id'], obj['id']))
                print(f"Stored new object: ID={obj['id']}, Content='{obj['content'][:20]}...', Vector_ID={obj.get('vector_id', 'N/A')}")
            else:
                # UPDATE EXISTING OBJECT: Object already exists in database
                if 'vector_id' in obj and obj['id'] is not None:
                    # Update all fields for existing objects to reflect current state
                    image_blob = self._encode_image_blob(obj['image'])
                    hash_blob = pickle.dumps(obj['hash'])
                    
                    # POSITION HISTORY UPDATE: Get existing history and append current position
                    cursor.execute("SELECT history_centers FROM objects WHERE id = ?", (obj['id'],))
                    existing_history = cursor.fetchone()
                    if existing_history and existing_history[0]:
                        history_centers = json.loads(existing_history[0])
                    else:
                        # Fallback: if no history exists, initialize with empty list
                        history_centers = []
                        print(f"📍 POSITION HISTORY MISSING: Object {obj['id']}")
                    
                    # Add current position with smart deduplication
                    current_pos = list(obj['center'])  # Normalize to list format
                    
                    # Check if this position is different from the last recorded position
                    # This handles both duplicates and ABA scenarios correctly
                    should_add = True
                    if history_centers:
                        last_pos = list(history_centers[-1]) if history_centers[-1] else []
                        if last_pos == current_pos:
                            should_add = False  # Same as last position, skip
                    
                    if should_add:
                        history_centers.append(current_pos)
                        print(f"📍 POSITION UPDATED: Object {obj['id']} moved to {current_pos}")
                    # else: position unchanged, no log needed
                    
                    cursor.execute(
                        "UPDATE objects SET content = ?, image = ?, hash = ?, area = ?, bbox = ?, center = ?, history_centers = ?, vector_id = ? WHERE id = ?", 
                        (
                            obj['content'],
                            image_blob,
                            hash_blob,
                            obj['area'],
                            json.dumps(obj['bbox']),
                            json.dumps(obj['center']),
                            json.dumps(history_centers),
                            obj['vector_id'],
                            obj['id']
                        )
                    )
                    print(f"Updated existing object: ID={obj['id']}, Content='{obj['content'][:20]}...', Vector_ID={obj.get('vector_id', 'N/A')}")
                else:
                    # ORPHANED OBJECT: Has vector_id but no SQL ID - this shouldn't happen
                    print(f"🚨 ORPHANED OBJECT: Vector_ID={obj.get('vector_id', 'N/A')} but SQL ID=None for '{obj['content'][:20]}...'")
                    print(f"    This object will be skipped in this update cycle.")
        
        # Batch update VectorDB metadata with SQL IDs
        if vector_sql_id_pairs:
            success = self.vector_memory.batch_update_object_metadata(vector_sql_id_pairs)
            if success:
                print(f"🔄 BATCH UPDATED VECTOR METADATA: {len(vector_sql_id_pairs)} objects")
            else:
                print(f"⚠️ BATCH METADATA UPDATE FAILED for {len(vector_sql_id_pairs)} objects")
        
        cursor.execute("UPDATE states SET object_ids = ? WHERE id = ?", (json.dumps(state['object_ids']), state['id']))
        print(f"Updated VecDB objects nums: {updated_objects_nums}")
        print(f"Vector database stats: {self.vector_memory.get_collection_stats()}")
        conn.commit()
        return processed_objects

    def get_object_image_by_id(self, id):
        conn = self.get_connection()
        cursor = conn.cursor()
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

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO states (state_feature, mcts, object_ids, skill_clusters) VALUES (?, ?, ?, ?)", 
                       (state_feature_blob, mcts_str, objects_ids_str, skill_clusters_str))
        conn.commit()

        state_id = cursor.lastrowid
        return state_id

    def get_state(self, ob, sim_threshold=0.85):
        conn = self.get_connection()
        cursor = conn.cursor()
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

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE states SET mcts = ?, object_ids = ?, skill_clusters = ? WHERE id = ?", 
                       (mcts_str, objects_ids_str, skill_clusters_str, state['id']))
        conn.commit()

    """   skill clusters   """
    
    def get_skill_clusters_by_id(self, id):
        conn = self.get_connection()
        cursor = conn.cursor()
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

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO skill_clusters(state_feature, name, description, members, explore_nums) VALUES (?, ?, ?, ?, ?)", \
                       (state_feature_blob, name, description, json.dumps(members), explore_nums))
        
        skill_cluster_id = cursor.lastrowid
        conn.commit()

        return skill_cluster_id
    
    def get_skill_clusters_by_ids(self, ids):
        skill_clusters = []
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, members, explore_nums FROM skill_clusters WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for record in records:
            skill_cluster = {"id": record[0], "name": record[1], "description": record[2], "members": json.loads(record[3]), "explore_nums": record[4]}
            skill_clusters.append(skill_cluster)

        return skill_clusters
    
    def update_skill_cluster(self, id, state_feature, name, description, members):

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE skill_clusters SET name = ?, description = ?, members = ? WHERE id = ?", \
                       (name, description, json.dumps(members), id))
        conn.commit()

    def update_skill_cluster_explore_nums(self, id, explore_nums):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE skill_clusters SET explore_nums = ? WHERE id = ?", (explore_nums, id))
        conn.commit()


    """   skills   """
    
    def save_skill(self, name, description, operations, fitness, num, state_id, mcts_node_id, image1, image2, skill_type=0):
        _, image1_blob = cv2.imencode('.png', image1)
        image1_blob = image1_blob.tobytes()
        _, image2_blob = cv2.imencode('.png', image2)
        image2_blob = image2_blob.tobytes()

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO skills (name, description, operations, skill_type, fitness, num, state_id, mcts_node_id, image1, image2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", \
                       (name, description, json.dumps(operations), skill_type, fitness, num, state_id, mcts_node_id, image1_blob, image2_blob))
        conn.commit()

        skill_id = cursor.lastrowid
        
        # Update linked_skill_id for objects involved in this skill
        self._link_objects_to_skill(operations, skill_id)
        
        return skill_id

    def _link_objects_to_skill(self, operations, skill_id):
        """Link objects involved in operations to the skill"""
        object_ids = set()
        
        # Extract object IDs from operations
        for operation in operations:
            if isinstance(operation, dict) and 'object_id' in operation:
                object_ids.add(operation['object_id'])
        
        if not object_ids:
            print(f"No objects involved in operations for skill {skill_id}")
            return
        
        # Update linked_skill_id for all involved objects in SQL database
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for object_id in object_ids:
            cursor.execute("UPDATE objects SET linked_skill_id = ? WHERE id = ?", (skill_id, object_id))
        
        conn.commit()
        print(f"[OBJ LIBRARY] LINKED {len(object_ids)} objects to skill {skill_id}")

    def update_skill(self, id, fitness, num):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE skills SET fitness = ?, num = ? WHERE id = ?", (fitness, num, id))
        conn.commit()
    
    def update_skill_type(self, skill_id, skill_type):
        """Update skill type for a specific skill"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE skills SET skill_type = ? WHERE id = ?", (skill_type, skill_id))
        type_str = 'PROGRESS_CONTRIBUTING' if skill_type == 0 else 'NAVIGATION_ONLY' if skill_type == 1 else 'INCOMPLETE'
        conn.commit()
        print(f"Updated skill {skill_id} type to: [{type_str}]")
    
    def update_skill_with_type(self, skill_id, fitness, num, skill_type=None):
        """Update skill fitness, num, and optionally skill_type"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if skill_type is not None:
            cursor.execute("UPDATE skills SET fitness = ?, num = ?, skill_type = ? WHERE id = ?", 
                          (fitness, num, skill_type, skill_id))
            print(f"Updated skill {skill_id}: fitness={fitness}, num={num}, skill_type={skill_type}")
        else:
            cursor.execute("UPDATE skills SET fitness = ?, num = ? WHERE id = ?", 
                          (fitness, num, skill_id))
            print(f"Updated skill {skill_id}: fitness={fitness}, num={num}")
        
        conn.commit()

    
    def get_skills_by_ids(self, ids):
        skills = []
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, operations, skill_type , fitness, num, state_id, mcts_node_id FROM skills WHERE id IN ({})'.format(','.join('?'*len(ids))), ids)
        records = cursor.fetchall()

        for record in records:
            skill = {
                "id": record[0],
                "name": record[1],
                "description": record[2],
                "operations": json.loads(record[3]),
                "skill_type": record[4],
                "fitness": record[5],
                "num": record[6],
                "state_id": record[7],
                "mcts_node_id": record[8]
            }
            skills.append(skill)

        return skills

    def delete_skill(self, skill, skill_cluster):
        conn = self.get_connection()
        cursor = conn.cursor()
        
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
        conn.commit()

    def get_skills(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, operations, skill_type, fitness, num, state_id, mcts_node_id FROM skills')
        records = cursor.fetchall()

        skills = []
        for record in records:
            skill = {
                "id": record[0],
                "name": record[1],
                "description": record[2],
                "operations": json.loads(record[3]),
                "skill_type": record[4],
                "fitness": record[5],
                "num": record[6],
                "state_id": record[7],
                "mcts_node_id": record[8]
            }
            skills.append(skill)

        return skills

    """   Vector Memory Integration   """
    
    def search_similar_objects(self, query_obj: dict, threshold: float = 0.8, top_k: int = 5):
        """
        Search for similar objects using vector database
        
        Args:
            query_obj: Query object containing image field
            threshold: Similarity threshold
            top_k: Maximum number of results to return
            
        Returns:
            List of similar objects
        """
        return self.vector_memory.find_similar_objects(query_obj, threshold, top_k)
    
    def search_objects_by_content(self, content_query: str, top_k: int = 10):
        """
        Search objects by content
        
        Args:
            content_query: Content query string
            top_k: Maximum number of results to return
            
        Returns:
            List of matching objects
        """
        return self.vector_memory.search_objects_by_content(content_query, top_k)
    
    def get_vector_db_stats(self):
        """Get vector database statistics"""
        return self.vector_memory.get_collection_stats()
    
    def clear_vector_database(self):
        """Clear vector database"""
        self.vector_memory.clear_database()
    
    def find_objects_in_region(self, bbox_region: list, expand_ratio: float = 0.1):
        """
        Find objects within specified region
        
        Args:
            bbox_region: [x, y, w, h] region coordinates
            expand_ratio: Region expansion ratio
            
        Returns:
            List of objects within the region
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, bbox, center, vector_id FROM objects WHERE bbox IS NOT NULL')
        records = cursor.fetchall()
        
        region_objects = []
        x_min, y_min, w, h = bbox_region
        x_max, y_max = x_min + w, y_min + h
        
        # Expand search region
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
                
                # Check if object is within search region
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
        Get object interaction history (based on skill records)
        
        Args:
            object_id: Object ID
            limit: Maximum number of records to return
            
        Returns:
            List of interaction history
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find skills involving this object
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

    # TODO: below are reserved for future use
    
    def update_object_clickability_by_id(self, object_id: int, is_click_change: bool):
        """Update object clickability status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE objects SET is_click_change = ? WHERE id = ?", (int(is_click_change), object_id))
        conn.commit()
        print(f"Updated object {object_id} clickability to: {is_click_change}")
    
    def update_object_hover_info(self, object_id: int, is_hover_change: bool, hover_tooltip: str = None):
        """Update object hover information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE objects SET is_hover_change = ?, hover_tooltip = ? WHERE id = ?", 
                      (int(is_hover_change), hover_tooltip, object_id))
        conn.commit()
        print(f"Updated object {object_id} hover info - hoverable: {is_hover_change}, tooltip: {hover_tooltip}")
    
    def link_object_to_skill(self, object_id: int, skill_id: int):
        """Link object to a related skill"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE objects SET related_skill_id = ? WHERE id = ?", (skill_id, object_id))
        conn.commit()
        print(f"[MEMORY] Linked object {object_id} to skill {skill_id}")
    
    def get_objects_by_clickability(self, is_click_change: bool = True):
        """Get objects by clickability status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, bbox, center FROM objects WHERE is_click_change = ?', (int(is_click_change),))
        records = cursor.fetchall()
        
        objects = []
        for record in records:
            id, content, bbox_str, center_str = record
            bbox = json.loads(bbox_str)
            center = json.loads(center_str)
            objects.append({
                "id": id,
                "content": content,
                "bbox": bbox,
                "center": center
            })
        
        return objects
    
    def get_objects_with_hover_info(self):
        """Get objects that have hover information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, hover_tooltip, bbox, center FROM objects WHERE is_hover_change = 1 AND hover_tooltip IS NOT NULL')
        records = cursor.fetchall()
        
        objects = []
        for record in records:
            id, content, hover_tooltip, bbox_str, center_str = record
            bbox = json.loads(bbox_str)
            center = json.loads(center_str)
            objects.append({
                "id": id,
                "content": content,
                "hover_tooltip": hover_tooltip,
                "bbox": bbox,
                "center": center
            })
        
        return objects
    
    def get_object_hover_status(self, object_id: int):
        """Get hover status of a specific object"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT is_hover_change, hover_tooltip FROM objects WHERE id = ?', (object_id,))
        record = cursor.fetchone()
        
        if record is None:
            return None, None
        
        is_hover_change, hover_tooltip = record
        return bool(is_hover_change) if is_hover_change is not None else False, hover_tooltip
    
    def get_objects_needing_hover_test(self, object_ids: list):
        """Get objects that need hover testing (new objects or objects with is_hover_change=0 and hover_tooltip=None)"""
        if not object_ids:
            return []
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create placeholders for the IN clause
        placeholders = ','.join('?' * len(object_ids))
        cursor.execute(f'''
            SELECT id, content, bbox, center, is_hover_change, hover_tooltip 
            FROM objects 
            WHERE id IN ({placeholders}) 
            AND (is_hover_change = 0 OR is_hover_change IS NULL OR hover_tooltip IS NULL)
        ''', object_ids)
        
        records = cursor.fetchall()
        objects = []
        for record in records:
            id, content, bbox_str, center_str, is_hover_change, hover_tooltip = record
            bbox = json.loads(bbox_str)
            center = json.loads(center_str)
            objects.append({
                "id": id,
                "content": content,
                "bbox": bbox,
                "center": center,
                "is_hover_change": bool(is_hover_change) if is_hover_change is not None else False,
                "hover_tooltip": hover_tooltip
            })
        
        return objects