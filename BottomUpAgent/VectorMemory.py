import chromadb
import torch
import clip
import cv2
import numpy as np
from PIL import Image
import json
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
import logging
from chromadb.utils.embedding_functions import known_embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings

class VectorMemory:
    def __init__(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.game_name = config['game_name']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        game_name_normalized = self.game_name.lower().replace(" ", "_")
        self.vector_db_path = os.path.join(config.get('database', {}).get('vector_db_path', os.path.join('data', 'vector')), game_name_normalized)

        self.similarity_threshold = config.get('vector_memory', {}).get('similarity_threshold', 0.5)
        self.vector_dim = 512  # CLIP ViT-B/32 output dimension
        self.embedding_type = config.get('vector_memory', {}).get('embedding_type', None)
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        clear_on_init = config.get('vector_memory', {}).get('clear_on_init', False)
        if clear_on_init:
            self.logger.info("CLEARING VECTOR DATABASE AS CONFIGURED")
            self.clear_database()
        
        self._init_chroma_db()
        
    def _init_chroma_db(self):
        """Initialize Chroma vector database"""
        try:
            if not self.embedding_type or self.embedding_type == "":
                print(f"Use default embedding in the vector database as {self.embedding_type}.")
                self.object_collection = self.client.get_or_create_collection(
                    name="ui_objects",
                    metadata={"hnsw:space": "cosine"}
                )
            elif self.embedding_type in known_embedding_functions:
                print(f"Use {self.embedding_type} to embed in the vector database.")
                try:
                    embedding_function = known_embedding_functions[self.embedding_type]()
                    self.object_collection = self.client.get_or_create_collection(
                        name="ui_objects",
                        embedding_function=embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )
                except Exception as e:
                    self.logger.error(f"Failed to use embedding_type '{self.embedding_type}': {e}")
                    self.logger.info("Falling back to default embedding.")
                    self.object_collection = self.client.get_or_create_collection(
                        name="ui_objects",
                        metadata={"hnsw:space": "cosine"}
                    )
            else:
                self.logger.warning(f"Unsupported embedding_type '{self.embedding_type}', using default embedding.")
                self.object_collection = self.client.get_or_create_collection(
                    name="ui_objects",
                    metadata={"hnsw:space": "cosine"}
                )
            
            self.logger.info(f"CHROMA DB INITIALIZED: {self.vector_db_path}")
            
        except Exception as e:
            self.logger.error(f"CHROMA DB INITIALIZATION FAILED: {e}")
            raise
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image using CLIP model
        
        Args:
            image: OpenCV format image (BGR)
            
        Returns:
            Normalized image feature vector
        """
        try:
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            pil_image = Image.fromarray(image_rgb)
            
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"IMAGE ENCODING FAILED: {e}")
            return np.zeros(self.vector_dim)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using CLIP model
        
        Args:
            text: Input text
            
        Returns:
            Normalized text feature vector
        """
        try:
            text_input = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"TEXT ENCODING FAILED: {e}")
            return np.zeros(self.vector_dim)
    
    def _preprocess_object_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Standardize object image
        
        Args:
            image: Original image
            target_size: Target size
            
        Returns:
            Preprocessed image
        """
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
            
        return resized
    
    def _generate_object_hash(self, obj: Dict) -> str:
        """
        Generate unique hash for object
        
        Args:
            obj: Object dictionary
            
        Returns:
            Hash string
        """
        # Use bbox, content and image hash to generate unique identifier
        hash_data = {
            'bbox': obj['bbox'],
            'content': obj.get('content', ''),
            'area': obj['area'],
            'image_hash': obj.get('hash', '')
        }
        
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def store_object(self, obj: Dict) -> str:
        """
        Store single object to vector database
        
        Args:
            obj: Object dictionary containing image, bbox, content and other info
            
        Returns:
            Stored object ID
        """
        try:
            obj_type = obj.get('type', 'None')
            
            if obj_type == 'icon':
                processed_image = self._preprocess_object_image(obj['image'])
                embedding_vector = self.encode_image(processed_image)
            elif obj_type == 'text':
                # For text type, use content text embedding
                content = obj.get('content', '')
                if content:
                    embedding_vector = self.encode_text(content)
                else:
                    # If no content, fallback to image embedding
                    processed_image = self._preprocess_object_image(obj['image'])
                    embedding_vector = self.encode_image(processed_image)
            else:
                # Other types default to image embedding
                processed_image = self._preprocess_object_image(obj['image'])
                embedding_vector = self.encode_image(processed_image)
            
            # Generate unique ID
            object_hash = self._generate_object_hash(obj)
            object_id = f"obj_{obj_type}_{object_hash[:8]}"
            
            # Prepare metadata
            metadata = {
                'object_id': str(obj.get('id', '')),
                'bbox': json.dumps(obj['bbox']),
                'area': obj['area'],
                'hash': obj.get('hash', ''),
                'center': json.dumps(obj.get('center', [0, 0])),
                'type': obj_type,
                'timestamp': datetime.now().isoformat(),
                'embedding_type': 'image' if obj_type == 'icon' or (obj_type == 'text' and not obj.get('content', '')) else 'text' if obj_type == 'text' else 'image'
            }
            
            # add embedding, content, metadatas to Chroma
            self.object_collection.add(
                embeddings=[embedding_vector.tolist()],
                documents=[obj.get('content', '')],
                metadatas=[metadata],
                ids=[object_id]
            )
            
            self.logger.info(f"OBJECT STORED: {object_id}")
            return object_id
            
        except Exception as e:
            self.logger.error(f"OBJECT STORAGE FAILED: {e}")
            return ""
    
    def batch_store_objects(self, objects: List[Dict]) -> List[str]:
        """
        Batch store objects
        
        Args:
            objects: Object list
            
        Returns:
            List of stored object IDs
        """
        if not objects:
            return []
            
        try:
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for obj in objects:
                obj_type = obj.get('type', 'None')

                if obj_type == 'icon':
                    processed_image = self._preprocess_object_image(obj['image'])
                    embedding_vector = self.encode_image(processed_image)
                elif obj_type == 'text':
                    content = obj.get('content', '')
                    if content:
                        embedding_vector = self.encode_text(content)
                    else:
                        processed_image = self._preprocess_object_image(obj['image'])
                        embedding_vector = self.encode_image(processed_image)
                else:
                    processed_image = self._preprocess_object_image(obj['image'])
                    embedding_vector = self.encode_image(processed_image)
                
                object_hash = self._generate_object_hash(obj)
                object_id = f"obj_{obj_type}_{object_hash[:8]}"
                
                metadata = {
                    'object_id': str(obj.get('id', '')),
                    'bbox': json.dumps(obj['bbox']),
                    'area': obj['area'],
                    'hash': obj.get('hash', ''),
                    'center': json.dumps(obj.get('center', [0, 0])),
                    'type': obj_type,
                    'timestamp': datetime.now().isoformat(),
                    'embedding_type': 'image' if obj_type == 'icon' or (obj_type == 'text' and not obj.get('content', '')) else 'text' if obj_type == 'text' else 'image'
                }
                
                embeddings.append(embedding_vector.tolist())
                documents.append(obj.get('content', ''))
                metadatas.append(metadata)
                ids.append(object_id)
            
            self.object_collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"BATCH STORAGE COMPLETED: {len(objects)} objects")
            return ids
            
        except Exception as e:
            self.logger.error(f"BATCH STORAGE FAILED: {e}")
            return []
    
    def find_similar_objects(self, query_obj: Dict, threshold: float = None, top_k: int = 5, embedding_type="default") -> List[Dict]:
        """
        Find similar objects
        
        Args:
            query_obj: Query object
            threshold: Similarity threshold
            top_k: Maximum number of results to return
            
        Returns:
            List of similar objects, sorted by similarity in descending order
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        try:
            if embedding_type == "CLIP" or (self.embedding_type == "CLIP"):
                processed_image = self._preprocess_object_image(query_obj['image'])
                query_vector = self.encode_image(processed_image)
                
                results = self.object_collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    include=['embeddings', 'documents', 'metadatas', 'distances']
                )
            elif embedding_type in known_embedding_functions:
                processed_image = self._preprocess_object_image(query_obj['image'])
                query_vector = self.encode_image(processed_image)
                
                results = self.object_collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    include=['embeddings', 'documents', 'metadatas', 'distances']
                )
            else:
                results = self.object_collection.query(
                    query_texts=[query_obj.get('content', '')],
                    n_results=top_k,
                    include=['embeddings', 'documents', 'metadatas', 'distances']
                )
            
            similar_objects = []
            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    similarity = 1 - distance
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        similar_objects.append({
                            'id': results['ids'][0][i],
                            'similarity': similarity,
                            'content': results['documents'][0][i],
                            'bbox': json.loads(metadata['bbox']),
                            'area': metadata['area'],
                            'center': json.loads(metadata['center']),
                            'type': metadata['type'],
                            'object_id': metadata['object_id'],
                            'timestamp': metadata['timestamp']
                        })
            
            return similar_objects
            
        except Exception as e:
            self.logger.error(f"SIMILARITY SEARCH FAILED: {e}")
            return []

    def _check_instance_similarity(self, obj1: Dict, obj2: Dict) -> Tuple[bool, float, str]:
        """
        Check instance-level similarity between two objects using Hash and Geometric distance
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            (is_duplicate, similarity_score, reason)
        """
        # [1] Hash similarity check (highest priority)
        hash1 = obj1.get('hash', '')
        hash2 = obj2.get('hash', '')
        
        if hash1 and hash2 and hash1 == hash2:
            return True, 1.0, 'Hash'
        
        # [2] Geometric distance check
        center1 = obj1.get('center', [0, 0])
        center2 = obj2.get('center', [0, 0])
        
        if center1 and center2:
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance <= 10:  # Centers within 10 pixels
                similarity = 1.0 - (distance / 1000)
                return True, similarity, f'Geo_{distance:.1f}px'
        
        # [3] Hash similarity (partial match for similar but not identical hashes)
        if hash1 and hash2:
            try:
                # Calculate Hamming distance similarity of hashes
                hash1_bin = bin(int(hash1, 16))[2:].zfill(64)
                hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
                
                hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1_bin, hash2_bin))
                
                # Very similar hashes (≤4 bit differences) are considered duplicates
                if hamming_distance <= 4:
                    similarity = 1.0 - (hamming_distance / 64.0)
                    return True, similarity, f'similar_hash_{hamming_distance}bits'
                    
            except ValueError:
                # If hash format is incorrect, use string similarity
                common_chars = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
                if len(hash1) > 0 and len(hash2) > 0:
                    string_similarity = common_chars / max(len(hash1), len(hash2))
                    if string_similarity > 0.95:  # 95% string similarity
                        return True, string_similarity, f'string_hash_{string_similarity:.3f}'
        
        return False, 0.0, 'no_match'

    def deduplicate_objects(self, objects: List[Dict], threshold: float = None, 
                           text_weight: float = 0.3, image_weight: float = 0.5, 
                           bbox_weight: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        Object deduplication using optimized mixed similarity detection
        Optimization strategies:
        1. Use more comprehensive candidate object set
        2. Adopt multi-stage similarity detection
        3. Adjust weights and thresholds for deduplication scenarios
        
        Args:
            objects: List of objects to process
            threshold: Deduplication threshold
            text_weight: Text similarity weight
            image_weight: Image similarity weight
            bbox_weight: Bounding box similarity weight
            
        Returns:
            (new objects list, duplicate objects list)
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        new_objects = []
        duplicate_objects = []
        
        collection_count = self.object_collection.count()
        database_is_empty = collection_count == 0
        
        if database_is_empty:
            self.logger.info("DATABASE IS EMPTY - Skipping database deduplication, only checking batch duplicates")
        
        # Generate hash for each object to improve comparison efficiency
        for obj in objects:
            if 'hash' not in obj or not obj['hash']:
                obj['hash'] = self._generate_object_hash(obj)
        
        for i, obj in enumerate(objects):
            found_duplicate = False
            
                
            # [1] Instance-level similarity check in database (Hash only, keep it simple)
            if not database_is_empty and obj.get('hash'):
                try:
                    hash_results = self.object_collection.get(
                        where={"hash": obj['hash']},
                        include=['metadatas', 'documents']
                    )
                    
                    if hash_results['ids'] and len(hash_results['ids']) > 0:
                        # Check all matched objects with instance similarity
                        for i in range(len(hash_results['ids'])):
                            matched_id = hash_results['ids'][i]
                            matched_metadata = hash_results['metadatas'][i]
                            matched_content = hash_results['documents'][i]
                            
                            candidate = {
                                'id': matched_id,
                                'content': matched_content,
                                'hash': matched_metadata['hash'],
                                'bbox': json.loads(matched_metadata['bbox']),
                                'area': matched_metadata['area'],
                                'center': json.loads(matched_metadata['center']),
                                'type': matched_metadata['type'],
                                'object_id': matched_metadata['object_id']
                            }
                            
                            is_duplicate, similarity, reason = self._check_instance_similarity(obj, candidate)
                            if is_duplicate:
                                duplicate_objects.append({
                                    'original': obj,
                                    'similar': candidate,
                                    'similarity': similarity,
                                    'reason': f'instance_db_{reason}'
                                })
                                print(f"🔄 DUPLICATE DETECTED (Hash/Geo): {obj.get('content', 'Unknown')} ({reason}, Sim: 1.000)")
                                found_duplicate = True
                                break
                        
                except Exception as e:
                    self.logger.debug(f"⚠️  DATABASE HASH QUERY FAILED: {e}")
                    # Fallback to text query + hash comparison (keep original logic)
                    try:
                        hash_results = self.object_collection.query(
                            query_texts=[obj.get('content', 'unknown')],
                            n_results=20,  # Get more results for hash comparison
                            include=['metadatas', 'documents', 'distances']
                        )
                        
                        # [1b] Check if there are same hashes in returned results
                        for j in range(len(hash_results['ids'][0])):
                            metadata = hash_results['metadatas'][0][j]
                            stored_hash = metadata.get('hash', '')
                            
                            if stored_hash and stored_hash == obj['hash']:
                                candidate = {
                                    'id': hash_results['ids'][0][j],
                                    'content': hash_results['documents'][0][j],
                                    'hash': stored_hash,
                                    'bbox': json.loads(metadata['bbox']),
                                    'area': metadata['area'],
                                    'center': json.loads(metadata['center']),
                                    'type': metadata['type'],
                                    'object_id': metadata['object_id']
                                }
                                
                                is_duplicate, similarity, reason = self._check_instance_similarity(obj, candidate)
                                if is_duplicate:
                                    duplicate_objects.append({
                                        'original': obj,
                                        'similar': candidate,
                                        'similarity': similarity,
                                        'reason': f'instance_db_{reason}'
                                    })
                                    print(f"🔄 DUPLICATE DETECTED (Hash/Geo): {obj.get('content', 'Unknown')} ({reason}, Sim: {similarity:.3f})")
                                    found_duplicate = True
                                    break
                    except Exception as e2:
                        self.logger.debug(f"⚠️  FALLBACK HASH QUERY ALSO FAILED: {e2}")
            
            if found_duplicate:
                continue
                
            # [2]: Use mixed query for more precise similarity detection (only when database is not empty)
            if not database_is_empty:
                similar_objects = self.query_collection_mixed(
                    obj, 
                    top_k=3,  # Get top 3 most similar objects for comparison
                    text_weight=text_weight, 
                    image_weight=image_weight, 
                    bbox_weight=bbox_weight
                )

                if similar_objects:
                    best_match = similar_objects[0]
                    if best_match['similarity'] >= threshold:
                        duplicate_objects.append({
                            'original': obj,
                            'similar': best_match,
                            'similarity': best_match['similarity'],
                            'reason': 'mixed_similarity'
                        })
                        print(f"🔄 DUPLICATE DETECTED (Mixed): {obj.get('content', 'Unknown')} (Sim: {best_match['similarity']:.3f})")
                        found_duplicate = True
            
            if not found_duplicate:
                new_objects.append(obj)
        
        return new_objects, duplicate_objects
    
    def update_object_with_vector_storage(self, objects: List[Dict], 
                                         text_weight: float = 0.3, image_weight: float = 0.5, 
                                         bbox_weight: float = 0.2, similarity_threshold: float = None) -> List[Dict]:
        """
        Update objects and store to vector database using mixed similarity detection
        
        Args:
            objects: Object list
            text_weight: Text similarity weight
            image_weight: Image similarity weight
            bbox_weight: Bounding box similarity weight
            similarity_threshold: Similarity threshold
            
        Returns:
            Processed object list
        """
        if not objects:
            return []
        
        # Use mixed similarity for deduplication
        new_objects, duplicate_objects = self.deduplicate_objects(
            objects, 
            threshold=similarity_threshold, 
            text_weight=text_weight, 
            image_weight=image_weight, 
            bbox_weight=bbox_weight
        )
        
        # TODO: missing ID (default as 'None') for old objects
        if new_objects:
            stored_ids = self.batch_store_objects(new_objects)
            for i, obj in enumerate(new_objects):
                if i < len(stored_ids):
                    obj['vector_id'] = stored_ids[i]
        
        for dup in duplicate_objects:
            original = dup['original']
            similar = dup['similar']
            
            original['id'] = similar.get('object_id', similar.get('id'))
            original['vector_id'] = similar.get('id')
            print(f"🔗 OBJECT DEDUPLICATED: {original.get('content', 'Unknown')} -> Vector ID: {similar.get('id')}, SQL ID: {original.get('id')} (Similarity: {dup.get('similarity', 0):.3f})")
        
        all_objects = new_objects + [dup['original'] for dup in duplicate_objects]
        
        self.logger.info(f"📊 OBJECT PROCESSING COMPLETED: {len(new_objects)} new, {len(duplicate_objects)} duplicates")
        return all_objects
    
    def query_collection_by_text(self, content_query: str, top_k: int = 10) -> List[Dict]:
        """
        Search objects by content
        
        Args:
            content_query: Content query string
            top_k: Maximum number of results to return
            
        Returns:
            List of matched objects
        """
        try:
            if not self.embedding_type or self.embedding_type == "":
                results = self.object_collection.query(
                    query_texts=[content_query],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            elif self.embedding_type == "CLIP":
                text_vector = self.encode_text(content_query)
                results = self.object_collection.query(
                    query_embeddings=[text_vector.tolist()],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            elif self.embedding_type in known_embedding_functions:
                results = self.object_collection.query(
                    query_texts=[content_query],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            else:
                results = self.object_collection.query(
                    query_texts=[content_query],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )

            objects = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    embedding = results['embeddings'][0][i] if results.get('embeddings') else None
                    # Convert distance to similarity: for cosine distance, similarity = 1 - distance/2
                    # This ensures similarity is in [0, 1] range, 1 means most similar, 0 means least similar
                    similarity = max(0, 1 - distance / 2)
                    objects.append({
                        'id': results['ids'][0][i],
                        'content': doc,
                        'bbox': json.loads(metadata['bbox']),
                        'area': metadata['area'],
                        'center': json.loads(metadata['center']),
                        'type': metadata['type'],
                        'object_id': metadata['object_id'],
                        'hash': metadata.get('hash', ''),
                        'distance': distance,
                        'similarity': similarity,
                        'embedding': embedding
                    })
            
            return objects

        except Exception as e:
            self.logger.error(f"Content search failed: {e}")
            return []

    def query_collection_by_image(self, query_image: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Search objects by image
        
        Args:
            query_image: Query image numpy array
            top_k: Maximum number of results to return
            
        Returns:
            List of matched objects
        """
        try:
            # Keep logic consistent with _init_chroma_db
            if not self.embedding_type or self.embedding_type == "":
                # Default embedding does not support image queries
                self.logger.warning("Default embedding does not support image queries")
                return []
            elif self.embedding_type == "CLIP":
                # Use CLIP to encode images
                image_vector = self.encode_image(query_image)
                # Use vector search
                results = self.object_collection.query(
                    query_embeddings=[image_vector.tolist()],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            elif self.embedding_type in known_embedding_functions:
                # For multimodal embedding functions (like open_clip), use image queries
                if self.embedding_type == "open_clip":
                    # OpenCLIPEmbeddingFunction supports image queries
                    results = self.object_collection.query(
                        query_images=[query_image],
                        n_results=top_k,
                        include=['documents', 'metadatas', 'distances', 'embeddings']
                    )
                else:
                    # Other embedding functions may not support image queries
                    self.logger.warning(f"Embedding function '{self.embedding_type}' may not support image queries")
                    return []
            else:
                # Unsupported embedding_type
                self.logger.warning(f"Unsupported embedding_type '{self.embedding_type}' for image queries")
                return []

            objects = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    embedding = results['embeddings'][0][i] if results.get('embeddings') else None
                    # Convert distance to similarity: for cosine distance, similarity = 1 - distance/2
                    # This ensures similarity is in [0, 1] range, 1 means most similar, 0 means least similar
                    similarity = max(0, 1 - distance / 2)
                    objects.append({
                        'id': results['ids'][0][i],
                        'content': doc,
                        'bbox': json.loads(metadata['bbox']),
                        'area': metadata['area'],
                        'center': json.loads(metadata['center']),
                        'type': metadata['type'],
                        'object_id': metadata['object_id'],
                        'hash': metadata.get('hash', ''),
                        'distance': distance,
                        'similarity': similarity,
                        'embedding': embedding
                    })
            
            return objects

        except Exception as e:
            self.logger.error(f"Image search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get vector database statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            count = self.object_collection.count()
            return {
                'total_objects': count,
                'game_name': self.game_name,
                'vector_dim': self.vector_dim,
                'db_path': self.vector_db_path
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection statistics: {e}")
            return {}
    
    def query_collection_mixed(self, query_obj: Dict, top_k: int = 10, 
                              text_weight: float = 0.4, image_weight: float = 0.4, 
                              bbox_weight: float = 0.2, bbox_scale: float = 1000.0) -> List[Dict]:
        """
        Mixed query method combining text similarity, image similarity and bbox position distance
        Optimized version: Get comprehensive candidate object set to avoid missing similar objects
        
        Args:
            query_obj: Query object containing content, cropped_image, bbox and other fields
            top_k: Number of results to return
            text_weight: Text similarity weight
            image_weight: Image similarity weight  
            bbox_weight: Bbox distance weight
            bbox_scale: Bbox distance normalization scale factor
            
        Returns:
            Object list sorted by comprehensive similarity
        """
        try:
            # Generate hash for query_obj if not present
            if 'hash' not in query_obj or not query_obj['hash']:
                query_obj['hash'] = self._generate_object_hash(query_obj)
            
            # Get total object count in database
            collection_count = self.object_collection.count()
            if collection_count == 0:
                self.logger.info("📭 VECTOR DATABASE IS EMPTY - No candidates available for similarity search")
                return []
            
            # Dynamically adjust candidate object count to ensure no similar objects are missed
            # For deduplication scenarios, we need a more comprehensive candidate set
            if top_k == 1:  # Deduplication scenario, get more candidate objects
                candidate_k = min(collection_count, max(500, collection_count // 2))
            else:  # Regular query scenario
                candidate_k = min(collection_count, max(top_k * 10, 100))
            
            # Get candidate object set - use multiple strategies to ensure coverage
            all_candidates = {}
            
            # [1] Text query
            if 'content' in query_obj and query_obj['content']:
                text_results = self.query_collection_by_text(query_obj['content'], candidate_k)
                for result in text_results:
                    all_candidates[result['id']] = result
            
            # [2]: Image query
            if 'image' in query_obj and query_obj['image'] is not None:
                image_results = self.query_collection_by_image(query_obj['image'], candidate_k)
                for result in image_results:
                    if result['id'] not in all_candidates:
                        all_candidates[result['id']] = result
            
            # [3]: If candidate objects are still insufficient, get all objects for comprehensive comparison
            if len(all_candidates) < candidate_k:
                try:
                    remaining_needed = candidate_k - len(all_candidates)
                    results = self.object_collection.query(
                        query_texts=["*"], 
                        n_results=min(remaining_needed, collection_count),
                        include=['metadatas', 'documents', 'distances', 'embeddings']
                    )
                    
                    for i in range(len(results['ids'][0])):
                        obj_id = results['ids'][0][i]
                        if obj_id not in all_candidates:  # Avoid duplicates
                            metadata = results['metadatas'][0][i]
                            doc = results['documents'][0][i]
                            distance = results['distances'][0][i]
                            embedding = results['embeddings'][0][i] if results.get('embeddings') else None
                            similarity = max(0, 1 - distance / 2)
                            
                            all_candidates[obj_id] = {
                                'id': obj_id,
                                'content': doc,
                                'bbox': json.loads(metadata['bbox']),
                                'area': metadata['area'],
                                'center': json.loads(metadata['center']),
                                'type': metadata['type'],
                                'object_id': metadata['object_id'],
                                'hash': metadata.get('hash', ''),
                                'distance': distance,
                                'similarity': similarity,
                                'embedding': embedding
                            }
                except Exception as e:
                    self.logger.warning(f"⚠️  FAILED TO GET ADDITIONAL CANDIDATES: {e}")
            
            # complete text and image similarity for all candidate objects
            query_text_embedding = None
            
            # Prepare query embeddings
            if 'content' in query_obj and query_obj['content']:
                query_text_embedding = self.encode_text(query_obj['content'])
            
            # similarity for each candidate object
            for obj_id, candidate in all_candidates.items():
                text_similarity = 0.0
                image_similarity = 0.0
                
                # text similarity
                if query_text_embedding is not None and candidate.get('content'):
                    try:
                        candidate_text_embedding = self.encode_text(candidate['content'])
                        # cosine similarity
                        text_cosine_sim = np.dot(query_text_embedding, candidate_text_embedding) / (
                            np.linalg.norm(query_text_embedding) * np.linalg.norm(candidate_text_embedding)
                        )
                        text_similarity = max(0, (text_cosine_sim + 1) / 2)  # Convert to [0,1] range
                    except Exception as e:
                        self.logger.debug(f"⚠️  TEXT SIMILARITY CALCULATION FAILED: {e}")
                
                # Optimized image similarity calculation - use hash comparison and more precise thresholds
                if 'hash' in query_obj and query_obj['hash'] and 'hash' in candidate and candidate['hash']:
                    try:
                        query_hash = query_obj['hash']
                        candidate_hash = candidate['hash']
                        
                        # If hashes are identical, similarity is 1.0
                        if query_hash == candidate_hash:
                            image_similarity = 1.0
                        else:
                            # Calculate Hamming distance similarity of hashes
                            try:
                                query_bin = bin(int(query_hash, 16))[2:].zfill(64)  # Convert to 64-bit binary
                                candidate_bin = bin(int(candidate_hash, 16))[2:].zfill(64)
                                
                                # Calculate Hamming distance
                                hamming_distance = sum(c1 != c2 for c1, c2 in zip(query_bin, candidate_bin))
                                # Convert to similarity, use more sensitive threshold
                                max_distance = 64.0
                                # For deduplication scenarios, we need stricter similarity judgment
                                if hamming_distance <= 8:  # Allow small differences
                                    image_similarity = 1.0 - (hamming_distance / max_distance) * 0.5
                                else:
                                    image_similarity = max(0, 1 - hamming_distance / max_distance)
                            except ValueError:
                                # If hash format is incorrect, use string similarity
                                common_chars = sum(c1 == c2 for c1, c2 in zip(query_hash, candidate_hash))
                                image_similarity = common_chars / max(len(query_hash), len(candidate_hash))
                    except Exception as e:
                        self.logger.debug(f"⚠️  IMAGE HASH SIMILARITY CALCULATION FAILED: {e}")
                        image_similarity = 0.0
                else:
                    image_similarity = 0.0
                
                candidate['text_similarity'] = text_similarity
                candidate['image_similarity'] = image_similarity
            
            # Calculate bbox distance similarity
            query_bbox = query_obj.get('bbox', None)
            if query_bbox:
                query_center = query_obj.get('center', None)
                
                for obj_id, candidate in all_candidates.items():
                    candidate_center = candidate.get('center', None)
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt((query_center[0] - candidate_center[0])**2 + 
                                     (query_center[1] - candidate_center[1])**2)
                    
                    # Check if centers are very close (within 10 pixels) - consider as duplicate
                    if distance <= 10:
                        bbox_similarity = 1.0  # Very high similarity for close centers
                    else:
                        # Convert distance to similarity (smaller distance, higher similarity)
                        bbox_similarity = max(0, 1 - distance / bbox_scale)
                    
                    candidate['bbox_similarity'] = bbox_similarity
                    candidate['center_distance'] = distance  # Store distance for debugging
            else:
                # If no bbox info, set bbox similarity to 0.5 (neutral value)
                for candidate in all_candidates.values():
                    candidate['bbox_similarity'] = 0.5
                    candidate['center_distance'] = float('inf')
            
            # Calculate comprehensive similarity
            for candidate in all_candidates.values():
                # Normalize weights
                total_weight = text_weight + image_weight + bbox_weight
                norm_text_weight = text_weight / total_weight
                norm_image_weight = image_weight / total_weight
                norm_bbox_weight = bbox_weight / total_weight
                
                # Calculate weighted comprehensive similarity
                mixed_similarity = (candidate['text_similarity'] * norm_text_weight + 
                                  candidate['image_similarity'] * norm_image_weight + 
                                  candidate['bbox_similarity'] * norm_bbox_weight)
                
                candidate['mixed_similarity'] = mixed_similarity
                candidate['similarity'] = mixed_similarity  # Update main similarity field
            
            # Sort by comprehensive similarity and return top_k results
            sorted_results = sorted(all_candidates.values(), 
                                  key=lambda x: x['mixed_similarity'], reverse=True)
            
            return sorted_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ MIXED QUERY FAILED: {e}")
            return []
    
    def update_object_metadata(self, vector_id: str, sql_id: int):
        """
        Update the object_id metadata in VectorDB after SQL storage
        
        Args:
            vector_id: The vector database ID
            sql_id: The SQL database ID (lastrowid)
        """
        try:
            # Get current metadata
            result = self.object_collection.get(ids=[vector_id], include=['metadatas'])
            if result['ids'] and len(result['ids']) > 0:
                current_metadata = result['metadatas'][0]
                # Update object_id with SQL ID
                current_metadata['object_id'] = str(sql_id)
                
                # Update the metadata in VectorDB
                self.object_collection.update(
                    ids=[vector_id],
                    metadatas=[current_metadata]
                )
                self.logger.debug(f"🔄 UPDATED VECTOR METADATA: Vector_ID={vector_id} -> SQL_ID={sql_id}")
                return True
            else:
                self.logger.warning(f"⚠️  VECTOR ID NOT FOUND: {vector_id}")
                return False
        except Exception as e:
            self.logger.error(f"❌ FAILED TO UPDATE METADATA: Vector_ID={vector_id}, Error={e}")
            return False

    def batch_update_object_metadata(self, vector_sql_id_pairs: List[Tuple[str, int]]):
        """
        Batch update object_id metadata in VectorDB
        
        Args:
            vector_sql_id_pairs: List of (vector_id, sql_id) tuples
        """
        try:
            vector_ids = [pair[0] for pair in vector_sql_id_pairs]
            sql_ids = [pair[1] for pair in vector_sql_id_pairs]
            
            # Get current metadata for all objects
            result = self.object_collection.get(ids=vector_ids, include=['metadatas'])
            
            if result['ids'] and len(result['ids']) > 0:
                updated_metadatas = []
                for i, metadata in enumerate(result['metadatas']):
                    if i < len(sql_ids):
                        metadata['object_id'] = str(sql_ids[i])
                        updated_metadatas.append(metadata)
                
                # Batch update metadata
                self.object_collection.update(
                    ids=vector_ids[:len(updated_metadatas)],
                    metadatas=updated_metadatas
                )
                
                self.logger.info(f"🔄 BATCH METADATA UPDATE COMPLETED: {len(updated_metadatas)} objects")
                return True
            else:
                self.logger.warning(f"⚠️  NO VECTOR IDS FOUND FOR BATCH UPDATE")
                return False
        except Exception as e:
            self.logger.error(f"❌ BATCH METADATA UPDATE FAILED: {e}")
            return False


    def clear_database(self):
        """Clear vector database"""
        try:
            # Delete collection
            self.client.delete_collection("ui_objects")
            
            # Recreate
            # self.object_collection = self.client.create_collection(
            #     name="ui_objects",
            #     metadata={"hnsw:space": "cosine"}
            # )
            
            self.logger.info("Vector database cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}")