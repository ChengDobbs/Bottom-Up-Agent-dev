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
    """
    基于Chroma向量数据库的对象存储和检索系统
    支持图像向量化、相似度匹配和智能去重
    """
    
    def __init__(self, config):
        # 日志配置 - 首先初始化
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.game_name = config['game_name']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化CLIP模型用于图像编码
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 向量数据库配置 - 标准化路径
        game_name_normalized = self.game_name.lower().replace(" ", "_")
        self.vector_db_path = os.path.join(config.get('database', {}).get('vector_db_path', os.path.join('data', 'vector')), game_name_normalized)

        self.similarity_threshold = config.get('vector_memory', {}).get('similarity_threshold', 0.5)
        self.vector_dim = 512  # CLIP ViT-B/32 输出维度
        
        # 获取配置中的embedding_type
        self.embedding_type = config.get('vector_memory', {}).get('embedding_type', None)
        
        # 创建持久化客户端
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        
        # 检查是否需要清空数据库
        clear_on_init = config.get('vector_memory', {}).get('clear_on_init', False)
        if clear_on_init:
            self.logger.info("配置要求清空向量数据库")
            self.clear_database()
        
        # 初始化Chroma数据库 - get_or_create_collection会自动处理创建逻辑
        self._init_chroma_db()
        
    def _init_chroma_db(self):
        """初始化Chroma向量数据库"""
        try:
            # 处理embedding_type配置
            if not self.embedding_type or self.embedding_type == "":
                # 空字符串或None，使用默认embedding
                print(f"Use default embedding in the vector database as {self.embedding_type}.")
                # 创建持久化客户端
                self.object_collection = self.client.get_or_create_collection(
                    name="ui_objects",
                    metadata={"hnsw:space": "cosine"}
                )
            elif self.embedding_type == "CLIP":
                # 使用现成的CLIP编码方法
                print(f"Use CLIP embedding with custom encode/decode methods.")
                self.object_collection = self.client.get_or_create_collection(
                    embedding_function=CLIPEmbeddingFunction(),
                    name="ui_objects",
                    metadata={"hnsw:space": "cosine"}
                )
            elif self.embedding_type in known_embedding_functions:
                # 使用known_embedding_functions中的embedding
                print(f"Use {self.embedding_type} to embed in the vector database.")
                try:
                    # 实例化embedding函数
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
                # 不支持的embedding_type，使用默认
                self.logger.warning(f"Unsupported embedding_type '{self.embedding_type}', using default embedding.")
                self.object_collection = self.client.get_or_create_collection(
                    name="ui_objects",
                    metadata={"hnsw:space": "cosine"}
                )
            
            self.logger.info(f"Chroma数据库初始化成功: {self.vector_db_path}")
            
        except Exception as e:
            self.logger.error(f"Chroma数据库初始化失败: {e}")
            raise
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        使用CLIP模型对图像进行向量编码
        
        Args:
            image: OpenCV格式的图像 (BGR)
            
        Returns:
            归一化的图像特征向量
        """
        try:
            # 转换为PIL图像 (RGB)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            pil_image = Image.fromarray(image_rgb)
            
            # 预处理并编码
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                # 归一化向量
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"图像编码失败: {e}")
            return np.zeros(self.vector_dim)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        使用CLIP模型对文本进行向量编码
        
        Args:
            text: 输入文本
            
        Returns:
            归一化的文本特征向量
        """
        try:
            # 对文本进行tokenize
            text_input = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                # 归一化向量
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"文本编码失败: {e}")
            return np.zeros(self.vector_dim)
    
    def _preprocess_object_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        标准化对象图像
        
        Args:
            image: 原始图像
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像
        """
        # 调整大小
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 确保是3通道
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
            
        return resized
    
    def _generate_object_hash(self, obj: Dict) -> str:
        """
        生成对象的唯一哈希值
        
        Args:
            obj: 对象字典
            
        Returns:
            哈希字符串
        """
        # 使用bbox、content和图像hash生成唯一标识
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
        存储单个对象到向量数据库
        
        Args:
            obj: 包含image、bbox、content等信息的对象字典
            
        Returns:
            存储的对象ID
        """
        try:
            obj_type = obj.get('type', 'None')
            
            # 根据对象类型选择不同的embedding方式
            if obj_type == 'icon':
                # 对于icon类型，使用cropped图像的embedding
                processed_image = self._preprocess_object_image(obj['image'])
                embedding_vector = self.encode_image(processed_image)
            elif obj_type == 'text':
                # 对于text类型，使用content的文本embedding
                content = obj.get('content', '')
                if content:
                    embedding_vector = self.encode_text(content)
                else:
                    # 如果没有content，回退到图像embedding
                    processed_image = self._preprocess_object_image(obj['image'])
                    embedding_vector = self.encode_image(processed_image)
            else:
                # 其他类型默认使用图像embedding
                processed_image = self._preprocess_object_image(obj['image'])
                embedding_vector = self.encode_image(processed_image)
            
            # 生成唯一ID
            object_hash = self._generate_object_hash(obj)
            object_id = f"obj_{obj_type}_{object_hash[:8]}"
            
            # 准备元数据
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
            
            self.logger.info(f"对象存储成功: {object_id}")
            return object_id
            
        except Exception as e:
            self.logger.error(f"对象存储失败: {e}")
            return ""
    
    def batch_store_objects(self, objects: List[Dict]) -> List[str]:
        """
        批量存储对象
        
        Args:
            objects: 对象列表
            
        Returns:
            存储的对象ID列表
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
                
                # 根据对象类型选择不同的embedding方式
                if obj_type == 'icon':
                    # 对于icon类型，使用cropped图像的embedding
                    processed_image = self._preprocess_object_image(obj['image'])
                    embedding_vector = self.encode_image(processed_image)
                elif obj_type == 'text':
                    # 对于text类型，使用content的文本embedding
                    content = obj.get('content', '')
                    if content:
                        embedding_vector = self.encode_text(content)
                    else:
                        # 如果没有content，回退到图像embedding
                        processed_image = self._preprocess_object_image(obj['image'])
                        embedding_vector = self.encode_image(processed_image)
                else:
                    # 其他类型默认使用图像embedding
                    processed_image = self._preprocess_object_image(obj['image'])
                    embedding_vector = self.encode_image(processed_image)
                
                # 生成ID和元数据
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
            
            # 批量存储
            self.object_collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"批量存储完成: {len(objects)} 个对象")
            return ids
            
        except Exception as e:
            self.logger.error(f"批量存储失败: {e}")
            return []
    
    def find_similar_objects(self, query_obj: Dict, threshold: float = None, top_k: int = 5, embedding_type="default") -> List[Dict]:
        """
        查找相似的对象
        
        Args:
            query_obj: 查询对象
            threshold: 相似度阈值
            top_k: 返回的最大结果数
            
        Returns:
            相似对象列表，按相似度降序排列
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        try:
            # 编码查询图像
            if embedding_type == "CLIP" or (self.embedding_type == "CLIP"):
                # 使用CLIP自定义编码方法
                processed_image = self._preprocess_object_image(query_obj['image'])
                query_vector = self.encode_image(processed_image)
                
                # 在Chroma中搜索
                results = self.object_collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    include=['embeddings', 'documents', 'metadatas', 'distances']
                )
            elif embedding_type in known_embedding_functions:
                # 使用known_embedding_functions中的embedding
                processed_image = self._preprocess_object_image(query_obj['image'])
                query_vector = self.encode_image(processed_image)
                
                # 在Chroma中搜索
                results = self.object_collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    include=['embeddings', 'documents', 'metadatas', 'distances']
                )
            else:
                # 使用默认的文本搜索
                results = self.object_collection.query(
                    query_texts=[query_obj.get('content', '')],
                    n_results=top_k,
                    include=['embeddings', 'documents', 'metadatas', 'distances']
                )
            
            # 处理结果
            similar_objects = []
            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    similarity = 1 - distance  # 转换为相似度
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
            self.logger.error(f"相似度搜索失败: {e}")
            return []
    
    def deduplicate_objects(self, objects: List[Dict], threshold: float = None, 
                           text_weight: float = 0.3, image_weight: float = 0.5, 
                           bbox_weight: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        对象去重处理，使用优化的混合相似性检测
        优化策略：
        1. 使用更全面的候选对象集合
        2. 采用多阶段相似性检测
        3. 针对去重场景调整权重和阈值
        
        Args:
            objects: 待处理的对象列表
            threshold: 去重阈值
            text_weight: 文本相似性权重
            image_weight: 图像相似性权重
            bbox_weight: 边界框相似性权重
            
        Returns:
            (新对象列表, 重复对象列表)
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        new_objects = []
        duplicate_objects = []
        
        # 为每个对象生成hash以提高比较效率
        for obj in objects:
            if 'hash' not in obj or not obj['hash']:
                obj['hash'] = self._generate_object_hash(obj)
        
        for i, obj in enumerate(objects):
            # 首先进行快速hash比较
            found_duplicate = False
            
            # 阶段1a: 在已处理的新对象中查找完全相同的hash（批次内去重）
            for existing_obj in new_objects:
                if (obj.get('hash') and existing_obj.get('hash') and 
                    obj['hash'] == existing_obj['hash']):
                    duplicate_objects.append({
                        'original': obj,
                        'similar': existing_obj,
                        'similarity': 1.0,
                        'reason': 'identical_hash_batch'
                    })
                    print(f"Duplicate Obj (Hash-Batch): {obj.get('content', 'Unknown')} (Sim: 1.000)")
                    found_duplicate = True
                    break
            
            if found_duplicate:
                continue
                
            # 阶段1b: 在数据库中查找完全相同的hash（与已存储对象去重）
            if obj.get('hash'):
                try:
                    # 直接使用where条件查找相同hash的对象
                    hash_results = self.object_collection.get(
                        where={"hash": obj['hash']},
                        include=['metadatas', 'documents']
                    )
                    
                    # 如果找到相同hash的对象
                    if hash_results['ids'] and len(hash_results['ids']) > 0:
                        # 取第一个匹配的对象
                        matched_id = hash_results['ids'][0]
                        matched_metadata = hash_results['metadatas'][0]
                        matched_content = hash_results['documents'][0]
                        
                        duplicate_objects.append({
                            'original': obj,
                            'similar': {
                                'id': matched_id,
                                'content': matched_content,
                                'hash': matched_metadata['hash'],
                                'bbox': json.loads(matched_metadata['bbox']),
                                'area': matched_metadata['area'],
                                'center': json.loads(matched_metadata['center']),
                                'type': matched_metadata['type'],
                                'object_id': matched_metadata['object_id']
                            },
                            'similarity': 1.0,
                            'reason': 'identical_hash_db'
                        })
                        print(f"Duplicate Obj (Hash-DB): {obj.get('content', 'Unknown')} (Sim: 1.000)")
                        found_duplicate = True
                        
                except Exception as e:
                    self.logger.debug(f"数据库hash查询失败: {e}")
                    # 如果直接hash查询失败，回退到文本查询+hash比较的方法
                    try:
                        hash_results = self.object_collection.query(
                            query_texts=[obj.get('content', 'unknown')],
                            n_results=20,  # 获取更多结果以便hash比较
                            include=['metadatas', 'documents', 'distances']
                        )
                        
                        # 检查返回结果中是否有相同的hash
                        for j in range(len(hash_results['ids'][0])):
                            metadata = hash_results['metadatas'][0][j]
                            stored_hash = metadata.get('hash', '')
                            
                            if stored_hash and stored_hash == obj['hash']:
                                duplicate_objects.append({
                                    'original': obj,
                                    'similar': {
                                        'id': hash_results['ids'][0][j],
                                        'content': hash_results['documents'][0][j],
                                        'hash': stored_hash,
                                        'bbox': json.loads(metadata['bbox']),
                                        'area': metadata['area'],
                                        'center': json.loads(metadata['center']),
                                        'type': metadata['type'],
                                        'object_id': metadata['object_id']
                                    },
                                    'similarity': 1.0,
                                    'reason': 'identical_hash_db'
                                })
                                print(f"Duplicate Obj (Hash-DB): {obj.get('content', 'Unknown')} (Sim: 1.000)")
                                found_duplicate = True
                                break
                    except Exception as e2:
                        self.logger.debug(f"回退hash查询也失败: {e2}")
            
            if found_duplicate:
                continue
                
            # 阶段2: 使用混合查询进行更精确的相似性检测
            similar_objects = self.query_collection_mixed(
                obj, 
                top_k=3,  # 获取前3个最相似的对象进行比较
                text_weight=text_weight, 
                image_weight=image_weight, 
                bbox_weight=bbox_weight
            )

            if similar_objects:
                # 检查是否有超过阈值的相似对象
                best_match = similar_objects[0]
                if best_match['similarity'] >= threshold:
                    # 找到相似对象，标记为重复
                    duplicate_objects.append({
                        'original': obj,
                        'similar': best_match,
                        'similarity': best_match['similarity'],
                        'reason': 'mixed_similarity'
                    })
                    print(f"Duplicate Obj (Mixed): {obj.get('content', 'Unknown')} (Sim: {best_match['similarity']:.3f})")
                    found_duplicate = True
            
            if not found_duplicate:
                # 新对象
                new_objects.append(obj)
        
        return new_objects, duplicate_objects
    
    def update_object_with_vector_storage(self, objects: List[Dict], 
                                         text_weight: float = 0.3, image_weight: float = 0.5, 
                                         bbox_weight: float = 0.2, similarity_threshold: float = None) -> List[Dict]:
        """
        更新对象并存储到向量数据库，使用混合相似性检测
        
        Args:
            objects: 对象列表
            text_weight: 文本相似性权重
            image_weight: 图像相似性权重
            bbox_weight: 边界框相似性权重
            similarity_threshold: 相似性阈值
            
        Returns:
            处理后的对象列表
        """
        if not objects:
            return []
        
        # 使用混合相似性进行去重处理
        new_objects, duplicate_objects = self.deduplicate_objects(
            objects, 
            threshold=similarity_threshold, 
            text_weight=text_weight, 
            image_weight=image_weight, 
            bbox_weight=bbox_weight
        )
        
        # 存储新对象
        if new_objects:
            stored_ids = self.batch_store_objects(new_objects)
            for i, obj in enumerate(new_objects):
                if i < len(stored_ids):
                    obj['vector_id'] = stored_ids[i]
        
        # 处理重复对象
        for dup in duplicate_objects:
            original = dup['original']
            similar = dup['similar']
            
            # 更新原对象的ID为相似对象的ID
            original['id'] = similar.get('object_id', similar.get('id'))
            original['vector_id'] = similar.get('id')
            
            self.logger.debug(f"对象去重: {original.get('content', 'Unknown')} -> {similar.get('id')}(相似度: {similar.get('mixed_similarity', 0):.3f})")
        
        # 合并结果
        all_objects = new_objects + [dup['original'] for dup in duplicate_objects]
        
        self.logger.debug(f"对象处理完成: 新增 {len(new_objects)}, 去重 {len(duplicate_objects)}")
        return all_objects
    
    def query_collection_by_text(self, content_query: str, top_k: int = 10) -> List[Dict]:
        """
        根据内容搜索对象
        
        Args:
            content_query: 内容查询字符串
            top_k: 返回的最大结果数
            
        Returns:
            匹配的对象列表
        """
        try:
            # 判断逻辑与_init_chroma_db保持一致
            if not self.embedding_type or self.embedding_type == "":
                # 空字符串或None，使用默认文本查询
                results = self.object_collection.query(
                    query_texts=[content_query],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            elif self.embedding_type == "CLIP":
                # 使用CLIP对文本进行编码，确保与图像向量维度一致
                text_vector = self.encode_text(content_query)
                # 使用向量搜索而不是文档搜索
                results = self.object_collection.query(
                    query_embeddings=[text_vector.tolist()],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            elif self.embedding_type in known_embedding_functions:
                # 使用known_embedding_functions中的embedding进行文本查询
                results = self.object_collection.query(
                    query_texts=[content_query],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            else:
                # 不支持的embedding_type，使用默认文本查询
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
                    # 将距离转换为相似度：对于余弦距离，相似度 = 1 - distance/2
                    # 这样可以确保相似度在 [0, 1] 范围内，1表示最相似，0表示最不相似
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
            self.logger.error(f"内容搜索失败: {e}")
            return []

    def query_collection_by_image(self, query_image: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        根据图像搜索对象
        
        Args:
            query_image: 查询图像的numpy数组
            top_k: 返回的最大结果数
            
        Returns:
            匹配的对象列表
        """
        try:
            # 判断逻辑与_init_chroma_db保持一致
            if not self.embedding_type or self.embedding_type == "":
                # 默认embedding不支持图像查询
                self.logger.warning("默认embedding不支持图像查询")
                return []
            elif self.embedding_type == "CLIP":
                # 使用CLIP对图像进行编码
                image_vector = self.encode_image(query_image)
                # 使用向量搜索
                results = self.object_collection.query(
                    query_embeddings=[image_vector.tolist()],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances', 'embeddings']
                )
            elif self.embedding_type in known_embedding_functions:
                # 对于支持多模态的embedding函数（如open_clip），使用图像查询
                if self.embedding_type == "open_clip":
                    # OpenCLIPEmbeddingFunction支持图像查询
                    results = self.object_collection.query(
                        query_images=[query_image],
                        n_results=top_k,
                        include=['documents', 'metadatas', 'distances', 'embeddings']
                    )
                else:
                    # 其他embedding函数可能不支持图像查询
                    self.logger.warning(f"Embedding函数 '{self.embedding_type}' 可能不支持图像查询")
                    return []
            else:
                # 不支持的embedding_type
                self.logger.warning(f"不支持的embedding_type '{self.embedding_type}' 进行图像查询")
                return []

            objects = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    embedding = results['embeddings'][0][i] if results.get('embeddings') else None
                    # 将距离转换为相似度：对于余弦距离，相似度 = 1 - distance/2
                    # 这样可以确保相似度在 [0, 1] 范围内，1表示最相似，0表示最不相似
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
            self.logger.error(f"图像搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        获取向量数据库统计信息
        
        Returns:
            统计信息字典
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
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def query_collection_mixed(self, query_obj: Dict, top_k: int = 10, 
                              text_weight: float = 0.4, image_weight: float = 0.4, 
                              bbox_weight: float = 0.2, bbox_scale: float = 1000.0) -> List[Dict]:
        """
        混合查询方法，结合文本相似度、图像相似度和bbox位置距离
        优化版本：获取更全面的候选对象集合，避免遗漏相似对象
        
        Args:
            query_obj: 查询对象，包含 content, cropped_image, bbox 等字段
            top_k: 返回结果数量
            text_weight: 文本相似度权重
            image_weight: 图像相似度权重  
            bbox_weight: bbox距离权重
            bbox_scale: bbox距离归一化缩放因子
            
        Returns:
            综合相似度排序的对象列表
        """
        try:
            # 为query_obj生成hash（如果没有的话）
            if 'hash' not in query_obj or not query_obj['hash']:
                query_obj['hash'] = self._generate_object_hash(query_obj)
            
            # 获取数据库中的总对象数量
            collection_count = self.object_collection.count()
            if collection_count == 0:
                self.logger.warning("向量数据库为空，无法获取候选对象")
                return []
            
            # 动态调整候选对象数量，确保不遗漏相似对象
            # 对于去重场景，我们需要更全面的候选集合
            if top_k == 1:  # 去重场景，获取更多候选对象
                candidate_k = min(collection_count, max(500, collection_count // 2))
            else:  # 常规查询场景
                candidate_k = min(collection_count, max(top_k * 10, 100))
            
            # 获取候选对象集合 - 使用多种策略确保覆盖面
            all_candidates = {}
            
            # 策略1: 文本查询获取候选对象
            if 'content' in query_obj and query_obj['content']:
                text_results = self.query_collection_by_text(query_obj['content'], candidate_k)
                for result in text_results:
                    all_candidates[result['id']] = result
            
            # 策略2: 图像查询获取候选对象（与文本查询并行，不是补充）
            if 'image' in query_obj and query_obj['image'] is not None:
                image_results = self.query_collection_by_image(query_obj['image'], candidate_k)
                for result in image_results:
                    if result['id'] not in all_candidates:
                        all_candidates[result['id']] = result
            
            # 策略3: 如果候选对象仍然不足，获取所有对象进行全面比较
            if len(all_candidates) < candidate_k:
                try:
                    remaining_needed = candidate_k - len(all_candidates)
                    results = self.object_collection.query(
                        query_texts=["*"],  # 通用查询
                        n_results=min(remaining_needed, collection_count),
                        include=['metadatas', 'documents', 'distances', 'embeddings']
                    )
                    
                    for i in range(len(results['ids'][0])):
                        obj_id = results['ids'][0][i]
                        if obj_id not in all_candidates:  # 避免重复
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
                    self.logger.warning(f"获取补充候选对象失败: {e}")
            
            # 为所有候选对象计算完整的文本和图像相似度
            query_text_embedding = None
            
            # 准备查询嵌入
            if 'content' in query_obj and query_obj['content']:
                query_text_embedding = self.encode_text(query_obj['content'])
            
            # 为每个候选对象计算相似度
            for obj_id, candidate in all_candidates.items():
                text_similarity = 0.0
                image_similarity = 0.0
                
                # 计算文本相似度
                if query_text_embedding is not None and candidate.get('content'):
                    try:
                        candidate_text_embedding = self.encode_text(candidate['content'])
                        # 计算余弦相似度
                        text_cosine_sim = np.dot(query_text_embedding, candidate_text_embedding) / (
                            np.linalg.norm(query_text_embedding) * np.linalg.norm(candidate_text_embedding)
                        )
                        text_similarity = max(0, (text_cosine_sim + 1) / 2)  # 转换到[0,1]范围
                    except Exception as e:
                        self.logger.debug(f"计算文本相似度失败: {e}")
                
                # 优化的图像相似度计算 - 使用hash比较和更精确的阈值
                if 'hash' in query_obj and query_obj['hash'] and 'hash' in candidate and candidate['hash']:
                    try:
                        query_hash = query_obj['hash']
                        candidate_hash = candidate['hash']
                        
                        # 如果hash完全相同，相似度为1.0
                        if query_hash == candidate_hash:
                            image_similarity = 1.0
                        else:
                            # 计算hash的汉明距离相似度
                            try:
                                query_bin = bin(int(query_hash, 16))[2:].zfill(64)  # 转为64位二进制
                                candidate_bin = bin(int(candidate_hash, 16))[2:].zfill(64)
                                
                                # 计算汉明距离
                                hamming_distance = sum(c1 != c2 for c1, c2 in zip(query_bin, candidate_bin))
                                # 转换为相似度，使用更敏感的阈值
                                max_distance = 64.0
                                # 对于去重场景，我们需要更严格的相似度判断
                                if hamming_distance <= 8:  # 允许少量差异
                                    image_similarity = 1.0 - (hamming_distance / max_distance) * 0.5
                                else:
                                    image_similarity = max(0, 1 - hamming_distance / max_distance)
                            except ValueError:
                                # 如果hash格式不正确，使用字符串相似度
                                common_chars = sum(c1 == c2 for c1, c2 in zip(query_hash, candidate_hash))
                                image_similarity = common_chars / max(len(query_hash), len(candidate_hash))
                    except Exception as e:
                        self.logger.debug(f"计算图像hash相似度失败: {e}")
                        image_similarity = 0.0
                else:
                    image_similarity = 0.0
                
                candidate['text_similarity'] = text_similarity
                candidate['image_similarity'] = image_similarity
            
            # 计算bbox距离相似度
            query_bbox = query_obj.get('bbox', None)
            if query_bbox:
                query_center = query_obj.get('center', None)
                
                for obj_id, candidate in all_candidates.items():
                    candidate_center = candidate.get('center', None)
                    
                    # 计算欧几里得距离
                    distance = np.sqrt((query_center[0] - candidate_center[0])**2 + 
                                     (query_center[1] - candidate_center[1])**2)
                    
                    # 将距离转换为相似度 (距离越小，相似度越高)
                    bbox_similarity = max(0, 1 - distance / bbox_scale)
                    candidate['bbox_similarity'] = bbox_similarity
            else:
                # 如果没有bbox信息，bbox相似度设为0.5（中性值）
                for candidate in all_candidates.values():
                    candidate['bbox_similarity'] = 0.5
            
            # 计算综合相似度
            for candidate in all_candidates.values():
                # 归一化权重
                total_weight = text_weight + image_weight + bbox_weight
                norm_text_weight = text_weight / total_weight
                norm_image_weight = image_weight / total_weight
                norm_bbox_weight = bbox_weight / total_weight
                
                # 计算加权综合相似度
                mixed_similarity = (candidate['text_similarity'] * norm_text_weight + 
                                  candidate['image_similarity'] * norm_image_weight + 
                                  candidate['bbox_similarity'] * norm_bbox_weight)
                
                candidate['mixed_similarity'] = mixed_similarity
                candidate['similarity'] = mixed_similarity  # 更新主相似度字段
            
            # 按综合相似度排序并返回top_k结果
            sorted_results = sorted(all_candidates.values(), 
                                  key=lambda x: x['mixed_similarity'], reverse=True)
            
            return sorted_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"混合查询失败: {e}")
            return []
    
    def clear_database(self):
        """清空向量数据库"""
        try:
            # 删除collection
            self.client.delete_collection("ui_objects")
            
            # 重新创建
            # self.object_collection = self.client.create_collection(
            #     name="ui_objects",
            #     metadata={"hnsw:space": "cosine"}
            # )
            
            self.logger.info("向量数据库已清空")
            
        except Exception as e:
            self.logger.error(f"清空数据库失败: {e}")