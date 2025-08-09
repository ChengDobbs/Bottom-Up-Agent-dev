import chromadb
from chromadb.config import Settings
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

        self.similarity_threshold = config.get('vector_memory', {}).get('similarity_threshold', 0.8)
        self.vector_dim = 512  # CLIP ViT-B/32 输出维度
        
        # 初始化Chroma客户端
        self._init_chroma_db()
        
    def _init_chroma_db(self):
        """初始化Chroma向量数据库"""
        try:
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(path=self.vector_db_path)
            
            # 创建或获取collection
            self.object_collection = self.client.get_or_create_collection(
                name="ui_objects",
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
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
            # 预处理图像
            processed_image = self._preprocess_object_image(obj['image'])
            
            # 编码图像
            image_vector = self.encode_image(processed_image)
            
            # 生成唯一ID
            object_hash = self._generate_object_hash(obj)
            object_id = f"obj_{obj.get('id', 'new')}_{object_hash[:8]}"
            
            # 准备元数据
            metadata = {
                'object_id': str(obj.get('id', '')),
                'bbox': json.dumps(obj['bbox']),
                'area': obj['area'],
                'hash': obj.get('hash', ''),
                'center': json.dumps(obj.get('center', [0, 0])),
                'type': 'icon' if not obj.get('content') else 'text',
                'timestamp': datetime.now().isoformat(),
            }
            
            # add embedding, content, metadatas to Chroma
            self.object_collection.add(
                embeddings=[image_vector.tolist()],
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
                # 预处理和编码
                processed_image = self._preprocess_object_image(obj['image'])
                image_vector = self.encode_image(processed_image)
                
                # 生成ID和元数据
                object_hash = self._generate_object_hash(obj)
                object_id = f"obj_{obj.get('id', 'new')}_{object_hash[:8]}"
                
                metadata = {
                    'object_id': str(obj.get('id', '')),
                    'bbox': json.dumps(obj['bbox']),
                    'area': obj['area'],
                    'hash': obj.get('hash', ''),
                    'center': json.dumps(obj.get('center', [0, 0])),
                    'type': 'icon' if not obj.get('content') else 'text',
                    'timestamp': datetime.now().isoformat(),
                }
                
                embeddings.append(image_vector.tolist())
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
    
    def find_similar_objects(self, query_obj: Dict, threshold: float = None, top_k: int = 5) -> List[Dict]:
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
            processed_image = self._preprocess_object_image(query_obj['image'])
            query_vector = self.encode_image(processed_image)
            
            # 在Chroma中搜索
            results = self.object_collection.query(
                query_embeddings=[query_vector.tolist()],
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
    
    def deduplicate_objects(self, objects: List[Dict], threshold: float = None) -> Tuple[List[Dict], List[Dict]]:
        """
        对象去重处理
        
        Args:
            objects: 待处理的对象列表
            threshold: 去重阈值
            
        Returns:
            (新对象列表, 重复对象列表)
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        new_objects = []
        duplicate_objects = []
        
        for obj in objects:
            similar_objects = self.find_similar_objects(obj, threshold=threshold, top_k=1)
            
            if similar_objects:
                # 找到相似对象，标记为重复
                duplicate_objects.append({
                    'original': obj,
                    'similar': similar_objects[0]
                })
            else:
                # 新对象
                new_objects.append(obj)
        
        return new_objects, duplicate_objects
    
    def update_object_with_vector_storage(self, objects: List[Dict]) -> List[Dict]:
        """
        更新对象并存储到向量数据库
        
        Args:
            objects: 对象列表
            
        Returns:
            处理后的对象列表
        """
        if not objects:
            return []
        
        # 去重处理
        new_objects, duplicate_objects = self.deduplicate_objects(objects)
        
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
            original['id'] = similar['object_id']
            original['vector_id'] = similar['id']
            
            self.logger.info(f"对象去重: {original.get('content', 'Unknown')} -> {similar['id']}")
        
        # 合并结果
        all_objects = new_objects + [dup['original'] for dup in duplicate_objects]
        
        self.logger.info(f"对象处理完成: 新增 {len(new_objects)}, 去重 {len(duplicate_objects)}")
        return all_objects
    
    def search_objects_by_content(self, content_query: str, top_k: int = 10) -> List[Dict]:
        """
        根据内容搜索对象
        
        Args:
            content_query: 内容查询字符串
            top_k: 返回的最大结果数
            
        Returns:
            匹配的对象列表
        """
        try:
            # 使用CLIP对文本进行编码，确保与图像向量维度一致
            text_vector = self.encode_text(content_query)
            
            # 使用向量搜索而不是文档搜索
            results = self.object_collection.query(
                query_embeddings=[text_vector.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            objects = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    objects.append({
                        'id': results['ids'][0][i],
                        'content': doc,
                        'bbox': json.loads(metadata['bbox']),
                        'area': metadata['area'],
                        'center': json.loads(metadata['center']),
                        'type': metadata['type'],
                        'object_id': metadata['object_id'],
                        'distance': results['distances'][0][i]
                    })
            
            return objects
            
        except Exception as e:
            self.logger.error(f"内容搜索失败: {e}")
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
    
    def clear_database(self):
        """清空向量数据库"""
        try:
            # 删除collection
            self.client.delete_collection("ui_objects")
            
            # 重新创建
            self.object_collection = self.client.create_collection(
                name="ui_objects",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info("向量数据库已清空")
            
        except Exception as e:
            self.logger.error(f"清空数据库失败: {e}")