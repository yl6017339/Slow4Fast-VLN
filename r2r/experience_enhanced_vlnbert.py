import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from .reflection_memory_bank import ReflectionMemoryBank, ReflectionExperience
from models.model import VLNBert, Critic
from models.vilmodel import GraphLXRTXLayer

class ExperienceEncoder(nn.Module):
    """经验编码器 - 将抽象经验转换为可用的嵌入"""
    
    def __init__(self, hidden_size: int = 768, experience_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.experience_dim = experience_dim
        
        # 场景类型编码器
        self.scene_type_encoder = nn.Embedding(20, experience_dim)  # 假设20种场景类型
        
        # 空间上下文编码器
        self.spatial_context_encoder = nn.Embedding(50, experience_dim)  # 假设50种空间上下文
        
        # 行为模式编码器
        self.action_pattern_encoder = nn.Embedding(30, experience_dim)  # 假设30种行为模式
        
        # 文本规则编码器（使用预训练的BERT）
        self.rule_encoder = nn.Linear(hidden_size, experience_dim)
    
        # 经验融合层
        self.experience_fusion = nn.Sequential(
            nn.Linear(experience_dim * 4, experience_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(experience_dim * 2, experience_dim)
        )
        
        # 经验权重计算
        self.experience_weight_net = nn.Sequential(
            nn.Linear(experience_dim, experience_dim // 2),
            nn.ReLU(),
            nn.Linear(experience_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, experiences: List[ReflectionExperience]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将经验列表编码为嵌入向量和权重
        Args:
            experiences: 反思经验列表
        Returns:
            experience_embeds: 经验嵌入 (num_experiences, experience_dim)
            experience_weights: 经验权重 (num_experiences, 1)
        """
        if not experiences:
            # 返回空的经验嵌入
            return torch.zeros(1, self.experience_dim), torch.zeros(1, 1)
        
        batch_size = len(experiences)
        device = next(self.parameters()).device
        
        # 编码各种经验组件
        scene_type_ids = torch.tensor([self._get_scene_type_id(exp.scene_type) for exp in experiences], device=device)
        spatial_context_ids = torch.tensor([self._get_spatial_context_id(exp.spatial_context) for exp in experiences], device=device)
        action_pattern_ids = torch.tensor([self._get_action_pattern_id(exp.action_pattern) for exp in experiences], device=device)
        
        # 获取嵌入
        scene_embeds = self.scene_type_encoder(scene_type_ids)  # (batch_size, experience_dim)
        spatial_embeds = self.spatial_context_encoder(spatial_context_ids)  # (batch_size, experience_dim)
        action_embeds = self.action_pattern_encoder(action_pattern_ids)  # (batch_size, experience_dim)
        
        # 编码文本规则（这里简化处理，实际应该使用BERT）
        rule_embeds = torch.zeros(batch_size, self.experience_dim, device=device)
        for i, exp in enumerate(experiences):
            # 简单的文本编码（实际应该使用预训练的文本编码器）
            rule_text = f"{exp.spatial_rule} {exp.navigation_strategy}"
            rule_embeds[i] = self._simple_text_encode(rule_text)
        
        # 融合所有经验组件
        combined_embeds = torch.cat([scene_embeds, spatial_embeds, action_embeds, rule_embeds], dim=-1)
        experience_embeds = self.experience_fusion(combined_embeds)  # (batch_size, experience_dim)
        
        # 计算经验权重（基于成功率和置信度）
        experience_weights = self.experience_weight_net(experience_embeds)  # (batch_size, 1)
        
        # 根据实际成功率和置信度调整权重
        for i, exp in enumerate(experiences):
            success_weight = exp.success_rate
            confidence_weight = exp.decision_confidence
            frequency_weight = min(exp.frequency / 10.0, 1.0)  # 频次权重，最多10次
            experience_weights[i] *= (success_weight * confidence_weight * frequency_weight)
        
        return experience_embeds, experience_weights

    def _get_scene_type_id(self, scene_type: str) -> int:
        """将场景类型转换为ID"""
        scene_type_map = {
            'hallway': 0, 'room': 1, 'corridor': 2, 'kitchen': 3, 'bathroom': 4,
            'bedroom': 5, 'living_room': 6, 'office': 7, 'unknown': 8
        }
        return scene_type_map.get(scene_type, 8)

    def _get_spatial_context_id(self, spatial_context: str) -> int:
        """将空间上下文转换为ID"""
        spatial_context_map = {
            'near_door': 0, 'corner': 1, 'center': 2, 'wall': 3, 'entrance': 4,
            'exit': 5, 'junction': 6, 'dead_end': 7, 'unknown': 8
        }
        return spatial_context_map.get(spatial_context, 8)

    def _get_action_pattern_id(self, action_pattern: str) -> int:
        """将行为模式转换为ID"""
        action_pattern_map = {
            'exploration': 0, 'goal_directed': 1, 'forward_preference': 2,
            'turn_preference': 3, 'stop_preference': 4, 'unknown': 5
        }
        return action_pattern_map.get(action_pattern, 5)

    def _simple_text_encode(self, text: str) -> torch.Tensor:
        """简单的文本编码（实际应该使用BERT）"""
        # 这里使用简单的字符级编码作为占位符
        char_embeds = torch.zeros(self.experience_dim)
        for i, char in enumerate(text[:self.experience_dim]):
            char_embeds[i] = ord(char) / 128.0  # 归一化
        return char_embeds

class ExperienceEnhancedVLNBert(VLNBert):
    """增强的VLNBert模型 - 集成反思经验"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # 经验编码器
        self.experience_encoder = ExperienceEncoder(
            hidden_size=args.hidden_size if hasattr(args, 'hidden_size') else 768,
            experience_dim=args.experience_dim if hasattr(args, 'experience_dim') else 256
        )
        
        # 经验注意力机制
        self.experience_attention = nn.MultiheadAttention(
            embed_dim=768,  # VLNBert的隐藏维度
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 经验融合层
        self.experience_fusion = nn.Sequential(
            nn.Linear(768 + 256, 768),  # VLNBert特征 + 经验特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768)
        )
        
        # 经验门控机制
        self.experience_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid()
        )

    def forward_with_experience(self, mode: str, batch: Dict[str, Any], 
                              relevant_experiences: List[ReflectionExperience]) -> Any:
        """
        带经验增强的前向传播
        Args:
            mode: 模式 ('language', 'panorama', 'navigation')
            batch: 输入批次
            relevant_experiences: 相关经验列表
        """
        # 获取基础VLNBert输出
        if mode == 'language':
            txt_embeds = super().forward(mode, batch)
            return txt_embeds
        elif mode == 'panorama':
            pano_embeds, pano_masks = super().forward(mode, batch)
            return pano_embeds, pano_masks
        elif mode == 'navigation':
            # 导航模式需要经验增强
            return self._navigation_with_experience(batch, relevant_experiences)
        else:
            raise NotImplementedError(f'wrong mode: {mode}')

    def _navigation_with_experience(self, batch: Dict[str, Any], 
                                  relevant_experiences: List[ReflectionExperience]) -> Dict[str, torch.Tensor]:
        """带经验增强的导航前向传播"""
        # 获取基础导航输出
        base_outputs = super().forward('navigation', batch)
        
        if not relevant_experiences:
            return base_outputs
        
        # 编码相关经验
        experience_embeds, experience_weights = self.experience_encoder(relevant_experiences)
        
        # 获取视觉特征（假设batch中有view_img_fts）
        if 'view_img_fts' in batch:
            visual_features = batch['view_img_fts']  # (batch_size, seq_len, hidden_size)
            
            # 经验注意力：让视觉特征关注相关经验
            attended_features, attention_weights = self.experience_attention(
                query=visual_features,
                key=experience_embeds.unsqueeze(0).expand(visual_features.size(0), -1, -1),
                value=experience_embeds.unsqueeze(0).expand(visual_features.size(0), -1, -1),
                key_padding_mask=None
            )
            
            # 融合视觉特征和经验特征
            fused_features = self.experience_fusion(
                torch.cat([visual_features, attended_features], dim=-1)
            )
            
            # 经验门控：控制经验的影响程度
            experience_gate = self.experience_gate(visual_features)
            enhanced_features = fused_features * experience_gate + visual_features * (1 - experience_gate)
            
            # 更新batch中的视觉特征
            batch['view_img_fts'] = enhanced_features
        
        # 重新进行导航前向传播
        enhanced_outputs = super().forward('navigation', batch)
        
        # 添加经验信息到输出
        enhanced_outputs['experience_embeds'] = experience_embeds
        enhanced_outputs['experience_weights'] = experience_weights
        enhanced_outputs['attention_weights'] = attention_weights if 'attention_weights' in locals() else None
        
        return enhanced_outputs

class ExperienceEnhancedAgent:
    """经验增强的导航智能体"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化记忆银行
        self.reflection_memory_bank = ReflectionMemoryBank(max_experiences=10000)
        
        # 初始化增强的VLNBert模型
        self.vln_bert = ExperienceEnhancedVLNBert(args)
        self.critic = Critic(args)
        
        # 经验检索参数
        self.experience_retrieval_threshold = 0.7
        self.max_relevant_experiences = 5
        
        # 经验应用策略
        self.experience_application_strategy = 'adaptive'  # 'adaptive', 'conservative', 'aggressive'

    def get_relevant_experiences(self, current_context: Dict[str, Any]) -> List[ReflectionExperience]:
        """获取相关经验"""
        # 从当前上下文推断场景信息
        scene_type = self._infer_scene_type(current_context)
        spatial_context = self._infer_spatial_context(current_context)
        action_pattern = self._infer_action_pattern(current_context)
        
        # 检索相关经验
        relevant_experiences = self.reflection_memory_bank.retrieve_relevant_experiences(
            scene_type, spatial_context, action_pattern
        )
        
        # 过滤低质量经验
        filtered_experiences = [
            exp for exp in relevant_experiences
            if exp.success_rate > 0.3 and exp.decision_confidence > 0.5
        ]
        
        return filtered_experiences[:self.max_relevant_experiences]

    def _infer_scene_type(self, context: Dict[str, Any]) -> str:
        """从上下文推断场景类型"""
        # 这里可以根据观察信息、位置信息等推断场景类型
        # 简化实现：根据指令关键词推断
        instruction = context.get('instruction', '').lower()
        
        if 'kitchen' in instruction:
            return 'kitchen'
        elif 'bathroom' in instruction:
            return 'bathroom'
        elif 'bedroom' in instruction:
            return 'bedroom'
        elif 'hallway' in instruction or 'corridor' in instruction:
            return 'hallway'
        else:
            return 'room'

    def _infer_spatial_context(self, context: Dict[str, Any]) -> str:
        """从上下文推断空间上下文"""
        # 根据候选动作和位置信息推断
        candidates = context.get('candidates', [])
        
        if len(candidates) == 1:
            return 'dead_end'
        elif len(candidates) > 3:
            return 'junction'
        else:
            return 'corridor'

    def _infer_action_pattern(self, context: Dict[str, Any]) -> str:
        """从上下文推断行为模式"""
        # 根据历史动作推断行为模式
        # 简化实现：默认为探索模式
        return 'exploration'

    def forward_with_experience(self, mode: str, batch: Dict[str, Any], 
                              current_context: Dict[str, Any] = None) -> Any:
        """带经验增强的前向传播"""
        # 获取相关经验
        relevant_experiences = []
        if current_context:
            relevant_experiences = self.get_relevant_experiences(current_context)
        
        # 使用增强的VLNBert进行前向传播
        return self.vln_bert.forward_with_experience(mode, batch, relevant_experiences)

    def add_experience_from_context(self, context_data: Dict[str, Any]) -> bool:
        """从上下文数据添加经验"""
        return self.reflection_memory_bank.add_experience(context_data)

    def save_memory_bank(self, filepath: str):
        """保存记忆银行"""
        self.reflection_memory_bank.save_memory_bank(filepath)

    def load_memory_bank(self, filepath: str):
        """加载记忆银行"""
        self.reflection_memory_bank.load_memory_bank(filepath)
