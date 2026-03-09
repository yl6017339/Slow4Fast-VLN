import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReflectionExperience:
    """反思经验的抽象表示"""
    # 场景上下文
    scene_type: str  # "hallway", "room", "corridor"等
    spatial_context: str  # "near_door", "corner", "center"等
    
    # 行为模式
    action_pattern: str  # "forward_preference", "exploration", "goal_directed"等
    decision_confidence: float  # 决策置信度
    
    # 抽象知识
    spatial_rule: str  # "doors_lead_to_rooms", "hallways_connect_spaces"等
    navigation_strategy: str  # "explore_first", "direct_path"等
    place_relationship: str  # "place_3_to_place_4", "adjacent_rooms"等
    
    # 视觉上下文
    visual_context: str  # "door_visible", "corridor_view", "room_interior"等
    visual_landmarks: str  # "door_frame", "window", "furniture"等
    visual_orientation: str  # "facing_door", "side_view", "back_view"等
    
    # 元信息
    success_rate: float  # 该经验的成功率
    frequency: int  # 出现频次
    last_updated: int  # 最后更新时间戳

class LLMReflectionModule:
    """LLM反思模块 - 分析导航行为并提取抽象经验"""
    
    def __init__(self, llm_model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.llm_model_name = llm_model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # 反思提示模板
        self.reflection_prompt_template = """
你是一个导航专家，需要分析智能体的导航行为并提取抽象经验。

导航上下文：
- 指令：{instruction}
- 当前位置：{current_vp} (place: {current_place})
- 观察：{observation}
- 可见邻居：{visible_neighbors}
- 停止概率：{stop_prob}
- 采取动作：{action_taken}
- 成功状态：{success}
- 轨迹信息：{trajectory_info}
- 性能指标：{metrics}
- 视觉信息：{visual_descriptions}

请从以下维度进行反思分析：

1. 场景识别：
   - 当前处于什么类型的空间？（房间、走廊、大厅等）
   - place信息反映了什么空间结构？
   - 空间的空间特征是什么？（门的位置、通道方向等）

2. 空间关系分析：
   - 相邻节点的place关系如何？
   - 相对朝向和距离反映了什么空间布局？
   - 这种空间关系是否常见？

3. 视觉上下文分析：
   - 当前视角能看到什么关键视觉元素？（门、窗、家具等）
   - 视觉地标如何影响导航决策？
   - 视角朝向与目标方向的关系如何？

4. 行为分析：
   - 智能体的决策是否合理？
   - 体现了什么导航策略？
   - 决策的置信度如何？

5. 经验提取：
   - 从这次导航中能学到什么空间规则？
   - 什么导航策略在这种情况下有效？
   - 视觉信息如何指导导航决策？
   - 如何改进未来的导航决策？

请以JSON格式返回分析结果：
{{
    "scene_type": "场景类型",
    "spatial_context": "空间上下文描述",
    "action_pattern": "行为模式",
    "decision_confidence": 0.8,
    "spatial_rule": "提取的空间规则",
    "navigation_strategy": "导航策略",
    "place_relationship": "place关系描述",
    "visual_context": "视觉上下文描述",
    "visual_landmarks": "关键视觉地标",
    "visual_orientation": "视角朝向描述",
    "improvement_suggestion": "改进建议"
}}
"""

    def reflect_on_episode(self, context_data: Dict[str, Any]) -> Optional[ReflectionExperience]:
        """对单个导航片段进行反思"""
        try:
            # 构建反思提示
            prompt = self._build_reflection_prompt(context_data)
            
            # 调用LLM进行反思（这里需要实际的LLM API调用）
            reflection_result = self._call_llm_api(prompt)
            
            if reflection_result:
                return self._parse_reflection_result(reflection_result, context_data)
            
        except Exception as e:
            logger.error(f"反思过程出错: {e}")
        
        return None

    def _build_reflection_prompt(self, context_data: Dict[str, Any]) -> str:
        """构建反思提示"""
        # 提取关键信息
        instruction = context_data.get('instruction', '')
        steps = context_data.get('steps', [])
        trajectory_vpids = context_data.get('trajectory_vpids', [])
        metrics = context_data.get('metrics', {})
        details = context_data.get('details', {})
        
        if not steps:
            return ""
        
        # 取最后一个步骤进行分析
        last_step = steps[-1]
        current_vp = last_step.get('cur_vp', '')
        current_place = last_step.get('place', '')
        candidates = last_step.get('candidates', [])
        
        # 构建观察描述
        observation = f"当前位置: {current_vp} (place: {current_place})"
        
        # 构建邻居描述（包含place信息）
        visible_neighbors = []
        for cand in candidates:
            if not cand.get('is_current_vp', False):
                vp = cand.get('vp', '')
                place = cand.get('place', '')
                dist = cand.get('dist', 0)
                direction = cand.get('direction_text', '')
                rel_heading = cand.get('rel_heading', 0)
                visible_neighbors.append(f"'{vp}'(place: {place}, 距离{dist:.2f}m, {direction}, 朝向{rel_heading:.2f}rad)")
        
        neighbors_text = ", ".join(visible_neighbors) if visible_neighbors else "无可见邻居"
        
        # 获取停止概率（从details中获取）
        stop_prob = details.get(current_vp, {}).get('stop_prob', 0.0)
        
        # 模拟采取的动作（实际应该从轨迹中获取）
        action_taken = "stop" if stop_prob > 0.5 else "move"
        
        # 获取成功状态
        success = context_data.get('success', 0)
        
        # 构建轨迹信息
        trajectory_info = f"轨迹: {' -> '.join(trajectory_vpids)}" if trajectory_vpids else "轨迹: 无"
        
        # 构建性能指标信息
        metrics_text = f"SPL: {metrics.get('spl', 0):.2f}, 导航误差: {metrics.get('nav_error', 0):.2f}, 动作步数: {metrics.get('action_steps', 0)}"
        
        # 构建视觉信息描述
        visual_descriptions = self._extract_visual_descriptions(context_data)
        
        return self.reflection_prompt_template.format(
            instruction=instruction,
            current_vp=current_vp,
            current_place=current_place,
            observation=observation,
            visible_neighbors=neighbors_text,
            stop_prob=stop_prob,
            action_taken=action_taken,
            success=success,
            trajectory_info=trajectory_info,
            metrics=metrics_text,
            visual_descriptions=visual_descriptions
        )

    def _extract_visual_descriptions(self, context_data: Dict[str, Any]) -> str:
        """提取视觉信息描述"""
        visual_descriptions = []
        
        # 从context_data中提取视觉信息
        # 这里假设视觉信息已经转换为文本描述并存储在context_data中
        visual_info = context_data.get('visual_descriptions', {})
        
        if visual_info:
            # 如果有视觉信息，构建描述
            for vp_id, description in visual_info.items():
                visual_descriptions.append(f"视点 {vp_id}: {description}")
        else:
            # 如果没有视觉信息，基于place和空间关系推断
            steps = context_data.get('steps', [])
            if steps:
                last_step = steps[-1]
                current_place = last_step.get('place', '')
                candidates = last_step.get('candidates', [])
                
                # 基于place和邻居信息推断视觉上下文
                visual_context = self._infer_visual_context(current_place, candidates)
                visual_descriptions.append(f"推断的视觉上下文: {visual_context}")
        
        return "; ".join(visual_descriptions) if visual_descriptions else "无视觉信息"

    def _infer_visual_context(self, current_place: str, candidates: List[Dict]) -> str:
        """基于place和邻居信息推断视觉上下文"""
        # 分析邻居的place分布
        neighbor_places = [cand.get('place', '') for cand in candidates if not cand.get('is_current_vp', False)]
        
        # 基于place模式推断视觉上下文
        if len(neighbor_places) > 3:
            return "多方向通道或开放空间"
        elif any(abs(cand.get('rel_heading', 0)) < 0.5 for cand in candidates if not cand.get('is_current_vp', False)):
            return "前方有通道或门"
        elif any(abs(cand.get('rel_heading', 0) - 1.57) < 0.5 for cand in candidates if not cand.get('is_current_vp', False)):
            return "左侧有通道或门"
        elif any(abs(cand.get('rel_heading', 0) + 1.57) < 0.5 for cand in candidates if not cand.get('is_current_vp', False)):
            return "右侧有通道或门"
        else:
            return "转角或特殊空间布局"

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API进行反思"""
        # 这里需要实际的LLM API调用实现
        # 可以使用OpenAI API、本地LLM或其他LLM服务
        
        # 模拟LLM响应（实际实现时需要替换）
        mock_response = {
            "scene_type": "hallway",
            "spatial_context": "near_door",
            "action_pattern": "exploration",
            "decision_confidence": 0.7,
            "spatial_rule": "doors_lead_to_rooms",
            "navigation_strategy": "explore_first",
            "improvement_suggestion": "在走廊中应该优先探索可见的门"
        }
        
        return json.dumps(mock_response, ensure_ascii=False)

    def _parse_reflection_result(self, llm_response: str, context_data: Dict[str, Any]) -> ReflectionExperience:
        """解析LLM反思结果"""
        try:
            result = json.loads(llm_response)
            
            return ReflectionExperience(
                scene_type=result.get('scene_type', 'unknown'),
                spatial_context=result.get('spatial_context', 'unknown'),
                action_pattern=result.get('action_pattern', 'unknown'),
                decision_confidence=result.get('decision_confidence', 0.5),
                spatial_rule=result.get('spatial_rule', ''),
                navigation_strategy=result.get('navigation_strategy', ''),
                place_relationship=result.get('place_relationship', ''),
                visual_context=result.get('visual_context', 'unknown'),
                visual_landmarks=result.get('visual_landmarks', ''),
                visual_orientation=result.get('visual_orientation', ''),
                success_rate=1.0 if context_data.get('success', 0) else 0.0,
                frequency=1,
                last_updated=int(time.time())
            )
        except Exception as e:
            logger.error(f"解析反思结果失败: {e}")
            return None

class ReflectionMemoryBank:
    """具备反思能力的记忆银行"""
    
    def __init__(self, max_experiences: int = 10000):
        self.max_experiences = max_experiences
        self.experiences: List[ReflectionExperience] = []
        
        # 经验索引
        self.scene_type_index: Dict[str, List[int]] = defaultdict(list)
        self.spatial_context_index: Dict[str, List[int]] = defaultdict(list)
        self.action_pattern_index: Dict[str, List[int]] = defaultdict(list)
        
        # LLM反思模块
        self.reflection_module = LLMReflectionModule()
        
        # 经验融合权重
        self.fusion_weights = {
            'scene_type': 0.2,
            'spatial_context': 0.25,
            'action_pattern': 0.2,
            'place_relationship': 0.15,
            'visual_context': 0.1,
            'visual_landmarks': 0.05,
            'visual_orientation': 0.05
        }

    def add_experience(self, context_data: Dict[str, Any]) -> bool:
        """添加新的反思经验"""
        try:
            # 进行LLM反思
            experience = self.reflection_module.reflect_on_episode(context_data)
            
            if experience:
                # 检查是否已存在相似经验
                similar_exp_idx = self._find_similar_experience(experience)
                
                if similar_exp_idx is not None:
                    # 更新现有经验
                    self._update_existing_experience(similar_exp_idx, experience)
                else:
                    # 添加新经验
                    self._add_new_experience(experience)
                
                return True
                
        except Exception as e:
            logger.error(f"添加经验失败: {e}")
        
        return False

    def _find_similar_experience(self, new_exp: ReflectionExperience) -> Optional[int]:
        """查找相似经验"""
        for i, exp in enumerate(self.experiences):
            # 计算相似度
            similarity = self._calculate_similarity(new_exp, exp)
            if similarity > 0.8:  # 相似度阈值
                return i
        return None

    def _calculate_similarity(self, exp1: ReflectionExperience, exp2: ReflectionExperience) -> float:
        """计算两个经验的相似度"""
        similarities = []
        
        # 场景类型相似度
        if exp1.scene_type == exp2.scene_type:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # 空间上下文相似度
        if exp1.spatial_context == exp2.spatial_context:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # 行为模式相似度
        if exp1.action_pattern == exp2.action_pattern:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # place关系相似度
        if exp1.place_relationship == exp2.place_relationship:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # 视觉上下文相似度
        if exp1.visual_context == exp2.visual_context:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # 视觉地标相似度
        if exp1.visual_landmarks == exp2.visual_landmarks:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # 视觉朝向相似度
        if exp1.visual_orientation == exp2.visual_orientation:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # 加权平均
        return sum(sim * weight for sim, weight in zip(similarities, self.fusion_weights.values()))

    def _update_existing_experience(self, idx: int, new_exp: ReflectionExperience):
        """更新现有经验"""
        exp = self.experiences[idx]
        
        # 更新成功率和频次
        total_success = exp.success_rate * exp.frequency + new_exp.success_rate
        exp.frequency += 1
        exp.success_rate = total_success / exp.frequency
        
        # 更新决策置信度（加权平均）
        exp.decision_confidence = (exp.decision_confidence * (exp.frequency - 1) + 
                                 new_exp.decision_confidence) / exp.frequency
        
        exp.last_updated = new_exp.last_updated

    def _add_new_experience(self, experience: ReflectionExperience):
        """添加新经验"""
        if len(self.experiences) >= self.max_experiences:
            # 移除最旧的经验
            self.experiences.pop(0)
        
        self.experiences.append(experience)
        
        # 更新索引
        idx = len(self.experiences) - 1
        self.scene_type_index[experience.scene_type].append(idx)
        self.spatial_context_index[experience.spatial_context].append(idx)
        self.action_pattern_index[experience.action_pattern].append(idx)

    def retrieve_relevant_experiences(self, scene_type: str, spatial_context: str, 
                                    action_pattern: str, visual_context: str = None,
                                    visual_landmarks: str = None, visual_orientation: str = None) -> List[ReflectionExperience]:
        """检索相关经验"""
        relevant_indices = set()
        
        # 根据场景类型检索
        if scene_type in self.scene_type_index:
            relevant_indices.update(self.scene_type_index[scene_type])
        
        # 根据空间上下文检索
        if spatial_context in self.spatial_context_index:
            relevant_indices.update(self.spatial_context_index[spatial_context])
        
        # 根据行为模式检索
        if action_pattern in self.action_pattern_index:
            relevant_indices.update(self.action_pattern_index[action_pattern])
        
        # 返回相关经验
        relevant_experiences = [self.experiences[i] for i in relevant_indices if i < len(self.experiences)]
        
        # 按成功率和置信度排序
        relevant_experiences.sort(key=lambda x: (x.success_rate, x.decision_confidence), reverse=True)
        
        return relevant_experiences[:10]  # 返回前10个最相关的经验

    def get_experience_embeddings(self, experiences: List[ReflectionExperience]) -> torch.Tensor:
        """将经验转换为嵌入向量"""
        if not experiences:
            return torch.zeros(1, 768)  # 默认维度
        
        # 构建经验文本
        experience_texts = []
        for exp in experiences:
            text = f"场景:{exp.scene_type} 上下文:{exp.spatial_context} 行为:{exp.action_pattern} 规则:{exp.spatial_rule} 策略:{exp.navigation_strategy}"
            experience_texts.append(text)
        
        # 这里需要使用文本编码器（如BERT）将文本转换为嵌入
        # 暂时返回随机向量作为占位符
        embeddings = torch.randn(len(experience_texts), 768)
        
        return embeddings

    def save_memory_bank(self, filepath: str):
        """保存记忆银行"""
        data = {
            'experiences': [
                {
                    'scene_type': exp.scene_type,
                    'spatial_context': exp.spatial_context,
                    'action_pattern': exp.action_pattern,
                    'decision_confidence': exp.decision_confidence,
                    'spatial_rule': exp.spatial_rule,
                    'navigation_strategy': exp.navigation_strategy,
                    'success_rate': exp.success_rate,
                    'frequency': exp.frequency,
                    'last_updated': exp.last_updated
                }
                for exp in self.experiences
            ],
            'fusion_weights': self.fusion_weights
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_memory_bank(self, filepath: str):
        """加载记忆银行"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建经验对象
            self.experiences = []
            for exp_data in data['experiences']:
                exp = ReflectionExperience(
                    scene_type=exp_data['scene_type'],
                    spatial_context=exp_data['spatial_context'],
                    action_pattern=exp_data['action_pattern'],
                    decision_confidence=exp_data['decision_confidence'],
                    spatial_rule=exp_data['spatial_rule'],
                    navigation_strategy=exp_data['navigation_strategy'],
                    success_rate=exp_data['success_rate'],
                    frequency=exp_data['frequency'],
                    last_updated=exp_data['last_updated']
                )
                self.experiences.append(exp)
            
            # 重建索引
            self._rebuild_indices()
            
            # 恢复融合权重
            self.fusion_weights = data.get('fusion_weights', self.fusion_weights)
            
        except Exception as e:
            logger.error(f"加载记忆银行失败: {e}")

    def _rebuild_indices(self):
        """重建索引"""
        self.scene_type_index = defaultdict(list)
        self.spatial_context_index = defaultdict(list)
        self.action_pattern_index = defaultdict(list)
        
        for i, exp in enumerate(self.experiences):
            self.scene_type_index[exp.scene_type].append(i)
            self.spatial_context_index[exp.spatial_context].append(i)
            self.action_pattern_index[exp.action_pattern].append(i)
