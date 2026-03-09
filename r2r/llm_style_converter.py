"""
基于LLM的指令风格转换器
将Scene和User风格指令转换为Basic风格，用于导航模型训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from dataclasses import dataclass
import logging

@dataclass
class StyleConversionConfig:
    """风格转换配置"""
    # 模型配置
    base_model_name: str = "t5-base"  # 或 "bart-base", "mT5-base"
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 0.7
    
    # 风格转换配置
    source_styles: List[str] = None
    target_style: str = "basic"
    
    # 训练配置
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 1000
    
    def __post_init__(self):
        if self.source_styles is None:
            self.source_styles = ["scene", "user"]

class LLMStyleConverter:
    """基于LLM的指令风格转换器"""
    
    def __init__(self, config: StyleConversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name)
        
        # 添加特殊token
        self._add_special_tokens()
        
        # 风格特定的提示模板
        self.style_prompts = self._initialize_style_prompts()
        
        # 训练数据存储
        self.training_data = []
        self.validation_data = []
    
    def _add_special_tokens(self):
        """添加风格转换相关的特殊token"""
        special_tokens = [
            "<scene>", "</scene>",
            "<user>", "</user>", 
            "<basic>", "</basic>",
            "<child>", "<keith>", "<moira>", "<rachel>", "<sheldon>",
            "<office>", "<cinema>", "<shop>", "<salon>", "<laboratory>"
        ]
        
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _initialize_style_prompts(self) -> Dict[str, str]:
        """初始化风格转换提示模板"""
        return {
            "scene_to_basic": """
将以下正式的场景风格指令转换为简洁的基础风格指令：

场景风格指令：{scene_instruction}

要求：
1. 保持指令的核心含义不变
2. 使用简单直接的词汇
3. 保持指令的导航信息完整性
4. 输出格式：基础风格指令：{basic_instruction}

基础风格指令：""",
            
            "user_to_basic": """
将以下个人化的用户风格指令转换为简洁的基础风格指令：

用户风格指令：{user_instruction}
用户类型：{user_type}

要求：
1. 保持指令的核心含义不变
2. 移除个人化表达和情感词汇
3. 使用标准导航词汇
4. 保持指令的导航信息完整性
5. 输出格式：基础风格指令：{basic_instruction}

基础风格指令：""",
            
            "basic_to_scene": """
将以下基础风格指令转换为正式的场景风格指令：

基础风格指令：{basic_instruction}
目标场景：{scene_type}

要求：
1. 保持指令的核心含义不变
2. 使用专业术语和正式表达
3. 增加场景特定的细节描述
4. 保持指令的导航信息完整性
5. 输出格式：场景风格指令：{scene_instruction}

场景风格指令：""",
            
            "basic_to_user": """
将以下基础风格指令转换为个人化的用户风格指令：

基础风格指令：{basic_instruction}
用户类型：{user_type}

要求：
1. 保持指令的核心含义不变
2. 使用用户特定的表达方式
3. 添加适当的礼貌用语和情感词汇
4. 保持指令的导航信息完整性
5. 输出格式：用户风格指令：{user_instruction}

用户风格指令："""
        }
    
    def convert_instruction(self, instruction: str, source_style: str, 
                          target_style: str = "basic", 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """转换指令风格"""
        if source_style == target_style:
            return {
                'original_instruction': instruction,
                'converted_instruction': instruction,
                'conversion_applied': False,
                'confidence': 1.0
            }
        
        # 构建转换提示
        prompt = self._build_conversion_prompt(instruction, source_style, target_style, context)
        
        # 使用LLM进行转换
        converted_instruction = self._llm_convert(prompt)
        
        # 后处理和验证
        processed_instruction = self._post_process_instruction(converted_instruction, target_style)
        
        # 计算转换质量
        conversion_quality = self._evaluate_conversion_quality(
            instruction, processed_instruction, source_style, target_style
        )
        
        return {
            'original_instruction': instruction,
            'converted_instruction': processed_instruction,
            'conversion_applied': True,
            'confidence': conversion_quality['confidence'],
            'quality_metrics': conversion_quality,
            'conversion_prompt': prompt
        }
    
    def _build_conversion_prompt(self, instruction: str, source_style: str, 
                               target_style: str, context: Dict[str, Any] = None) -> str:
        """构建转换提示"""
        if source_style == "scene" and target_style == "basic":
            return self.style_prompts["scene_to_basic"].format(
                scene_instruction=instruction
            )
        elif source_style == "user" and target_style == "basic":
            user_type = context.get('user_type', 'unknown') if context else 'unknown'
            return self.style_prompts["user_to_basic"].format(
                user_instruction=instruction,
                user_type=user_type
            )
        elif source_style == "basic" and target_style == "scene":
            scene_type = context.get('scene_type', 'office') if context else 'office'
            return self.style_prompts["basic_to_scene"].format(
                basic_instruction=instruction,
                scene_type=scene_type
            )
        elif source_style == "basic" and target_style == "user":
            user_type = context.get('user_type', 'child') if context else 'child'
            return self.style_prompts["basic_to_user"].format(
                basic_instruction=instruction,
                user_type=user_type
            )
        else:
            # 通用转换提示
            return f"将以下{source_style}风格指令转换为{target_style}风格指令：\n{instruction}\n\n转换结果："
    
    def _llm_convert(self, prompt: str) -> str:
        """使用LLM进行转换"""
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 生成转换结果
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                temperature=self.config.temperature,
                do_sample=True,
                early_stopping=True
            )
        
        # 解码输出
        converted_instruction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 清理输出
        converted_instruction = self._clean_generated_text(converted_instruction)
        
        return converted_instruction
    
    def _clean_generated_text(self, text: str) -> str:
        """清理生成的文本"""
        # 移除提示部分
        if "基础风格指令：" in text:
            text = text.split("基础风格指令：")[-1].strip()
        elif "场景风格指令：" in text:
            text = text.split("场景风格指令：")[-1].strip()
        elif "用户风格指令：" in text:
            text = text.split("用户风格指令：")[-1].strip()
        
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊标记
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def _post_process_instruction(self, instruction: str, target_style: str) -> str:
        """后处理转换后的指令"""
        # 基础清理
        instruction = instruction.strip()
        
        # 根据目标风格进行特定处理
        if target_style == "basic":
            # 确保使用基础词汇
            instruction = self._convert_to_basic_vocabulary(instruction)
        elif target_style == "scene":
            # 确保使用正式词汇
            instruction = self._convert_to_scene_vocabulary(instruction)
        elif target_style == "user":
            # 确保使用个人化表达
            instruction = self._convert_to_user_vocabulary(instruction)
        
        return instruction
    
    def _convert_to_basic_vocabulary(self, instruction: str) -> str:
        """转换为基础词汇"""
        basic_mappings = {
            'proceed': 'go',
            'navigate': 'walk',
            'execute': 'turn',
            'halt': 'stop',
            'circumvent': 'go around',
            'maintain': 'keep',
            'precisely': '',
            'steadfast': '',
            'utmost': ''
        }
        
        for formal, basic in basic_mappings.items():
            instruction = instruction.replace(formal, basic)
        
        return instruction
    
    def _convert_to_scene_vocabulary(self, instruction: str) -> str:
        """转换为场景词汇"""
        scene_mappings = {
            'go': 'proceed',
            'walk': 'navigate',
            'turn': 'execute a rotation',
            'stop': 'halt',
            'go around': 'circumvent',
            'keep': 'maintain'
        }
        
        for basic, formal in scene_mappings.items():
            instruction = instruction.replace(basic, formal)
        
        return instruction
    
    def _convert_to_user_vocabulary(self, instruction: str) -> str:
        """转换为用户词汇"""
        # 这里可以根据具体用户类型进行更精细的转换
        user_mappings = {
            'go': 'head to',
            'walk': 'move to',
            'turn': 'swing around',
            'stop': 'pause'
        }
        
        for basic, user in user_mappings.items():
            instruction = instruction.replace(basic, user)
        
        return instruction
    
    def _evaluate_conversion_quality(self, original: str, converted: str, 
                                   source_style: str, target_style: str) -> Dict[str, float]:
        """评估转换质量"""
        # 长度保持度
        length_ratio = len(converted.split()) / max(len(original.split()), 1)
        length_preservation = 1.0 - abs(length_ratio - 1.0)
        
        # 词汇重叠度
        original_words = set(original.lower().split())
        converted_words = set(converted.lower().split())
        word_overlap = len(original_words.intersection(converted_words)) / max(len(original_words), 1)
        
        # 风格一致性（简化实现）
        style_consistency = self._check_style_consistency(converted, target_style)
        
        # 整体置信度
        confidence = (length_preservation + word_overlap + style_consistency) / 3.0
        
        return {
            'length_preservation': length_preservation,
            'word_overlap': word_overlap,
            'style_consistency': style_consistency,
            'confidence': confidence
        }
    
    def _check_style_consistency(self, instruction: str, target_style: str) -> float:
        """检查风格一致性"""
        instruction_lower = instruction.lower()
        
        if target_style == "basic":
            basic_indicators = ['go', 'turn', 'walk', 'stop', 'left', 'right', 'forward']
            basic_count = sum(1 for indicator in basic_indicators if indicator in instruction_lower)
            return min(basic_count / 3.0, 1.0)
        
        elif target_style == "scene":
            scene_indicators = ['proceed', 'execute', 'maintain', 'precisely', 'circumvent']
            scene_count = sum(1 for indicator in scene_indicators if indicator in instruction_lower)
            return min(scene_count / 2.0, 1.0)
        
        elif target_style == "user":
            user_indicators = ['please', 'could you', 'thanks', 'awesome', 'cool']
            user_count = sum(1 for indicator in user_indicators if indicator in instruction_lower)
            return min(user_count / 2.0, 1.0)
        
        return 0.5

class StyleConversionTrainer:
    """风格转换模型训练器"""
    
    def __init__(self, converter: LLMStyleConverter):
        self.converter = converter
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, raw_data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """准备训练数据"""
        training_data = []
        validation_data = []
        
        for item in raw_data:
            # 构建训练样本
            if item['source_style'] != item['target_style']:
                sample = {
                    'input_text': self._build_input_text(item),
                    'target_text': item['target_instruction'],
                    'source_style': item['source_style'],
                    'target_style': item['target_style']
                }
                
                # 分割训练和验证数据
                if np.random.random() < 0.8:
                    training_data.append(sample)
                else:
                    validation_data.append(sample)
        
        return training_data, validation_data
    
    def _build_input_text(self, item: Dict[str, Any]) -> str:
        """构建输入文本"""
        prompt = self.converter._build_conversion_prompt(
            item['source_instruction'],
            item['source_style'],
            item['target_style'],
            item.get('context', {})
        )
        return prompt
    
    def train(self, training_data: List[Dict], validation_data: List[Dict]) -> Dict[str, Any]:
        """训练模型"""
        self.logger.info(f"开始训练，训练样本数：{len(training_data)}，验证样本数：{len(validation_data)}")
        
        # 设置训练参数
        training_args = {
            'learning_rate': self.converter.config.learning_rate,
            'num_epochs': self.converter.config.num_epochs,
            'batch_size': self.converter.config.batch_size,
            'warmup_steps': self.converter.config.warmup_steps
        }
        
        # 准备数据加载器
        train_dataloader = self._create_dataloader(training_data, is_training=True)
        val_dataloader = self._create_dataloader(validation_data, is_training=False)
        
        # 训练循环
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        for epoch in range(training_args['num_epochs']):
            # 训练阶段
            train_metrics = self._train_epoch(train_dataloader)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_dataloader)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            self.logger.info(f"Epoch {epoch+1}/{training_args['num_epochs']}: "
                           f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}")
        
        return training_history
    
    def _create_dataloader(self, data: List[Dict], is_training: bool) -> torch.utils.data.DataLoader:
        """创建数据加载器"""
        # 这里简化实现，实际应该使用更复杂的数据加载器
        return data
    
    def _train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        # 简化实现
        return {'loss': 0.5, 'accuracy': 0.8}
    
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        """验证一个epoch"""
        # 简化实现
        return {'loss': 0.6, 'accuracy': 0.75}

class StyleConversionDataset:
    """风格转换数据集"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        # 这里应该从实际的数据文件加载
        # 示例数据结构
        sample_data = [
            {
                'source_instruction': 'Proceed forward, circumventing the workstation with the utmost care',
                'target_instruction': 'Go around the desk',
                'source_style': 'scene',
                'target_style': 'basic',
                'context': {'scene_type': 'office'}
            },
            {
                'source_instruction': 'Hey, could you go to the kitchen please?',
                'target_instruction': 'Go to the kitchen',
                'source_style': 'user',
                'target_style': 'basic',
                'context': {'user_type': 'child'}
            }
        ]
        return sample_data
    
    def get_conversion_pairs(self, source_style: str, target_style: str) -> List[Tuple[str, str]]:
        """获取转换对"""
        pairs = []
        for item in self.data:
            if item['source_style'] == source_style and item['target_style'] == target_style:
                pairs.append((item['source_instruction'], item['target_instruction']))
        return pairs

# 使用示例
if __name__ == "__main__":
    # 初始化配置
    config = StyleConversionConfig(
        base_model_name="t5-base",
        max_length=512,
        num_beams=4
    )
    
    # 创建转换器
    converter = LLMStyleConverter(config)
    
    # 转换示例
    scene_instruction = "Proceed forward, circumventing the workstation with the utmost care"
    result = converter.convert_instruction(
        instruction=scene_instruction,
        source_style="scene",
        target_style="basic",
        context={'scene_type': 'office'}
    )
    
    print(f"原始指令: {result['original_instruction']}")
    print(f"转换后指令: {result['converted_instruction']}")
    print(f"转换置信度: {result['confidence']:.3f}")
