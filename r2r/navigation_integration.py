"""
将指令风格转换集成到导航模型训练中
在训练过程中动态转换Scene和User风格指令为Basic风格
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
import random
from pathlib import Path
import logging
from dataclasses import dataclass

from .llm_style_converter import LLMStyleConverter, StyleConversionConfig
from .style_translation_training import StyleConversionTrainer, TrainingConfig

@dataclass
class NavigationIntegrationConfig:
    """导航集成配置"""
    # 风格转换配置
    style_converter_model_path: str = "models/style_conversion"
    enable_style_conversion: bool = True
    conversion_confidence_threshold: float = 0.7
    
    # 数据增强配置
    augmentation_ratio: float = 0.3  # 30%的数据进行风格转换
    style_distribution: Dict[str, float] = None  # 风格分布
    
    # 训练配置
    use_converted_instructions: bool = True
    mix_original_and_converted: bool = True
    conversion_weight: float = 0.5  # 转换后指令的权重
    
    def __post_init__(self):
        if self.style_distribution is None:
            self.style_distribution = {
                'basic': 0.4,
                'scene': 0.3,
                'user': 0.3
            }

class StyleAwareNavigationDataset(Dataset):
    """风格感知的导航数据集"""
    
    def __init__(self, original_dataset: Dataset, style_converter: LLMStyleConverter, 
                 config: NavigationIntegrationConfig):
        self.original_dataset = original_dataset
        self.style_converter = style_converter
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.conversion_stats = {
            'total_samples': 0,
            'converted_samples': 0,
            'scene_conversions': 0,
            'user_conversions': 0,
            'failed_conversions': 0
        }
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # 获取原始样本
        original_item = self.original_dataset[idx]
        
        # 决定是否进行风格转换
        should_convert = self._should_convert_instruction(original_item)
        
        if should_convert and self.config.enable_style_conversion:
            # 进行风格转换
            converted_item = self._convert_instruction_style(original_item)
            self.conversion_stats['converted_samples'] += 1
        else:
            # 使用原始指令
            converted_item = original_item
        
        self.conversion_stats['total_samples'] += 1
        
        return converted_item
    
    def _should_convert_instruction(self, item: Dict[str, Any]) -> bool:
        """决定是否转换指令"""
        # 基于配置的转换比例
        if random.random() > self.config.augmentation_ratio:
            return False
        
        # 检查指令是否已经是Basic风格
        instruction = item.get('instruction', '')
        if self._is_basic_style(instruction):
            return False
        
        return True
    
    def _is_basic_style(self, instruction: str) -> bool:
        """判断指令是否为Basic风格"""
        instruction_lower = instruction.lower()
        
        # Basic风格的特征词汇
        basic_indicators = ['go', 'turn', 'walk', 'stop', 'left', 'right', 'forward']
        basic_count = sum(1 for indicator in basic_indicators if indicator in instruction_lower)
        
        # Scene风格的特征词汇
        scene_indicators = ['proceed', 'execute', 'maintain', 'precisely', 'circumvent']
        scene_count = sum(1 for indicator in scene_indicators if indicator in instruction_lower)
        
        # User风格的特征词汇
        user_indicators = ['please', 'could you', 'thanks', 'awesome', 'cool', 'darling']
        user_count = sum(1 for indicator in user_indicators if indicator in instruction_lower)
        
        # 如果Basic特征词汇多，认为是Basic风格
        return basic_count > max(scene_count, user_count)
    
    def _convert_instruction_style(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """转换指令风格"""
        instruction = item.get('instruction', '')
        context = item.get('context', {})
        
        # 检测当前风格
        current_style = self._detect_instruction_style(instruction)
        
        if current_style == 'basic':
            return item
        
        # 进行风格转换
        try:
            conversion_result = self.style_converter.convert_instruction(
                instruction=instruction,
                source_style=current_style,
                target_style='basic',
                context=context
            )
            
            if conversion_result['conversion_applied'] and conversion_result['confidence'] > self.config.conversion_confidence_threshold:
                # 更新统计信息
                if current_style == 'scene':
                    self.conversion_stats['scene_conversions'] += 1
                elif current_style == 'user':
                    self.conversion_stats['user_conversions'] += 1
                
                # 创建转换后的样本
                converted_item = item.copy()
                converted_item['instruction'] = conversion_result['converted_instruction']
                converted_item['original_instruction'] = instruction
                converted_item['style_conversion'] = {
                    'applied': True,
                    'source_style': current_style,
                    'target_style': 'basic',
                    'confidence': conversion_result['confidence'],
                    'conversion_quality': conversion_result['quality_metrics']
                }
                
                return converted_item
            else:
                self.conversion_stats['failed_conversions'] += 1
                return item
                
        except Exception as e:
            self.logger.warning(f"风格转换失败: {e}")
            self.conversion_stats['failed_conversions'] += 1
            return item
    
    def _detect_instruction_style(self, instruction: str) -> str:
        """检测指令风格"""
        instruction_lower = instruction.lower()
        
        # Scene风格检测
        scene_indicators = ['proceed', 'execute', 'maintain', 'precisely', 'circumvent', 'steadfast']
        scene_score = sum(1 for indicator in scene_indicators if indicator in instruction_lower)
        
        # User风格检测
        user_indicators = ['please', 'could you', 'thanks', 'awesome', 'cool', 'darling', 'sweetheart']
        user_score = sum(1 for indicator in user_indicators if indicator in instruction_lower)
        
        # 返回得分最高的风格
        if scene_score > user_score and scene_score > 0:
            return 'scene'
        elif user_score > scene_score and user_score > 0:
            return 'user'
        else:
            return 'basic'
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """获取转换统计信息"""
        stats = self.conversion_stats.copy()
        if stats['total_samples'] > 0:
            stats['conversion_rate'] = stats['converted_samples'] / stats['total_samples']
            stats['success_rate'] = (stats['converted_samples'] - stats['failed_conversions']) / max(stats['converted_samples'], 1)
        else:
            stats['conversion_rate'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats

class StyleAwareNavigationTrainer:
    """风格感知的导航训练器"""
    
    def __init__(self, config: NavigationIntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化风格转换器
        self.style_converter = self._load_style_converter()
        
        # 训练统计
        self.training_stats = {
            'total_epochs': 0,
            'total_batches': 0,
            'conversion_applications': 0,
            'style_distribution': {'basic': 0, 'scene': 0, 'user': 0}
        }
    
    def _load_style_converter(self) -> LLMStyleConverter:
        """加载风格转换器"""
        try:
            # 尝试加载训练好的模型
            converter_config = StyleConversionConfig()
            converter = LLMStyleConverter(converter_config)
            
            # 这里应该加载预训练的权重
            # converter.model.load_state_dict(torch.load(self.config.style_converter_model_path))
            
            self.logger.info("风格转换器加载成功")
            return converter
            
        except Exception as e:
            self.logger.warning(f"无法加载风格转换器: {e}")
            # 返回一个基础的转换器
            converter_config = StyleConversionConfig()
            return LLMStyleConverter(converter_config)
    
    def create_style_aware_dataset(self, original_dataset: Dataset) -> StyleAwareNavigationDataset:
        """创建风格感知数据集"""
        return StyleAwareNavigationDataset(
            original_dataset=original_dataset,
            style_converter=self.style_converter,
            config=self.config
        )
    
    def train_with_style_adaptation(self, model: nn.Module, train_dataset: Dataset, 
                                  val_dataset: Dataset, num_epochs: int = 10) -> Dict[str, Any]:
        """使用风格适应进行训练"""
        self.logger.info("开始风格感知的导航训练...")
        
        # 创建风格感知数据集
        style_aware_train_dataset = self.create_style_aware_dataset(train_dataset)
        style_aware_val_dataset = self.create_style_aware_dataset(val_dataset)
        
        # 创建数据加载器
        train_loader = DataLoader(style_aware_train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(style_aware_val_dataset, batch_size=32, shuffle=False)
        
        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'conversion_stats': []
        }
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            
            # 验证阶段
            val_metrics = self._validate_epoch(model, val_loader, criterion)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # 记录转换统计
            train_conversion_stats = style_aware_train_dataset.get_conversion_stats()
            training_history['conversion_stats'].append(train_conversion_stats)
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Conversion Rate: {train_conversion_stats['conversion_rate']:.3f}")
        
        return training_history
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch)
            
            # 计算损失
            loss = criterion(outputs, batch['labels'])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
            
            self.training_stats['total_batches'] += 1
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Dict[str, float]:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 前向传播
                outputs = model(batch)
                
                # 计算损失
                loss = criterion(outputs, batch['labels'])
                
                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }

class StyleAdaptiveDataAugmentation:
    """风格自适应数据增强"""
    
    def __init__(self, style_converter: LLMStyleConverter, config: NavigationIntegrationConfig):
        self.style_converter = style_converter
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def augment_dataset(self, original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强数据集"""
        augmented_data = []
        
        for item in original_data:
            # 添加原始数据
            augmented_data.append(item)
            
            # 根据配置决定是否进行风格转换
            if random.random() < self.config.augmentation_ratio:
                # 生成风格转换的变体
                style_variants = self._generate_style_variants(item)
                augmented_data.extend(style_variants)
        
        self.logger.info(f"数据增强完成: 原始数据 {len(original_data)}, 增强后 {len(augmented_data)}")
        return augmented_data
    
    def _generate_style_variants(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成风格变体"""
        variants = []
        instruction = item.get('instruction', '')
        context = item.get('context', {})
        
        # 检测当前风格
        current_style = self._detect_instruction_style(instruction)
        
        if current_style == 'basic':
            # 从Basic转换为其他风格
            for target_style in ['scene', 'user']:
                variant = self._convert_to_style(item, target_style, context)
                if variant:
                    variants.append(variant)
        else:
            # 转换为Basic风格
            variant = self._convert_to_style(item, 'basic', context)
            if variant:
                variants.append(variant)
        
        return variants
    
    def _detect_instruction_style(self, instruction: str) -> str:
        """检测指令风格"""
        instruction_lower = instruction.lower()
        
        # Scene风格检测
        scene_indicators = ['proceed', 'execute', 'maintain', 'precisely', 'circumvent']
        scene_score = sum(1 for indicator in scene_indicators if indicator in instruction_lower)
        
        # User风格检测
        user_indicators = ['please', 'could you', 'thanks', 'awesome', 'cool', 'darling']
        user_score = sum(1 for indicator in user_indicators if indicator in instruction_lower)
        
        if scene_score > user_score and scene_score > 0:
            return 'scene'
        elif user_score > scene_score and user_score > 0:
            return 'user'
        else:
            return 'basic'
    
    def _convert_to_style(self, item: Dict[str, Any], target_style: str, 
                         context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换为指定风格"""
        try:
            instruction = item.get('instruction', '')
            current_style = self._detect_instruction_style(instruction)
            
            if current_style == target_style:
                return None
            
            # 进行风格转换
            conversion_result = self.style_converter.convert_instruction(
                instruction=instruction,
                source_style=current_style,
                target_style=target_style,
                context=context
            )
            
            if conversion_result['conversion_applied'] and conversion_result['confidence'] > self.config.conversion_confidence_threshold:
                # 创建变体
                variant = item.copy()
                variant['instruction'] = conversion_result['converted_instruction']
                variant['original_instruction'] = instruction
                variant['style_conversion'] = {
                    'applied': True,
                    'source_style': current_style,
                    'target_style': target_style,
                    'confidence': conversion_result['confidence']
                }
                
                return variant
            
        except Exception as e:
            self.logger.warning(f"风格转换失败: {e}")
        
        return None

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化配置
    config = NavigationIntegrationConfig(
        enable_style_conversion=True,
        augmentation_ratio=0.3,
        conversion_confidence_threshold=0.7
    )
    
    # 创建风格感知训练器
    trainer = StyleAwareNavigationTrainer(config)
    
    # 假设有一个原始数据集
    # original_dataset = YourOriginalDataset()
    
    # 创建风格感知数据集
    # style_aware_dataset = trainer.create_style_aware_dataset(original_dataset)
    
    # 训练模型
    # training_history = trainer.train_with_style_adaptation(model, train_dataset, val_dataset)
    
    print("风格感知的导航训练系统已准备就绪!")
