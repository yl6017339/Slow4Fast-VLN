"""
指令风格转换训练流程
训练基于LLM的指令风格转换器，将Scene和User风格转换为Basic风格
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import random
from tqdm import tqdm

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "t5-base"
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 0.7
    
    # 训练配置
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # 数据配置
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 路径配置
    data_dir: str = "data/style_conversion"
    output_dir: str = "outputs/style_conversion"
    model_save_dir: str = "models/style_conversion"
    
    # 日志配置
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000

class StyleConversionDataset(Dataset):
    """风格转换数据集"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本（包含风格转换提示）
        input_text = self._build_input_text(item)
        target_text = item['target_instruction']
        
        # 编码输入和输出
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'source_style': item['source_style'],
            'target_style': item['target_style']
        }
    
    def _build_input_text(self, item: Dict[str, Any]) -> str:
        """构建输入文本"""
        source_style = item['source_style']
        target_style = item['target_style']
        instruction = item['source_instruction']
        context = item.get('context', {})
        
        if source_style == "scene" and target_style == "basic":
            return f"Convert scene style to basic: {instruction}"
        elif source_style == "user" and target_style == "basic":
            user_type = context.get('user_type', 'unknown')
            return f"Convert user style ({user_type}) to basic: {instruction}"
        elif source_style == "basic" and target_style == "scene":
            scene_type = context.get('scene_type', 'office')
            return f"Convert basic to scene style ({scene_type}): {instruction}"
        elif source_style == "basic" and target_style == "user":
            user_type = context.get('user_type', 'child')
            return f"Convert basic to user style ({user_type}): {instruction}"
        else:
            return f"Convert {source_style} to {target_style}: {instruction}"

class StyleConversionDataGenerator:
    """风格转换数据生成器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 基础指令模板
        self.basic_templates = [
            "Go to the {location}",
            "Turn {direction}",
            "Walk {direction}",
            "Stop at the {object}",
            "Go around the {object}",
            "Walk through the {location}",
            "Turn around",
            "Go back",
            "Continue {direction}",
            "Move to the {location}"
        ]
        
        # 场景风格转换规则
        self.scene_conversion_rules = {
            'go': 'proceed',
            'walk': 'navigate',
            'turn': 'execute a rotation',
            'stop': 'halt',
            'go around': 'circumvent',
            'continue': 'maintain',
            'move': 'proceed to',
            'back': 'reverse direction'
        }
        
        # 用户风格转换规则
        self.user_conversion_rules = {
            'child': {
                'go': 'go to',
                'walk': 'walk to',
                'turn': 'turn around',
                'stop': 'stop at',
                'please': 'please',
                'thanks': 'thanks'
            },
            'keith': {
                'go': 'proceed to',
                'walk': 'navigate to',
                'turn': 'execute a turn',
                'stop': 'halt at',
                'please': 'kindly',
                'thanks': 'appreciated'
            },
            'moira': {
                'go': 'darling, go to',
                'walk': 'sweetheart, walk to',
                'turn': 'dear, turn',
                'stop': 'lovely, stop at',
                'please': 'please',
                'thanks': 'wonderful'
            },
            'rachel': {
                'go': 'like, go to',
                'walk': 'totally walk to',
                'turn': 'amazing, turn',
                'stop': 'incredible, stop at',
                'please': 'please',
                'thanks': 'awesome'
            },
            'sheldon': {
                'go': 'fascinating, proceed to',
                'walk': 'intriguing, navigate to',
                'turn': 'precisely, execute a turn',
                'stop': 'logically, halt at',
                'please': 'please',
                'thanks': 'remarkable'
            }
        }
        
        # 场景和对象词汇
        self.scene_vocabulary = {
            'office': ['workstation', 'cubicle', 'conference room', 'reception desk'],
            'cinema': ['auditorium', 'lobby', 'concession stand', 'theater'],
            'shop': ['aisle', 'checkout', 'display', 'merchandise'],
            'salon': ['station', 'chair', 'mirror', 'styling area'],
            'laboratory': ['equipment', 'specimen', 'analysis area', 'research station']
        }
    
    def generate_training_data(self, num_samples: int = 10000) -> List[Dict[str, Any]]:
        """生成训练数据"""
        self.logger.info(f"生成 {num_samples} 个训练样本")
        
        data = []
        
        # 生成基础到场景的转换数据
        scene_data = self._generate_scene_conversion_data(num_samples // 4)
        data.extend(scene_data)
        
        # 生成基础到用户的转换数据
        user_data = self._generate_user_conversion_data(num_samples // 4)
        data.extend(user_data)
        
        # 生成场景到基础的转换数据
        scene_to_basic_data = self._generate_scene_to_basic_data(num_samples // 4)
        data.extend(scene_to_basic_data)
        
        # 生成用户到基础的转换数据
        user_to_basic_data = self._generate_user_to_basic_data(num_samples // 4)
        data.extend(user_to_basic_data)
        
        # 打乱数据
        random.shuffle(data)
        
        self.logger.info(f"生成了 {len(data)} 个训练样本")
        return data
    
    def _generate_scene_conversion_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成基础到场景的转换数据"""
        data = []
        
        for _ in range(num_samples):
            # 随机选择基础模板
            template = random.choice(self.basic_templates)
            
            # 随机选择场景类型
            scene_type = random.choice(list(self.scene_vocabulary.keys()))
            objects = self.scene_vocabulary[scene_type]
            
            # 填充模板
            basic_instruction = template.format(
                location=random.choice(objects),
                direction=random.choice(['left', 'right', 'forward', 'backward']),
                object=random.choice(objects)
            )
            
            # 转换为场景风格
            scene_instruction = self._convert_to_scene_style(basic_instruction, scene_type)
            
            data.append({
                'source_instruction': basic_instruction,
                'target_instruction': scene_instruction,
                'source_style': 'basic',
                'target_style': 'scene',
                'context': {'scene_type': scene_type}
            })
        
        return data
    
    def _generate_user_conversion_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成基础到用户的转换数据"""
        data = []
        
        for _ in range(num_samples):
            # 随机选择基础模板
            template = random.choice(self.basic_templates)
            
            # 随机选择用户类型
            user_type = random.choice(list(self.user_conversion_rules.keys()))
            
            # 填充模板
            basic_instruction = template.format(
                location=random.choice(['kitchen', 'bathroom', 'bedroom', 'living room']),
                direction=random.choice(['left', 'right', 'forward', 'backward']),
                object=random.choice(['door', 'window', 'table', 'chair'])
            )
            
            # 转换为用户风格
            user_instruction = self._convert_to_user_style(basic_instruction, user_type)
            
            data.append({
                'source_instruction': basic_instruction,
                'target_instruction': user_instruction,
                'source_style': 'basic',
                'target_style': 'user',
                'context': {'user_type': user_type}
            })
        
        return data
    
    def _generate_scene_to_basic_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成场景到基础的转换数据"""
        data = []
        
        for _ in range(num_samples):
            # 生成场景风格指令
            scene_instruction = self._generate_scene_instruction()
            
            # 转换为基础风格
            basic_instruction = self._convert_to_basic_style(scene_instruction)
            
            data.append({
                'source_instruction': scene_instruction,
                'target_instruction': basic_instruction,
                'source_style': 'scene',
                'target_style': 'basic',
                'context': {'scene_type': 'office'}
            })
        
        return data
    
    def _generate_user_to_basic_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成用户到基础的转换数据"""
        data = []
        
        for _ in range(num_samples):
            # 随机选择用户类型
            user_type = random.choice(list(self.user_conversion_rules.keys()))
            
            # 生成用户风格指令
            user_instruction = self._generate_user_instruction(user_type)
            
            # 转换为基础风格
            basic_instruction = self._convert_to_basic_style(user_instruction)
            
            data.append({
                'source_instruction': user_instruction,
                'target_instruction': basic_instruction,
                'source_style': 'user',
                'target_style': 'basic',
                'context': {'user_type': user_type}
            })
        
        return data
    
    def _convert_to_scene_style(self, instruction: str, scene_type: str) -> str:
        """转换为场景风格"""
        scene_instruction = instruction
        
        # 应用场景转换规则
        for basic, scene in self.scene_conversion_rules.items():
            scene_instruction = scene_instruction.replace(basic, scene)
        
        # 添加场景特定的修饰词
        if scene_type == 'office':
            scene_instruction = f"Proceed with precision: {scene_instruction}"
        elif scene_type == 'cinema':
            scene_instruction = f"Execute the following navigation: {scene_instruction}"
        
        return scene_instruction
    
    def _convert_to_user_style(self, instruction: str, user_type: str) -> str:
        """转换为用户风格"""
        user_rules = self.user_conversion_rules[user_type]
        user_instruction = instruction
        
        # 应用用户转换规则
        for basic, user in user_rules.items():
            user_instruction = user_instruction.replace(basic, user)
        
        # 添加用户特定的表达
        if user_type == 'child':
            user_instruction = f"Hey, {user_instruction} please!"
        elif user_type == 'keith':
            user_instruction = f"Kindly {user_instruction.lower()}"
        elif user_type == 'moira':
            user_instruction = f"Darling, {user_instruction.lower()}"
        elif user_type == 'rachel':
            user_instruction = f"Like, {user_instruction.lower()}, that would be amazing!"
        elif user_type == 'sheldon':
            user_instruction = f"Fascinating, {user_instruction.lower()}"
        
        return user_instruction
    
    def _generate_scene_instruction(self) -> str:
        """生成场景风格指令"""
        templates = [
            "Proceed forward, circumventing the {object} with the utmost care",
            "Execute a {direction} rotation while maintaining a steadfast course",
            "Navigate through the {location} with precision",
            "Halt precisely at the {object} and await further instructions",
            "Maintain forward progression while avoiding the {object}"
        ]
        
        template = random.choice(templates)
        objects = ['workstation', 'equipment', 'furniture', 'obstacle']
        locations = ['corridor', 'chamber', 'facility', 'area']
        directions = ['leftward', 'rightward', 'clockwise', 'counterclockwise']
        
        return template.format(
            object=random.choice(objects),
            location=random.choice(locations),
            direction=random.choice(directions)
        )
    
    def _generate_user_instruction(self, user_type: str) -> str:
        """生成用户风格指令"""
        if user_type == 'child':
            templates = [
                "Hey, could you go to the {location} please?",
                "Can you walk to the {object}? Thanks!",
                "Please turn {direction}, that would be awesome!"
            ]
        elif user_type == 'keith':
            templates = [
                "Kindly proceed to the {location}",
                "Please navigate to the {object} with precision",
                "Execute a {direction} turn when convenient"
            ]
        elif user_type == 'moira':
            templates = [
                "Darling, could you go to the {location} please?",
                "Sweetheart, walk to the {object} for me",
                "Dear, turn {direction} when you're ready"
            ]
        elif user_type == 'rachel':
            templates = [
                "Like, go to the {location}, that would be amazing!",
                "Totally walk to the {object}, it's incredible!",
                "Turn {direction}, that's fantastic!"
            ]
        elif user_type == 'sheldon':
            templates = [
                "Fascinating, proceed to the {location}",
                "Intriguing, navigate to the {object} with logical precision",
                "Execute a {direction} turn systematically"
            ]
        
        template = random.choice(templates)
        locations = ['kitchen', 'bathroom', 'bedroom', 'living room']
        objects = ['door', 'window', 'table', 'chair']
        directions = ['left', 'right', 'around']
        
        return template.format(
            location=random.choice(locations),
            object=random.choice(objects),
            direction=random.choice(directions)
        )
    
    def _convert_to_basic_style(self, instruction: str) -> str:
        """转换为基础风格"""
        basic_instruction = instruction.lower()
        
        # 移除用户特定的表达
        user_expressions = [
            'hey', 'could you', 'please', 'thanks', 'awesome', 'cool',
            'darling', 'sweetheart', 'dear', 'lovely', 'wonderful',
            'like', 'totally', 'amazing', 'incredible', 'fantastic',
            'fascinating', 'intriguing', 'precisely', 'logically'
        ]
        
        for expr in user_expressions:
            basic_instruction = basic_instruction.replace(expr, '')
        
        # 移除场景特定的表达
        scene_expressions = [
            'proceed', 'execute', 'maintain', 'precisely', 'circumvent',
            'steadfast', 'utmost', 'care', 'rotation', 'navigate'
        ]
        
        for expr in scene_expressions:
            if expr == 'proceed':
                basic_instruction = basic_instruction.replace(expr, 'go')
            elif expr == 'execute':
                basic_instruction = basic_instruction.replace(expr, 'turn')
            elif expr == 'navigate':
                basic_instruction = basic_instruction.replace(expr, 'walk')
            elif expr == 'circumvent':
                basic_instruction = basic_instruction.replace(expr, 'go around')
            elif expr == 'halt':
                basic_instruction = basic_instruction.replace(expr, 'stop')
        
        # 清理多余的空格和标点
        basic_instruction = ' '.join(basic_instruction.split())
        basic_instruction = basic_instruction.strip('.,!?')
        
        return basic_instruction

class StyleConversionTrainer:
    """风格转换训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # 添加特殊token
        self._add_special_tokens()
        
        # 初始化数据生成器
        self.data_generator = StyleConversionDataGenerator(config)
    
    def _add_special_tokens(self):
        """添加特殊token"""
        special_tokens = [
            "<scene>", "</scene>",
            "<user>", "</user>",
            "<basic>", "</basic>",
            "<child>", "<keith>", "<moira>", "<rachel>", "<sheldon>",
            "<office>", "<cinema>", "<shop>", "<salon>", "<laboratory>"
        ]
        
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_data(self, num_samples: int = 10000) -> Tuple[Dataset, Dataset, Dataset]:
        """准备训练数据"""
        self.logger.info("准备训练数据...")
        
        # 生成原始数据
        raw_data = self.data_generator.generate_training_data(num_samples)
        
        # 分割数据
        train_data, val_data, test_data = self._split_data(raw_data)
        
        # 创建数据集
        train_dataset = StyleConversionDataset(train_data, self.tokenizer, self.config.max_length)
        val_dataset = StyleConversionDataset(val_data, self.tokenizer, self.config.max_length)
        test_dataset = StyleConversionDataset(test_data, self.tokenizer, self.config.max_length)
        
        self.logger.info(f"数据分割完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _split_data(self, data: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """分割数据"""
        random.shuffle(data)
        
        n = len(data)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
        """训练模型"""
        self.logger.info("开始训练模型...")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        training_result = trainer.train()
        
        # 保存模型
        self._save_model()
        
        self.logger.info("训练完成!")
        return training_result
    
    def _save_model(self):
        """保存模型"""
        save_path = Path(self.config.model_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"模型已保存到: {save_path}")
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """评估模型"""
        self.logger.info("评估模型...")
        
        # 设置评估参数
        eval_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.batch_size,
            do_predict=True,
            report_to=None
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 创建评估器
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 进行评估
        eval_result = trainer.evaluate()
        
        self.logger.info(f"评估结果: {eval_result}")
        return eval_result

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化配置
    config = TrainingConfig(
        model_name="t5-base",
        num_epochs=5,
        batch_size=8,
        learning_rate=5e-5
    )
    
    # 创建训练器
    trainer = StyleConversionTrainer(config)
    
    # 准备数据
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(num_samples=1000)
    
    # 训练模型
    training_result = trainer.train(train_dataset, val_dataset)
    
    # 评估模型
    eval_result = trainer.evaluate(test_dataset)
    
    print("训练完成!")
    print(f"训练结果: {training_result}")
    print(f"评估结果: {eval_result}")
