"""
风格转换效果评估框架
评估指令风格转换的质量和导航性能提升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
import random
from pathlib import Path
import logging
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from .llm_style_converter import LLMStyleConverter, StyleConversionConfig

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估指标配置
    evaluate_conversion_quality: bool = True
    evaluate_navigation_performance: bool = True
    evaluate_style_consistency: bool = True
    
    # 测试数据配置
    test_data_size: int = 1000
    style_distribution: Dict[str, float] = None
    
    # 输出配置
    output_dir: str = "evaluation_results"
    save_plots: bool = True
    save_detailed_results: bool = True
    
    def __post_init__(self):
        if self.style_distribution is None:
            self.style_distribution = {
                'basic': 0.4,
                'scene': 0.3,
                'user': 0.3
            }

class StyleConversionEvaluator:
    """风格转换评估器"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化风格转换器
        self.style_converter = LLMStyleConverter(StyleConversionConfig())
        
        # 评估结果存储
        self.evaluation_results = {
            'conversion_quality': {},
            'navigation_performance': {},
            'style_consistency': {},
            'overall_metrics': {}
        }
    
    def evaluate_style_conversion(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估风格转换效果"""
        self.logger.info("开始评估风格转换效果...")
        
        # 转换质量评估
        if self.config.evaluate_conversion_quality:
            conversion_quality = self._evaluate_conversion_quality(test_data)
            self.evaluation_results['conversion_quality'] = conversion_quality
        
        # 风格一致性评估
        if self.config.evaluate_style_consistency:
            style_consistency = self._evaluate_style_consistency(test_data)
            self.evaluation_results['style_consistency'] = style_consistency
        
        # 导航性能评估
        if self.config.evaluate_navigation_performance:
            navigation_performance = self._evaluate_navigation_performance(test_data)
            self.evaluation_results['navigation_performance'] = navigation_performance
        
        # 计算总体指标
        self._calculate_overall_metrics()
        
        # 保存结果
        self._save_evaluation_results()
        
        return self.evaluation_results
    
    def _evaluate_conversion_quality(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估转换质量"""
        self.logger.info("评估转换质量...")
        
        quality_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confidence_scores': [],
            'conversion_success_rate': 0.0,
            'style_detection_accuracy': 0.0
        }
        
        # 转换质量统计
        total_conversions = 0
        successful_conversions = 0
        confidence_scores = []
        style_detection_correct = 0
        style_detection_total = 0
        
        for item in test_data:
            instruction = item.get('instruction', '')
            source_style = item.get('source_style', 'basic')
            target_style = item.get('target_style', 'basic')
            context = item.get('context', {})
            
            if source_style != target_style:
                total_conversions += 1
                
                # 进行风格转换
                conversion_result = self.style_converter.convert_instruction(
                    instruction=instruction,
                    source_style=source_style,
                    target_style=target_style,
                    context=context
                )
                
                if conversion_result['conversion_applied']:
                    successful_conversions += 1
                    confidence_scores.append(conversion_result['confidence'])
                
                # 检查风格检测准确性
                detected_style = self._detect_instruction_style(conversion_result['converted_instruction'])
                style_detection_total += 1
                if detected_style == target_style:
                    style_detection_correct += 1
        
        # 计算指标
        if total_conversions > 0:
            quality_metrics['conversion_success_rate'] = successful_conversions / total_conversions
            quality_metrics['confidence_scores'] = confidence_scores
            quality_metrics['avg_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
        
        if style_detection_total > 0:
            quality_metrics['style_detection_accuracy'] = style_detection_correct / style_detection_total
        
        return quality_metrics
    
    def _evaluate_style_consistency(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估风格一致性"""
        self.logger.info("评估风格一致性...")
        
        consistency_metrics = {
            'style_consistency_scores': [],
            'vocabulary_consistency': 0.0,
            'syntactic_consistency': 0.0,
            'semantic_consistency': 0.0
        }
        
        style_consistency_scores = []
        vocabulary_scores = []
        syntactic_scores = []
        semantic_scores = []
        
        for item in test_data:
            instruction = item.get('instruction', '')
            source_style = item.get('source_style', 'basic')
            target_style = item.get('target_style', 'basic')
            context = item.get('context', {})
            
            if source_style != target_style:
                # 进行风格转换
                conversion_result = self.style_converter.convert_instruction(
                    instruction=instruction,
                    source_style=source_style,
                    target_style=target_style,
                    context=context
                )
                
                if conversion_result['conversion_applied']:
                    converted_instruction = conversion_result['converted_instruction']
                    
                    # 计算风格一致性分数
                    consistency_score = self._calculate_style_consistency_score(
                        converted_instruction, target_style
                    )
                    style_consistency_scores.append(consistency_score)
                    
                    # 计算词汇一致性
                    vocab_score = self._calculate_vocabulary_consistency(
                        converted_instruction, target_style
                    )
                    vocabulary_scores.append(vocab_score)
                    
                    # 计算句法一致性
                    syntactic_score = self._calculate_syntactic_consistency(
                        converted_instruction, target_style
                    )
                    syntactic_scores.append(syntactic_score)
                    
                    # 计算语义一致性
                    semantic_score = self._calculate_semantic_consistency(
                        instruction, converted_instruction
                    )
                    semantic_scores.append(semantic_score)
        
        # 计算平均分数
        consistency_metrics['style_consistency_scores'] = style_consistency_scores
        consistency_metrics['vocabulary_consistency'] = np.mean(vocabulary_scores) if vocabulary_scores else 0.0
        consistency_metrics['syntactic_consistency'] = np.mean(syntactic_scores) if syntactic_scores else 0.0
        consistency_metrics['semantic_consistency'] = np.mean(semantic_scores) if semantic_scores else 0.0
        
        return consistency_metrics
    
    def _evaluate_navigation_performance(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估导航性能"""
        self.logger.info("评估导航性能...")
        
        performance_metrics = {
            'navigation_accuracy': 0.0,
            'success_rate': 0.0,
            'path_efficiency': 0.0,
            'style_adaptation_benefit': 0.0
        }
        
        # 这里应该与实际的导航模型集成
        # 简化实现：基于指令质量评估导航性能
        
        navigation_scores = []
        success_rates = []
        
        for item in test_data:
            instruction = item.get('instruction', '')
            source_style = item.get('source_style', 'basic')
            target_style = item.get('target_style', 'basic')
            context = item.get('context', {})
            
            # 模拟导航性能评估
            if source_style != target_style:
                # 转换指令
                conversion_result = self.style_converter.convert_instruction(
                    instruction=instruction,
                    source_style=source_style,
                    target_style=target_style,
                    context=context
                )
                
                if conversion_result['conversion_applied']:
                    # 基于转换质量评估导航性能
                    navigation_score = self._estimate_navigation_performance(
                        conversion_result['converted_instruction'],
                        conversion_result['confidence']
                    )
                    navigation_scores.append(navigation_score)
                    
                    # 模拟成功率
                    success_rate = min(conversion_result['confidence'], 0.9)
                    success_rates.append(success_rate)
        
        if navigation_scores:
            performance_metrics['navigation_accuracy'] = np.mean(navigation_scores)
            performance_metrics['success_rate'] = np.mean(success_rates)
        
        return performance_metrics
    
    def _detect_instruction_style(self, instruction: str) -> str:
        """检测指令风格"""
        instruction_lower = instruction.lower()
        
        # Scene风格检测
        scene_indicators = ['proceed', 'execute', 'maintain', 'precisely', 'circumvent']
        scene_score = sum(1 for indicator in scene_indicators if indicator in instruction_lower)
        
        # User风格检测
        user_indicators = ['please', 'could you', 'thanks', 'awesome', 'cool', 'darling']
        user_score = sum(1 for indicator in user_indicators if indicator in instruction_lower)
        
        # Basic风格检测
        basic_indicators = ['go', 'turn', 'walk', 'stop', 'left', 'right', 'forward']
        basic_score = sum(1 for indicator in basic_indicators if indicator in instruction_lower)
        
        scores = {'scene': scene_score, 'user': user_score, 'basic': basic_score}
        return max(scores, key=scores.get)
    
    def _calculate_style_consistency_score(self, instruction: str, target_style: str) -> float:
        """计算风格一致性分数"""
        instruction_lower = instruction.lower()
        
        if target_style == 'basic':
            basic_indicators = ['go', 'turn', 'walk', 'stop', 'left', 'right', 'forward']
            basic_count = sum(1 for indicator in basic_indicators if indicator in instruction_lower)
            return min(basic_count / 3.0, 1.0)
        
        elif target_style == 'scene':
            scene_indicators = ['proceed', 'execute', 'maintain', 'precisely', 'circumvent']
            scene_count = sum(1 for indicator in scene_indicators if indicator in instruction_lower)
            return min(scene_count / 2.0, 1.0)
        
        elif target_style == 'user':
            user_indicators = ['please', 'could you', 'thanks', 'awesome', 'cool']
            user_count = sum(1 for indicator in user_indicators if indicator in instruction_lower)
            return min(user_count / 2.0, 1.0)
        
        return 0.5
    
    def _calculate_vocabulary_consistency(self, instruction: str, target_style: str) -> float:
        """计算词汇一致性"""
        instruction_lower = instruction.lower()
        
        # 定义风格特定的词汇
        style_vocabularies = {
            'basic': ['go', 'turn', 'walk', 'stop', 'left', 'right', 'forward'],
            'scene': ['proceed', 'execute', 'maintain', 'precisely', 'circumvent'],
            'user': ['please', 'could you', 'thanks', 'awesome', 'cool', 'darling']
        }
        
        target_vocab = style_vocabularies.get(target_style, [])
        if not target_vocab:
            return 0.5
        
        # 计算目标风格词汇的匹配度
        matches = sum(1 for word in target_vocab if word in instruction_lower)
        return min(matches / len(target_vocab), 1.0)
    
    def _calculate_syntactic_consistency(self, instruction: str, target_style: str) -> float:
        """计算句法一致性"""
        # 简化的句法一致性计算
        word_count = len(instruction.split())
        
        if target_style == 'basic':
            # Basic风格通常较短
            return 1.0 - min(word_count / 10.0, 0.5)
        elif target_style == 'scene':
            # Scene风格通常较长
            return min(word_count / 15.0, 1.0)
        elif target_style == 'user':
            # User风格长度适中
            return 1.0 - abs(word_count - 8) / 10.0
        
        return 0.5
    
    def _calculate_semantic_consistency(self, original: str, converted: str) -> float:
        """计算语义一致性"""
        # 简化的语义一致性计算
        original_words = set(original.lower().split())
        converted_words = set(converted.lower().split())
        
        intersection = original_words.intersection(converted_words)
        union = original_words.union(converted_words)
        
        return len(intersection) / max(len(union), 1)
    
    def _estimate_navigation_performance(self, instruction: str, confidence: float) -> float:
        """估计导航性能"""
        # 基于指令质量和转换置信度估计导航性能
        instruction_quality = self._assess_instruction_quality(instruction)
        return (instruction_quality + confidence) / 2.0
    
    def _assess_instruction_quality(self, instruction: str) -> float:
        """评估指令质量"""
        # 简化的指令质量评估
        word_count = len(instruction.split())
        
        # 长度适中得分更高
        length_score = 1.0 - abs(word_count - 8) / 10.0
        
        # 包含导航关键词得分更高
        nav_keywords = ['go', 'turn', 'walk', 'stop', 'left', 'right', 'forward', 'back']
        keyword_score = sum(1 for keyword in nav_keywords if keyword in instruction.lower()) / len(nav_keywords)
        
        return (length_score + keyword_score) / 2.0
    
    def _calculate_overall_metrics(self):
        """计算总体指标"""
        overall_metrics = {
            'conversion_quality_score': 0.0,
            'style_consistency_score': 0.0,
            'navigation_performance_score': 0.0,
            'overall_score': 0.0
        }
        
        # 转换质量分数
        if 'conversion_quality' in self.evaluation_results:
            conv_quality = self.evaluation_results['conversion_quality']
            overall_metrics['conversion_quality_score'] = conv_quality.get('conversion_success_rate', 0.0)
        
        # 风格一致性分数
        if 'style_consistency' in self.evaluation_results:
            style_consistency = self.evaluation_results['style_consistency']
            overall_metrics['style_consistency_score'] = style_consistency.get('vocabulary_consistency', 0.0)
        
        # 导航性能分数
        if 'navigation_performance' in self.evaluation_results:
            nav_performance = self.evaluation_results['navigation_performance']
            overall_metrics['navigation_performance_score'] = nav_performance.get('navigation_accuracy', 0.0)
        
        # 总体分数
        scores = [
            overall_metrics['conversion_quality_score'],
            overall_metrics['style_consistency_score'],
            overall_metrics['navigation_performance_score']
        ]
        overall_metrics['overall_score'] = np.mean(scores)
        
        self.evaluation_results['overall_metrics'] = overall_metrics
    
    def _save_evaluation_results(self):
        """保存评估结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        if self.config.save_detailed_results:
            results_file = output_dir / "evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        # 保存图表
        if self.config.save_plots:
            self._create_evaluation_plots(output_dir)
        
        self.logger.info(f"评估结果已保存到: {output_dir}")
    
    def _create_evaluation_plots(self, output_dir: Path):
        """创建评估图表"""
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. 转换质量分布图
        if 'conversion_quality' in self.evaluation_results:
            self._plot_conversion_quality(output_dir)
        
        # 2. 风格一致性对比图
        if 'style_consistency' in self.evaluation_results:
            self._plot_style_consistency(output_dir)
        
        # 3. 总体性能雷达图
        self._plot_overall_performance(output_dir)
    
    def _plot_conversion_quality(self, output_dir: Path):
        """绘制转换质量分布图"""
        conv_quality = self.evaluation_results['conversion_quality']
        confidence_scores = conv_quality.get('confidence_scores', [])
        
        if confidence_scores:
            plt.figure(figsize=(10, 6))
            plt.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Style Conversion Confidence Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'conversion_confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_style_consistency(self, output_dir: Path):
        """绘制风格一致性对比图"""
        style_consistency = self.evaluation_results['style_consistency']
        
        metrics = ['vocabulary_consistency', 'syntactic_consistency', 'semantic_consistency']
        values = [style_consistency.get(metric, 0.0) for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['lightcoral', 'lightgreen', 'lightblue'])
        plt.ylabel('Consistency Score')
        plt.title('Style Consistency Metrics')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'style_consistency_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_performance(self, output_dir: Path):
        """绘制总体性能雷达图"""
        overall_metrics = self.evaluation_results['overall_metrics']
        
        categories = ['Conversion Quality', 'Style Consistency', 'Navigation Performance']
        values = [
            overall_metrics['conversion_quality_score'],
            overall_metrics['style_consistency_score'],
            overall_metrics['navigation_performance_score']
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Radar Chart', size=16, pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

class StyleConversionTestDataGenerator:
    """风格转换测试数据生成器"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_test_data(self) -> List[Dict[str, Any]]:
        """生成测试数据"""
        self.logger.info(f"生成 {self.config.test_data_size} 个测试样本...")
        
        test_data = []
        
        # 按风格分布生成数据
        for style, ratio in self.config.style_distribution.items():
            num_samples = int(self.config.test_data_size * ratio)
            style_data = self._generate_style_specific_data(style, num_samples)
            test_data.extend(style_data)
        
        # 打乱数据
        random.shuffle(test_data)
        
        self.logger.info(f"生成了 {len(test_data)} 个测试样本")
        return test_data
    
    def _generate_style_specific_data(self, style: str, num_samples: int) -> List[Dict[str, Any]]:
        """生成特定风格的测试数据"""
        data = []
        
        if style == 'basic':
            data = self._generate_basic_style_data(num_samples)
        elif style == 'scene':
            data = self._generate_scene_style_data(num_samples)
        elif style == 'user':
            data = self._generate_user_style_data(num_samples)
        
        return data
    
    def _generate_basic_style_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成Basic风格数据"""
        templates = [
            "Go to the {location}",
            "Turn {direction}",
            "Walk {direction}",
            "Stop at the {object}",
            "Go around the {object}"
        ]
        
        locations = ['kitchen', 'bathroom', 'bedroom', 'living room']
        directions = ['left', 'right', 'forward', 'backward']
        objects = ['door', 'window', 'table', 'chair']
        
        data = []
        for _ in range(num_samples):
            template = random.choice(templates)
            instruction = template.format(
                location=random.choice(locations),
                direction=random.choice(directions),
                object=random.choice(objects)
            )
            
            data.append({
                'instruction': instruction,
                'source_style': 'basic',
                'target_style': random.choice(['scene', 'user']),
                'context': {'scene_type': 'office', 'user_type': 'child'}
            })
        
        return data
    
    def _generate_scene_style_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成Scene风格数据"""
        templates = [
            "Proceed forward, circumventing the {object} with the utmost care",
            "Execute a {direction} rotation while maintaining a steadfast course",
            "Navigate through the {location} with precision",
            "Halt precisely at the {object} and await further instructions"
        ]
        
        locations = ['corridor', 'chamber', 'facility', 'area']
        directions = ['leftward', 'rightward', 'clockwise', 'counterclockwise']
        objects = ['workstation', 'equipment', 'furniture', 'obstacle']
        
        data = []
        for _ in range(num_samples):
            template = random.choice(templates)
            instruction = template.format(
                location=random.choice(locations),
                direction=random.choice(directions),
                object=random.choice(objects)
            )
            
            data.append({
                'instruction': instruction,
                'source_style': 'scene',
                'target_style': 'basic',
                'context': {'scene_type': 'office'}
            })
        
        return data
    
    def _generate_user_style_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成User风格数据"""
        user_templates = {
            'child': [
                "Hey, could you go to the {location} please?",
                "Can you walk to the {object}? Thanks!",
                "Please turn {direction}, that would be awesome!"
            ],
            'keith': [
                "Kindly proceed to the {location}",
                "Please navigate to the {object} with precision",
                "Execute a {direction} turn when convenient"
            ],
            'moira': [
                "Darling, could you go to the {location} please?",
                "Sweetheart, walk to the {object} for me",
                "Dear, turn {direction} when you're ready"
            ]
        }
        
        locations = ['kitchen', 'bathroom', 'bedroom', 'living room']
        objects = ['door', 'window', 'table', 'chair']
        directions = ['left', 'right', 'around']
        
        data = []
        for _ in range(num_samples):
            user_type = random.choice(list(user_templates.keys()))
            template = random.choice(user_templates[user_type])
            instruction = template.format(
                location=random.choice(locations),
                object=random.choice(objects),
                direction=random.choice(directions)
            )
            
            data.append({
                'instruction': instruction,
                'source_style': 'user',
                'target_style': 'basic',
                'context': {'user_type': user_type}
            })
        
        return data

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化配置
    config = EvaluationConfig(
        test_data_size=500,
        output_dir="evaluation_results"
    )
    
    # 生成测试数据
    test_data_generator = StyleConversionTestDataGenerator(config)
    test_data = test_data_generator.generate_test_data()
    
    # 创建评估器
    evaluator = StyleConversionEvaluator(config)
    
    # 进行评估
    results = evaluator.evaluate_style_conversion(test_data)
    
    print("评估完成!")
    print(f"总体分数: {results['overall_metrics']['overall_score']:.3f}")
    print(f"转换质量: {results['overall_metrics']['conversion_quality_score']:.3f}")
    print(f"风格一致性: {results['overall_metrics']['style_consistency_score']:.3f}")
    print(f"导航性能: {results['overall_metrics']['navigation_performance_score']:.3f}")
