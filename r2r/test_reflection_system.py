#!/usr/bin/env python3
"""
反思记忆银行系统测试脚本
"""

import sys
import os
import json
import unittest
from unittest.mock import Mock, patch
import tempfile

# 添加路径
sys.path.append(os.path.dirname(__file__))

from reflection_memory_bank import ReflectionMemoryBank, ReflectionExperience, LLMReflectionModule
from reflection_pipeline import ReflectionPipeline
from reflection_config import REFLECTION_CONFIG

class TestReflectionMemoryBank(unittest.TestCase):
    """测试反思记忆银行"""
    
    def setUp(self):
        """设置测试环境"""
        self.memory_bank = ReflectionMemoryBank(max_experiences=100)
        
    def test_add_experience(self):
        """测试添加经验"""
        # 创建模拟经验
        experience = ReflectionExperience(
            scene_type='hallway',
            spatial_context='near_door',
            action_pattern='exploration',
            decision_confidence=0.8,
            spatial_rule='doors_lead_to_rooms',
            navigation_strategy='explore_first',
            success_rate=0.9,
            frequency=1,
            last_updated=1234567890
        )
        
        # 添加经验
        self.memory_bank.experiences.append(experience)
        self.memory_bank._rebuild_indices()
        
        # 验证
        self.assertEqual(len(self.memory_bank.experiences), 1)
        self.assertEqual(self.memory_bank.experiences[0].scene_type, 'hallway')
        
    def test_retrieve_experiences(self):
        """测试检索经验"""
        # 添加多个经验
        experiences = [
            ReflectionExperience('hallway', 'near_door', 'exploration', 0.8, 'rule1', 'strategy1', 0.9, 1, 1234567890),
            ReflectionExperience('kitchen', 'center', 'goal_directed', 0.7, 'rule2', 'strategy2', 0.8, 1, 1234567890),
            ReflectionExperience('hallway', 'corner', 'exploration', 0.6, 'rule3', 'strategy3', 0.7, 1, 1234567890)
        ]
        
        for exp in experiences:
            self.memory_bank.experiences.append(exp)
        self.memory_bank._rebuild_indices()
        
        # 检索相关经验
        relevant = self.memory_bank.retrieve_relevant_experiences('hallway', 'near_door', 'exploration')
        
        # 验证
        self.assertGreater(len(relevant), 0)
        self.assertTrue(any(exp.scene_type == 'hallway' for exp in relevant))
        
    def test_similarity_calculation(self):
        """测试相似度计算"""
        exp1 = ReflectionExperience('hallway', 'near_door', 'exploration', 0.8, 'rule1', 'strategy1', 0.9, 1, 1234567890)
        exp2 = ReflectionExperience('hallway', 'near_door', 'exploration', 0.7, 'rule2', 'strategy2', 0.8, 1, 1234567890)
        exp3 = ReflectionExperience('kitchen', 'center', 'goal_directed', 0.6, 'rule3', 'strategy3', 0.7, 1, 1234567890)
        
        # 计算相似度
        sim12 = self.memory_bank._calculate_similarity(exp1, exp2)
        sim13 = self.memory_bank._calculate_similarity(exp1, exp3)
        
        # 验证
        self.assertGreater(sim12, sim13)  # exp1和exp2应该更相似
        self.assertGreater(sim12, 0.8)     # 相似度应该很高
        
    def test_save_load(self):
        """测试保存和加载"""
        # 添加经验
        experience = ReflectionExperience(
            'hallway', 'near_door', 'exploration', 0.8, 'rule1', 'strategy1', 0.9, 1, 1234567890
        )
        self.memory_bank.experiences.append(experience)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.memory_bank.save_memory_bank(temp_path)
            
            # 创建新的记忆银行并加载
            new_memory_bank = ReflectionMemoryBank(max_experiences=100)
            new_memory_bank.load_memory_bank(temp_path)
            
            # 验证
            self.assertEqual(len(new_memory_bank.experiences), 1)
            self.assertEqual(new_memory_bank.experiences[0].scene_type, 'hallway')
            
        finally:
            os.unlink(temp_path)

class TestReflectionPipeline(unittest.TestCase):
    """测试反思流水线"""
    
    def setUp(self):
        """设置测试环境"""
        self.args = Mock()
        for key, value in REFLECTION_CONFIG.items():
            setattr(self.args, key, value)
        
        self.pipeline = ReflectionPipeline(self.args)
        
    def test_should_reflect(self):
        """测试反思决策"""
        # 测试基于频率的反思
        self.pipeline.episode_count = 10
        self.assertTrue(self.pipeline._should_reflect())
        
        # 测试基于性能的反思
        self.pipeline.episode_count = 5
        self.pipeline.performance_history['success'] = [0, 0, 0, 0, 0]  # 低成功率
        self.assertTrue(self.pipeline._should_reflect())
        
    def test_performance_calculation(self):
        """测试性能计算"""
        # 添加性能历史
        self.pipeline.performance_history['success'] = [1, 1, 0, 1, 1] * 4  # 80%成功率
        self.pipeline.performance_history['spl'] = [0.8, 0.9, 0.7, 0.8, 0.9] * 4
        
        # 计算成功率
        success_rate = self.pipeline._calculate_recent_success_rate()
        self.assertAlmostEqual(success_rate, 0.8, places=2)
        
        # 计算性能
        performance = self.pipeline._calculate_recent_performance()
        self.assertAlmostEqual(performance, 0.82, places=2)
        
    def test_pipeline_stats(self):
        """测试流水线统计"""
        # 设置一些数据
        self.pipeline.episode_count = 100
        self.pipeline.reflection_count = 10
        self.pipeline.experience_application_count = 50
        
        # 添加一些经验
        for i in range(5):
            exp = ReflectionExperience(
                f'scene_{i}', f'context_{i}', f'pattern_{i}', 0.8, f'rule_{i}', f'strategy_{i}', 0.9, 1, 1234567890
            )
            self.pipeline.reflection_memory_bank.experiences.append(exp)
        
        # 获取统计
        stats = self.pipeline.get_pipeline_stats()
        
        # 验证
        self.assertEqual(stats['episode_count'], 100)
        self.assertEqual(stats['reflection_count'], 10)
        self.assertEqual(stats['experience_application_count'], 50)
        self.assertEqual(stats['memory_bank_size'], 5)

class TestLLMReflectionModule(unittest.TestCase):
    """测试LLM反思模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.reflection_module = LLMReflectionModule()
        
    @patch('reflection_memory_bank.time.time')
    def test_reflection_prompt_building(self, mock_time):
        """测试反思提示构建"""
        mock_time.return_value = 1234567890
        
        context_data = {
            'instruction': 'Go to the kitchen',
            'steps': [
                {
                    'cur_vp': 'vp_001',
                    'candidates': [
                        {'vp': 'vp_001', 'is_current_vp': True},
                        {'vp': 'vp_002', 'dist': 2.5, 'direction_text': 'go forward', 'is_current_vp': False}
                    ]
                }
            ],
            'success': 1
        }
        
        prompt = self.reflection_module._build_reflection_prompt(context_data)
        
        # 验证提示包含关键信息
        self.assertIn('Go to the kitchen', prompt)
        self.assertIn('vp_001', prompt)
        self.assertIn('go forward', prompt)
        
    def test_parse_reflection_result(self):
        """测试反思结果解析"""
        llm_response = json.dumps({
            'scene_type': 'kitchen',
            'spatial_context': 'center',
            'action_pattern': 'goal_directed',
            'decision_confidence': 0.8,
            'spatial_rule': 'kitchens_have_appliances',
            'navigation_strategy': 'direct_path',
            'improvement_suggestion': 'Use direct navigation'
        })
        
        context_data = {'success': 1}
        
        experience = self.reflection_module._parse_reflection_result(llm_response, context_data)
        
        # 验证
        self.assertIsNotNone(experience)
        self.assertEqual(experience.scene_type, 'kitchen')
        self.assertEqual(experience.spatial_context, 'center')
        self.assertEqual(experience.decision_confidence, 0.8)

def run_tests():
    """运行所有测试"""
    print("开始运行反思记忆银行系统测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestReflectionMemoryBank,
        TestReflectionPipeline,
        TestLLMReflectionModule
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print(f"\n测试结果:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
