#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拼音向量搜索演示
解决语音识别中发音差异导致的搜索问题

使用场景：
- 语音识别结果中的人名可能因发音不标准而错误
- 例如："周萍" 被识别为 "邹萍"
- 通过拼音向量搜索实现统一检索
"""

import os
import sys
import time
import json
from typing import List, Dict, Any

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入拼音向量搜索模块
from ragflow_integration.pinyin_vector_search import (
    PinyinVectorSearchEngine, 
    PinyinNormalizer,
    PinyinSearchResult
)


class PinyinSearchDemo:
    """拼音向量搜索演示类"""
    
    def __init__(self):
        """初始化演示"""
        
        print("🚀 拼音向量搜索演示")
        print("=" * 60)
        
        # 创建搜索引擎
        self.search_engine = PinyinVectorSearchEngine(similarity_threshold=0.6)
        
        # 初始化测试数据
        self._prepare_test_data()
    
    def _prepare_test_data(self):
        """准备测试数据"""
        
        print("\n📚 正在准备测试数据...")
        
        # 常见人名数据（模拟实际应用中的用户数据库）
        self.person_entities = [
            # 测试用例1：周萍 vs 邹萍
            {"entity_id": "emp_001", "entity_name": "周萍", "metadata": {"department": "销售部", "position": "经理"}},
            
            # 测试用例2：张敏 vs 张民
            {"entity_id": "emp_002", "entity_name": "张敏", "metadata": {"department": "技术部", "position": "工程师"}},
            
            # 测试用例3：李华（标准发音）
            {"entity_id": "emp_003", "entity_name": "李华", "metadata": {"department": "人事部", "position": "专员"}},
            
            # 测试用例4：王红 vs 忘红
            {"entity_id": "emp_004", "entity_name": "王红", "metadata": {"department": "财务部", "position": "会计"}},
            
            # 更多测试数据
            {"entity_id": "emp_005", "entity_name": "陈丽", "metadata": {"department": "市场部", "position": "主管"}},
            {"entity_id": "emp_006", "entity_name": "刘敏", "metadata": {"department": "技术部", "position": "架构师"}},
            {"entity_id": "emp_007", "entity_name": "杨华", "metadata": {"department": "运营部", "position": "总监"}},
            {"entity_id": "emp_008", "entity_name": "黄萍", "metadata": {"department": "客服部", "position": "组长"}},
            {"entity_id": "emp_009", "entity_name": "赵敏", "metadata": {"department": "产品部", "position": "经理"}},
            {"entity_id": "emp_010", "entity_name": "吴华", "metadata": {"department": "设计部", "position": "设计师"}},
            
            # 测试相似拼音
            {"entity_id": "emp_011", "entity_name": "周敏", "metadata": {"department": "法务部", "position": "律师"}},
            {"entity_id": "emp_012", "entity_name": "张华", "metadata": {"department": "行政部", "position": "助理"}},
            {"entity_id": "emp_013", "entity_name": "李敏", "metadata": {"department": "采购部", "position": "专员"}},
        ]
        
        # 批量添加实体到搜索引擎
        self.search_engine.batch_add_entities(self.person_entities)
        
        print(f"✅ 已添加 {len(self.person_entities)} 个测试实体")
    
    def demo_basic_search(self):
        """基础搜索演示"""
        
        print("\n" + "="*60)
        print("📝 基础拼音向量搜索演示")
        print("="*60)
        
        # 测试用例：模拟语音识别的错误结果
        test_cases = [
            {
                "query": "邹萍",
                "expected": "周萍",
                "description": "语音识别错误：zh音被识别为z音"
            },
            {
                "query": "张民", 
                "expected": "张敏",
                "description": "语音识别错误：ing韵母被识别为in"
            },
            {
                "query": "李华",
                "expected": "李华", 
                "description": "标准发音，应该精确匹配"
            },
            {
                "query": "忘红",
                "expected": "王红",
                "description": "语音识别错误：w声母被模糊识别"
            },
            {
                "query": "黄凭",
                "expected": "黄萍", 
                "description": "语音识别错误：ping被识别为ping的变体"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 测试用例 {i}: {test_case['query']}")
            print(f"   预期结果: {test_case['expected']}")
            print(f"   测试说明: {test_case['description']}")
            
            # 执行搜索
            start_time = time.time()
            results = self.search_engine.search(test_case['query'], top_k=5)
            search_time = time.time() - start_time
            
            # 显示结果
            if results:
                print(f"   ✅ 找到 {len(results)} 个匹配结果 (耗时: {search_time:.3f}秒)")
                
                for j, result in enumerate(results, 1):
                    match_indicator = "🎯" if result.entity_name == test_case['expected'] else "📄"
                    print(f"      {match_indicator} {j}. {result.entity_name}")
                    print(f"          拼音: {result.pinyin_text}")
                    print(f"          相似度: {result.similarity_score:.3f}")
                    print(f"          匹配类型: {result.match_type}")
                    
                    # 显示元数据
                    entity = self.search_engine.entities[result.entity_id]
                    if entity.metadata:
                        dept = entity.metadata.get('department', '未知')
                        pos = entity.metadata.get('position', '未知')
                        print(f"          部门: {dept}, 职位: {pos}")
                
                # 检查是否找到预期结果
                found_expected = any(r.entity_name == test_case['expected'] for r in results)
                if found_expected:
                    print(f"   ✅ 成功找到预期结果: {test_case['expected']}")
                else:
                    print(f"   ❌ 未找到预期结果: {test_case['expected']}")
            else:
                print(f"   ❌ 未找到任何匹配结果")
    
    def demo_comparison_mode(self):
        """对比模式演示：传统搜索 vs 拼音向量搜索"""
        
        print("\n" + "="*60)
        print("🔄 对比演示：传统搜索 vs 拼音向量搜索")
        print("="*60)
        
        test_queries = [
            "邹萍",  # 错误发音
            "张民",  # 错误发音  
            "忘红",  # 错误发音
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: {query}")
            
            # 1. 模拟传统精确匹配搜索
            print("   📰 传统精确匹配搜索:")
            traditional_results = self._traditional_exact_search(query)
            if traditional_results:
                for result in traditional_results:
                    print(f"      ✅ {result}")
            else:
                print("      ❌ 未找到匹配结果")
            
            # 2. 拼音向量搜索
            print("   🚀 拼音向量搜索:")
            pinyin_results = self.search_engine.search(query, top_k=3)
            if pinyin_results:
                for result in pinyin_results:
                    print(f"      ✅ {result.entity_name} (相似度: {result.similarity_score:.3f})")
            else:
                print("      ❌ 未找到匹配结果")
    
    def _traditional_exact_search(self, query: str) -> List[str]:
        """模拟传统精确匹配搜索"""
        
        matches = []
        for entity_data in self.person_entities:
            entity_name = entity_data['entity_name']
            if query == entity_name or query in entity_name:
                matches.append(entity_name)
        
        return matches
    
    def demo_interactive_mode(self):
        """交互模式演示"""
        
        print("\n" + "="*60)
        print("💬 交互模式演示")
        print("="*60)
        print("提示: 输入人名进行搜索，输入 'quit' 退出")
        print("建议测试: 邹萍、张民、忘红、李华等")
        
        while True:
            try:
                query = input("\n🔍 请输入要搜索的人名: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q', '退出']:
                    print("👋 再见！")
                    break
                
                if not query:
                    continue
                
                # 执行搜索
                print(f"\n正在搜索: {query}")
                start_time = time.time()
                results = self.search_engine.search(query, top_k=5)
                search_time = time.time() - start_time
                
                # 显示结果
                if results:
                    print(f"✅ 找到 {len(results)} 个匹配结果 (耗时: {search_time:.3f}秒)")
                    print()
                    
                    for i, result in enumerate(results, 1):
                        entity = self.search_engine.entities[result.entity_id]
                        dept = entity.metadata.get('department', '未知')
                        pos = entity.metadata.get('position', '未知')
                        
                        print(f"  {i}. {result.entity_name}")
                        print(f"     部门: {dept}")
                        print(f"     职位: {pos}")
                        print(f"     拼音: {result.pinyin_text}")
                        print(f"     相似度: {result.similarity_score:.3f}")
                        print(f"     匹配类型: {result.match_type}")
                        print()
                else:
                    print("❌ 未找到匹配结果")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 搜索出错: {e}")
    
    def demo_pinyin_analysis(self):
        """拼音分析演示"""
        
        print("\n" + "="*60)
        print("🔬 拼音分析演示")
        print("="*60)
        
        normalizer = PinyinNormalizer()
        
        test_names = ["周萍", "张敏", "李华", "王红"]
        
        for name in test_names:
            print(f"\n📝 分析: {name}")
            
            # 标准拼音
            standard_pinyin = normalizer.text_to_pinyin(name)
            print(f"   标准拼音: {standard_pinyin}")
            
            # 拼音变体
            variants = normalizer.generate_pinyin_variants(name)
            print(f"   拼音变体: {variants}")
            
            # 测试相似度计算
            test_inputs = [
                name.replace('周', '邹') if '周' in name else name,
                name.replace('敏', '民') if '敏' in name else name,
                name.replace('王', '忘') if '王' in name else name,
            ]
            
            for test_input in test_inputs:
                if test_input != name:
                    input_pinyin = normalizer.text_to_pinyin(test_input)
                    similarity = normalizer.calculate_pinyin_similarity(standard_pinyin, input_pinyin)
                    print(f"   {test_input} -> {input_pinyin} (相似度: {similarity:.3f})")
    
    def run_full_demo(self):
        """运行完整演示"""
        
        try:
            # 1. 基础搜索演示
            self.demo_basic_search()
            
            # 2. 对比演示
            self.demo_comparison_mode()
            
            # 3. 拼音分析演示
            self.demo_pinyin_analysis()
            
            # 4. 显示统计信息
            self._show_statistics()
            
            # 5. 询问是否进入交互模式
            print("\n" + "="*60)
            response = input("是否进入交互模式？(y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                self.demo_interactive_mode()
            
        except KeyboardInterrupt:
            print("\n👋 演示结束")
        except Exception as e:
            print(f"❌ 演示出错: {e}")
    
    def _show_statistics(self):
        """显示统计信息"""
        
        print("\n" + "="*60)
        print("📊 搜索引擎统计信息")
        print("="*60)
        
        stats = self.search_engine.get_statistics()
        
        print(f"总实体数量: {stats['total_entities']}")
        print(f"向量维度: {stats['vector_dimension']}")
        print(f"相似度阈值: {stats['similarity_threshold']}")
        
        print("\n实体列表:")
        for i, entity_name in enumerate(stats['entity_list'], 1):
            print(f"  {i:2d}. {entity_name}")


def main():
    """主函数"""
    
    print("🎯 拼音向量搜索演示程序")
    print("解决语音识别中发音差异导致的搜索问题")
    print()
    
    try:
        # 创建并运行演示
        demo = PinyinSearchDemo()
        demo.run_full_demo()
        
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 