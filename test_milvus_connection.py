#!/usr/bin/env python3
"""
Milvus连接测试脚本
测试Docker中的Milvus服务是否正常工作
"""

import sys
import time
from pymilvus import connections, utility

def test_milvus_connection():
    """测试Milvus连接"""
    print("🔄 开始测试Milvus连接...")
    
    try:
        # 连接到Milvus
        print("📡 连接Milvus服务 (localhost:19530)...")
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus连接成功!")
        
        # 检查服务状态
        print("🔍 检查Milvus服务状态...")
        if utility.get_server_version():
            print(f"✅ Milvus服务版本: {utility.get_server_version()}")
        
        # 列出现有集合
        print("📋 查看现有集合...")
        collections = utility.list_collections()
        if collections:
            print(f"✅ 发现 {len(collections)} 个集合: {collections}")
        else:
            print("ℹ️  暂无集合")
            
        return True
        
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return False
    
    finally:
        try:
            connections.disconnect("default")
            print("🔌 已断开Milvus连接")
        except:
            pass

def test_etcd_minio():
    """测试etcd和MinIO服务"""
    import requests
    
    print("\n🔄 测试依赖服务...")
    
    # 测试MinIO
    try:
        print("📡 测试MinIO (localhost:9000)...")
        response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
        if response.status_code == 200:
            print("✅ MinIO服务正常")
        else:
            print(f"⚠️  MinIO响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ MinIO连接失败: {e}")
    
    # 测试etcd (通过curl模拟)
    import subprocess
    try:
        print("📡 测试etcd (localhost:2379)...")
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:2379/health"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout == "200":
            print("✅ etcd服务正常")
        else:
            print(f"⚠️  etcd响应异常: {result.stdout}")
    except Exception as e:
        print(f"❌ etcd连接失败: {e}")

def main():
    print("=" * 50)
    print("🚀 Milvus Docker服务测试")
    print("=" * 50)
    
    # 等待服务启动
    print("⏳ 等待服务启动...")
    time.sleep(3)
    
    # 测试依赖服务
    test_etcd_minio()
    
    # 测试Milvus
    print("\n" + "=" * 50)
    success = test_milvus_connection()
    print("=" * 50)
    
    if success:
        print("🎉 Milvus服务测试通过!")
        print("✨ 可以开始使用RAG向量搜索功能")
        return 0
    else:
        print("💥 Milvus服务测试失败!")
        print("🔧 请检查Docker容器状态")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 