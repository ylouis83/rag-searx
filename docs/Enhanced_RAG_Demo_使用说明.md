(venv) louisliu@louisliudeMacBook-Air scripts % ./run_video_rag.sh qa
🎬 启动RAGFlow视频召回与生成系统

📦 激活Python虚拟环境...
✅ 虚拟环境已激活

🎯 视频RAG系统功能菜单:
   1. 完整演示 (Schema + 存储 + 搜索 + 问答 + 提示工程)
   2. 仅元数据Schema演示
   3. 仅智能问答演示
   4. 仅提示工程演示
   5. 交互式问答模式


🔥 启动模式: qa
⚡ 技术栈: RAGFlow + BGE + Milvus + Qwen + 提示工程

/Users/louisliu/.cursor/rag-searx/venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(

╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎬 RAGFlow视频召回与生成演示系统 🎬                        ║
║                                                                              ║
║  🔥 核心功能:                                                                ║
║     • YouTube视频元数据结构化存储                                            ║
║     • 基于BGE模型的语义向量化                                                ║
║     • Milvus向量数据库高效检索                                               ║
║     • Qwen大模型智能生成 + 视频推荐                                          ║
║                                                                              ║
║  🚀 技术栈: RAGFlow + BGE + Milvus + Qwen + 提示工程                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

⚠️  跳过存储演示，使用现有数据...
🚀 初始化视频RAG流水线...
🧮 初始化BGE嵌入模型...
🚀 初始化BGE嵌入模型
   模型: BAAI/bge-large-zh-v1.5
   设备: mps
   维度: 1024
   最大长度: 512
✅ BGE嵌入模型初始化完成
🔗 连接Milvus数据库...
✅ Milvus连接成功
📋 集合 video_rag_collection 已存在
✅ 集合已在内存中

================================================================================
🤖 视频智能问答演示
================================================================================
🎯 欢迎使用视频智能问答系统!
💡 提示: 输入 'quit' 退出，输入 'help' 查看帮助
📝 您可以询问关于糖尿病的任何问题

❓ 请输入您的问题 (第1个): 2型糖尿病怎么治？？

🔍 正在分析您的第1个问题...
============================================================

🚀 开始视频RAG查询流程
❓ 用户问题: 2型糖尿病怎么治？？
================================================================================
🔍 搜索查询: 2型糖尿病怎么治？？
📦 正在加载BGE模型...
✅ BGE模型加载完成，耗时: 11.21秒
   实际维度: 1024
🧮 查询向量化完成，耗时: 12.705秒
🔍 向量搜索完成，耗时: 0.219秒
✅ 找到 3 个相关结果
   最高相似度: 0.7270
   平均相似度: 0.6542
🤖 开始生成视频推荐回答...
📝 提示词长度: 1180 字符
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
✅ 回答生成完成，耗时: 9.81秒

================================================================================
🎯 视频RAG查询完成
================================================================================
💬 问题: 2型糖尿病怎么治？？
📖 回答: 治疗2型糖尿病主要依赖于生活方式的调整和必要的药物帮助。首先，保持健康的饮食习惯非常重要，通过合理安排饮食可以有效控制血糖水平。其次，定期进行体育锻炼也很关键，运动可以帮助减轻体重，同时提高身体对胰岛素的敏感性，从而更好地管理血糖。

如果仅靠饮食和运动还不足以控制血糖，医生可能会建议使用一些口服药物或者胰岛素注射来辅助治疗。这些方法都是为了让你的血糖保持在一个健康范围内。

如果您想更直观地了解这个过程，可以观看下面这个非常棒的科普视频：[了解 2 型糖尿病 (Understanding Type 2 Diabetes Mellitus)](https://www.youtube.com/watch?v=au-w0QXB6jg)。

请注意，本回答仅供参考，不能替代专业医疗建议。如有健康问题，请及时咨询医生。

📊 统计信息:
   ⏱️  总耗时: 22.75秒
   🎬 检索视频: 1个
   🧩 检索片段: 3个
   📈 平均相似度: 0.6542
   🚀 最高相似度: 0.7270

💡 AI智能回答:
----------------------------------------
治疗2型糖尿病主要依赖于生活方式的调整和必要的药物帮助。首先，保持健康的饮食习惯非常重要，通过合理安排饮食可以有效控制血糖水平。其次，定期进行体育锻炼也很关键，运动可以帮助减轻体重，同时提高身体对胰岛素的敏感性，从而更好地管理血糖。

如果仅靠饮食和运动还不足以控制血糖，医生可能会建议使用一些口服药物或者胰岛素注射来辅助治疗。这些方法都是为了让你的血糖保持在一个健康范围内。

如果您想更直观地了解这个过程，可以观看下面这个非常棒的科普视频：[了解 2 型糖尿病 (Understanding Type 2 Diabetes Mellitus)](https://www.youtube.com/watch?v=au-w0QXB6jg)。

请注意，本回答仅供参考，不能替代专业医疗建议。如有健康问题，请及时咨询医生。
----------------------------------------

📊 查询统计:
   ⏱️  总耗时: 22.75秒
   🎬 检索视频: 1个
   🧩 检索片段: 3个
   📈 最高相似度: 0.7270

📊 问答统计: 已回答 1 个问题