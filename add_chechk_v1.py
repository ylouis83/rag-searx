import os
import pandas as pd
import openai
import json
import time

# --- 1. 配置与准备工作 (已更新为阿里云百炼DashScope) ---

# DashScope 的 API Key。请勿将Key硬编码在代码中。
# 我们将从环境变量 'DASHSCOPE_API_KEY' 中读取。
# 您提供的Key仅作示例，实际运行时请使用您自己的有效Key。
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    # 如果环境变量中没有，作为后备方案使用您提供的示例Key（不推荐在生产环境中使用）
    raise ValueError("未找到环境变量 DASHSCOPE_API_KEY。请先设置您的API密钥。")

# DashScope 的模型名称和兼容模式API入口
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# 初始化客户端，指向DashScope
client = openai.OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)

# --- 2. 模拟原始数据 (与之前相同) ---
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'address': [
        '幸福小区1号楼', '中关村大街1号', '1201室', '1栋1201', '朝阳区',
        '创业路', '2期9-27', '上海市', '和平饭店', '融创凡尔赛领馆2期9-27-1'
    ],
    'refer': [
        '1单元101室', '中关村大街1号院', '1201室', '1栋1201', '北京市朝阳区霄云路',
        '创业路孵化器A座', '-19-27-1', '上海市南京东路20号', '南京东路20号','9-27-1'
    ]
}
df = pd.DataFrame(data)
df['full_address'] = df['address'] + df['refer']


# --- 3. 确定性规则引擎 (与之前完全相同) ---
def apply_deterministic_rules(row):
    address = str(row['address']).strip()
    refer = str(row['refer']).strip()
    if not address or not refer:
        return 'Normal', '地址或补充信息为空'
    if address == refer:
        return 'Duplicate', 'Rule: refer与address完全相同'
    if len(address) > 1 and refer.startswith(address):
        return 'Duplicate', f'Rule: refer以"{address}"开头'
    full_address = address + refer
    n = len(full_address)
    if n > 2 and n % 2 == 0:
        part1 = full_address[0:n//2]
        part2 = full_address[n//2:]
        if part1 == part2 and part1 == address:
             return 'Duplicate', 'Rule: 拼接后为完美重复模式'
    return 'Needs LLM', '无明显规则匹配'


# --- 4. LLM 分析器 (适配DashScope) ---
def analyze_with_llm(address, refer):
    """
    使用 DashScope LLM 分析地址是否重复。
    """
    # Prompt保持不变，其设计是模型无关的
    prompt = f"""
# 角色与任务
你是一个地址数据清洗专家。你的任务是判断一个由 `address` 和 `refer` 拼接的地址是否存在“异常重复”。

# “异常重复”规则
1. **语义重复**: `refer` 在语义上重复了 `address` 的信息，例如 `address`="上海市", `refer`="上海市黄浦区"。
2. **格式错乱重复**: `address` 和 `refer` 的内容有重叠，但格式混乱，例如 `address`="2期9-27", `refer`="-19-27-1"。
3. **部分包含**: `refer` 中包含了 `address` 的核心信息。

# 输出格式
你的回答必须是且只能是一个合法的JSON对象，不要有任何其他多余的文字或解释。JSON包含三个字段:
- `is_duplicate`: 布尔值(true/false)。
- `reason`: 字符串，解释判断理由。
- `corrected_address`: 字符串，如果重复，给出修正建议，否则返回原始拼接地址。

# 学习案例
- 输入: {{"address": "幸福小区1号楼", "refer": "1单元101室"}}
- 输出: {{"is_duplicate": false, "reason": "正常补充信息，无重复。", "corrected_address": "幸福小区1号楼1单元101室"}}
- 输入: {{"address": "朝阳区", "refer": "北京市朝阳区霄云路"}}
- 输出: {{"is_duplicate": true, "reason": "refer中已包含address的'朝阳区'信息。", "corrected_address": "北京市朝阳区霄云路"}}

# --- 现在，请分析以下地址 ---
输入: {{"address": "{address}", "refer": "{refer}"}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        response_content = response.choices[0].message.content
        # 尝试解析LLM返回的字符串为JSON
        result_json = json.loads(response_content)
        
        if result_json.get('is_duplicate'):
            return 'Duplicate', result_json.get('reason', 'LLM判断重复'), result_json.get('corrected_address', '')
        else:
            return 'Normal', result_json.get('reason', 'LLM判断正常'), result_json.get('corrected_address', '')

    except Exception as e:
        return 'Error', f'LLM API调用或解析失败: {str(e)}', ''

# --- 5. 主处理流程 (与之前完全相同) ---
def process_addresses(df):
    results = []
    total_rows = len(df)
    print(f"开始使用阿里云百炼模型 {MODEL_NAME} 处理地址数据...")
    for index, row in df.iterrows():
        print(f"处理中... {index + 1}/{total_rows}", end='\r')
        rule_class, rule_reason = apply_deterministic_rules(row)
        if rule_class == 'Needs LLM':
            # time.sleep(0.2) # 如果需要控制QPS，可以取消注释
            llm_class, llm_reason, corrected = analyze_with_llm(row['address'], row['refer'])
            results.append({
                'classification': llm_class,
                'source': f'LLM ({MODEL_NAME})',
                'reason': llm_reason,
                'corrected_address': corrected
            })
        else:
            corrected = row['address'] if rule_class == 'Duplicate' else row['full_address']
            results.append({
                'classification': rule_class,
                'source': 'Rule',
                'reason': rule_reason,
                'corrected_address': corrected
            })
    print("\n处理完成！")
    result_df = pd.DataFrame(results)
    return pd.concat([df, result_df], axis=1)

# --- 运行并展示结果 ---
if __name__ == "__main__":
    final_df = process_addresses(df)
    print("\n--- 处理结果分析 ---")
    print(final_df.to_markdown(index=False))
