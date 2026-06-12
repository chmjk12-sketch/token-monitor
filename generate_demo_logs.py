"""
generate_demo_logs.py — 生成演示用的 api_logs.jsonl 数据（新 schema：真实成本 + 缓存 + datetime）
运行：python3 generate_demo_logs.py
"""

import json
import random
import time
from datetime import datetime, timedelta

LOG_FILE = "api_logs.jsonl"

# 更多 Agent，覆盖 workbuddy、trae 和各种 agent
AGENTS = [
    "workbuddy",
    "trae",
    "agent_search",
    "agent_summary",
    "agent_chat",
    "agent_code_review",
    "agent_translate",
    "agent_data_analysis",
    "agent_doc_qa",
    "agent_email_writer",
    "agent_meeting_minutes",
    "agent_knowledge_base",
    "agent_sql_generator",
    "agent_bug_hunter",
    "agent_test_writer",
]

# 支持真实定价的模型
MODELS = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
]

# 模型定价（元/百万 tokens）
PRICING = {
    "deepseek/deepseek-chat": {
        "input_cache_hit": 0.5,
        "input_cache_miss": 2.0,
        "output": 8.0,
        "currency": "CNY",
    },
    "deepseek/deepseek-reasoner": {
        "input_cache_hit": 1.0,
        "input_cache_miss": 4.0,
        "output": 16.0,
        "currency": "CNY",
    },
    "openai/gpt-4o": {"input": 2.5, "output": 10.0, "currency": "USD"},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6, "currency": "USD"},
    "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0, "currency": "USD"},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25, "currency": "USD"},
}

USD_TO_CNY = 7.2

random.seed(42)

records = []
now = time.time()


def calc_cost(model, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens):
    pricing = PRICING[model]
    currency = pricing["currency"]

    if "input_cache_hit" in pricing:
        hit = cache_hit_tokens or 0
        miss = cache_miss_tokens or max(0, prompt_tokens - hit)
        input_cost = (hit * pricing["input_cache_hit"] + miss * pricing["input_cache_miss"]) / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        total = input_cost + output_cost
        if currency == "CNY":
            return round(total, 6), round(total / USD_TO_CNY, 6), "CNY"
        else:
            return round(total * USD_TO_CNY, 6), round(total, 6), "USD"
    else:
        input_cost = prompt_tokens * pricing["input"] / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        total = input_cost + output_cost
        if currency == "USD":
            return round(total * USD_TO_CNY, 6), round(total, 6), "USD"
        else:
            return round(total, 6), round(total / USD_TO_CNY, 6), "CNY"


# 生成过去 30 天内的 2000 条记录
for i in range(2000):
    # 时间分布：最近更密集，30 天内
    offset_hours = random.expovariate(1 / 48)  # 指数分布，平均 48 小时
    offset_hours = min(offset_hours, 30 * 24)
    timestamp = now - offset_hours * 3600

    agent_id = random.choice(AGENTS)
    model = random.choice(MODELS)

    # Token 数量：不同 agent 有不同规模
    base_tokens = {
        "workbuddy": (800, 4000),
        "trae": (500, 3000),
        "agent_search": (500, 2500),
        "agent_summary": (2000, 10000),
        "agent_chat": (300, 2000),
        "agent_code_review": (3000, 15000),
        "agent_translate": (200, 1200),
        "agent_data_analysis": (4000, 20000),
        "agent_doc_qa": (1500, 6000),
        "agent_email_writer": (400, 2500),
        "agent_meeting_minutes": (1000, 5000),
        "agent_knowledge_base": (2000, 8000),
        "agent_sql_generator": (800, 4000),
        "agent_bug_hunter": (2000, 10000),
        "agent_test_writer": (1500, 7000),
    }
    min_tok, max_tok = base_tokens.get(agent_id, (500, 3000))
    total_tokens = random.randint(min_tok, max_tok)
    prompt_ratio = random.uniform(0.3, 0.7)
    prompt_tokens = int(total_tokens * prompt_ratio)
    completion_tokens = total_tokens - prompt_tokens

    # DeepSeek 缓存模拟
    if "deepseek" in model:
        cache_hit_ratio = random.uniform(0.0, 0.6)
        cache_hit_tokens = int(prompt_tokens * cache_hit_ratio)
        cache_miss_tokens = prompt_tokens - cache_hit_tokens
    else:
        cache_hit_tokens = 0
        cache_miss_tokens = 0

    latency = random.uniform(0.5, 5.0) + total_tokens / 5000
    status = "success" if random.random() > 0.03 else "error"

    cost_cny, cost_usd, currency = calc_cost(
        model, prompt_tokens, completion_tokens if status == "success" else 0,
        cache_hit_tokens, cache_miss_tokens
    )

    dt = datetime.fromtimestamp(timestamp)

    record = {
        "call_id": f"demo_{i:05d}",
        "agent_id": agent_id,
        "model": model,
        "timestamp": timestamp,
        "datetime": dt.isoformat(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens if status == "success" else 0,
        "total_tokens": total_tokens if status == "success" else 0,
        "cache_hit_tokens": cache_hit_tokens if status == "success" else 0,
        "cache_miss_tokens": cache_miss_tokens if status == "success" else 0,
        "latency_seconds": round(latency, 4) if status == "success" else round(random.uniform(0.1, 2.0), 4),
        "cost_cny": cost_cny if status == "success" else 0.0,
        "cost_usd": cost_usd if status == "success" else 0.0,
        "cost_currency": currency,
        "status": status,
        "error": None if status == "success" else "Rate limit exceeded",
    }
    records.append(record)

# 按时间排序
records.sort(key=lambda r: r["timestamp"])

with open(LOG_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"已生成 {len(records)} 条演示日志到 {LOG_FILE}")
print(f"时间范围：{datetime.fromtimestamp(records[0]['timestamp'])} ~ {datetime.fromtimestamp(records[-1]['timestamp'])}")

# 统计
agent_counts = {}
agent_costs = {}
for r in records:
    agent_counts[r["agent_id"]] = agent_counts.get(r["agent_id"], 0) + r["total_tokens"]
    agent_costs[r["agent_id"]] = agent_costs.get(r["agent_id"], 0) + r["cost_cny"]

print("\nAgent Token 消耗排行：")
for agent, tokens in sorted(agent_counts.items(), key=lambda x: -x[1]):
    print(f"  {agent}: {tokens:,} tokens  ¥{agent_costs[agent]:.2f}")

model_counts = {}
for r in records:
    model_counts[r["model"]] = model_counts.get(r["model"], 0) + r["total_tokens"]
print("\n模型 Token 消耗排行：")
for model, tokens in sorted(model_counts.items(), key=lambda x: -x[1]):
    print(f"  {model}: {tokens:,} tokens")
