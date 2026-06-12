"""
generate_demo_logs.py — 生成演示用的 api_logs.jsonl 数据
运行：python3 generate_demo_logs.py
"""

import json
import random
import time
from datetime import datetime, timedelta

LOG_FILE = "api_logs.jsonl"

AGENTS = [
    "agent_search",
    "agent_summary",
    "agent_chat",
    "agent_code_review",
    "agent_translate",
    "agent_data_analysis",
]

MODELS = ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-haiku"]

random.seed(42)

records = []
now = time.time()

# 生成过去 24 小时内的 200 条记录
for i in range(200):
    # 时间分布：最近几小时更密集
    offset_hours = random.expovariate(1 / 8)  # 指数分布，平均 8 小时
    offset_hours = min(offset_hours, 24)
    timestamp = now - offset_hours * 3600

    agent_id = random.choice(AGENTS)
    model = random.choice(MODELS)

    # Token 数量：不同 agent 有不同规模
    base_tokens = {
        "agent_search": (500, 2000),
        "agent_summary": (2000, 8000),
        "agent_chat": (300, 1500),
        "agent_code_review": (3000, 12000),
        "agent_translate": (200, 1000),
        "agent_data_analysis": (4000, 15000),
    }
    min_tok, max_tok = base_tokens.get(agent_id, (500, 3000))
    total_tokens = random.randint(min_tok, max_tok)
    prompt_ratio = random.uniform(0.3, 0.7)
    prompt_tokens = int(total_tokens * prompt_ratio)
    completion_tokens = total_tokens - prompt_tokens

    latency = random.uniform(0.5, 5.0) + total_tokens / 5000

    status = "success" if random.random() > 0.05 else "error"

    record = {
        "call_id": f"demo_{i:04d}",
        "agent_id": agent_id,
        "model": model,
        "timestamp": timestamp,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens if status == "success" else 0,
        "total_tokens": total_tokens if status == "success" else 0,
        "latency_seconds": round(latency, 4) if status == "success" else round(random.uniform(0.1, 2.0), 4),
        "estimated_cost_usd": round(total_tokens / 1_000_000, 6) if status == "success" else 0.0,
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
for r in records:
    agent_counts[r["agent_id"]] = agent_counts.get(r["agent_id"], 0) + r["total_tokens"]
print("\nAgent Token 消耗排行：")
for agent, tokens in sorted(agent_counts.items(), key=lambda x: -x[1]):
    print(f"  {agent}: {tokens:,} tokens")
