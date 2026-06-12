"""
ai_client.py — 统一 AI 调用中间件（支持多模型、真实成本、缓存命中追踪）
所有 Agent 的 AI 接口调用必须经过此模块，实现统一监控与日志记录。

依赖：pip install litellm

支持的模型：
  - DeepSeek: deepseek/deepseek-chat, deepseek/deepseek-reasoner
  - OpenAI:   openai/gpt-4o, openai/gpt-4o-mini
  - Anthropic: anthropic/claude-3-sonnet, anthropic/claude-3-haiku
  - 本地模型: ollama/llama3 等

DeepSeek 定价（2025年2月，按百万 tokens）：
  deepseek-chat:    输入命中 0.5元/M, 输入未命中 2元/M, 输出 8元/M
  deepseek-reasoner: 输入命中 1元/M, 输入未命中 4元/M, 输出 16元/M
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime

import litellm
from litellm import completion

# ---------------------------------------------------------------------------
# 配置区域
# ---------------------------------------------------------------------------

LOG_FILE: Path = Path("api_logs.jsonl")

# 模型真实定价（元/百万 tokens）
# 支持按输入命中/未命中、输出分别定价
MODEL_PRICING: dict[str, dict] = {
    # DeepSeek
    "deepseek/deepseek-chat": {
        "input_cache_hit": 0.5,   # 缓存命中
        "input_cache_miss": 2.0,  # 缓存未命中
        "output": 8.0,
        "currency": "CNY",
    },
    "deepseek/deepseek-reasoner": {
        "input_cache_hit": 1.0,
        "input_cache_miss": 4.0,
        "output": 16.0,
        "currency": "CNY",
    },
    # OpenAI
    "openai/gpt-4o": {
        "input": 2.5,   # USD
        "output": 10.0,
        "currency": "USD",
    },
    "openai/gpt-4o-mini": {
        "input": 0.15,
        "output": 0.6,
        "currency": "USD",
    },
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0,
        "currency": "USD",
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.6,
        "currency": "USD",
    },
    # Anthropic
    "anthropic/claude-3-sonnet": {
        "input": 3.0,
        "output": 15.0,
        "currency": "USD",
    },
    "anthropic/claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
        "currency": "USD",
    },
}

# USD 到 CNY 汇率（可配置）
USD_TO_CNY: float = 7.2

# 默认模型
DEFAULT_MODEL: str = "deepseek/deepseek-chat"

# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _append_log(record: dict[str, Any]) -> None:
    """将一条日志记录追加写入 api_logs.jsonl"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
) -> dict[str, float]:
    """
    计算真实成本。
    返回 {"cost_cny": float, "cost_usd": float, "currency": str}
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING.get("deepseek/deepseek-chat"))
    currency = pricing.get("currency", "CNY")

    # DeepSeek 有缓存命中/未命中区分
    if "input_cache_hit" in pricing:
        hit_tokens = cache_hit_tokens or 0
        miss_tokens = cache_miss_tokens or (prompt_tokens - hit_tokens)
        miss_tokens = max(0, miss_tokens)

        input_cost = (hit_tokens * pricing["input_cache_hit"] +
                      miss_tokens * pricing["input_cache_miss"]) / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        total_cost = input_cost + output_cost

        if currency == "CNY":
            return {"cost_cny": round(total_cost, 6), "cost_usd": round(total_cost / USD_TO_CNY, 6), "currency": "CNY"}
        else:
            return {"cost_cny": round(total_cost * USD_TO_CNY, 6), "cost_usd": round(total_cost, 6), "currency": "USD"}
    else:
        # 普通定价
        input_cost = prompt_tokens * pricing["input"] / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        total_cost = input_cost + output_cost

        if currency == "USD":
            return {"cost_cny": round(total_cost * USD_TO_CNY, 6), "cost_usd": round(total_cost, 6), "currency": "USD"}
        else:
            return {"cost_cny": round(total_cost, 6), "cost_usd": round(total_cost / USD_TO_CNY, 6), "currency": "CNY"}


def _resolve_model(model: str) -> str:
    """解析模型名称，支持简写"""
    if model.startswith("deepseek/") or model.startswith("openai/") or model.startswith("anthropic/"):
        return model
    # 简写映射
    aliases = {
        "deepseek-chat": "deepseek/deepseek-chat",
        "deepseek-reasoner": "deepseek/deepseek-reasoner",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "claude-3-haiku": "anthropic/claude-3-haiku",
    }
    return aliases.get(model, model)


# ---------------------------------------------------------------------------
# 核心：统一调用函数
# ---------------------------------------------------------------------------

def call_ai(
    agent_id: str,
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    统一 AI 调用入口。支持 DeepSeek / OpenAI / Anthropic / 本地模型等。

    参数：
        agent_id    : 必填，标识调用来源（如 "workbuddy", "trae", "agent_search"）
        messages    : 必填，OpenAI 格式的消息列表
        model       : 模型名称，支持简写（如 "deepseek-chat", "gpt-4o"）
        temperature : 生成温度（默认 0.7）
        max_tokens  : 最大生成 token 数（可选）
        **kwargs    : 其他透传给 litellm.completion 的参数

    返回：模型生成的文本内容（str）

    示例：
        from ai_client import call_ai
        reply = call_ai(
            agent_id="workbuddy",
            messages=[{"role": "user", "content": "分析这份文档"}],
            model="deepseek-chat",
        )
    """
    resolved_model = _resolve_model(model)
    call_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    try:
        response = completion(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        latency = time.perf_counter() - start_time

        # 提取 usage
        usage = response.usage or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        # DeepSeek 缓存信息
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

        # 计算真实成本
        cost_info = _calculate_cost(
            resolved_model, prompt_tokens, completion_tokens,
            cache_hit_tokens, cache_miss_tokens
        )

        reply_text = response.choices[0].message.content if response.choices else ""

        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": resolved_model,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": cache_miss_tokens,
            "latency_seconds": round(latency, 4),
            "cost_cny": cost_info["cost_cny"],
            "cost_usd": cost_info["cost_usd"],
            "cost_currency": cost_info["currency"],
            "status": "success",
            "error": None,
        }

        _append_log(log_record)

        print(
            f"[AI] call_id={call_id} agent={agent_id} model={resolved_model} "
            f"tokens={total_tokens} cost=¥{cost_info['cost_cny']:.4f} latency={latency:.2f}s"
        )

        return reply_text

    except Exception as e:
        latency = time.perf_counter() - start_time

        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": resolved_model,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "latency_seconds": round(latency, 4),
            "cost_cny": 0.0,
            "cost_usd": 0.0,
            "cost_currency": "CNY",
            "status": "error",
            "error": str(e),
        }

        _append_log(log_record)
        print(f"[AI] call_id={call_id} agent={agent_id} ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# 异步版本
# ---------------------------------------------------------------------------

async def call_ai_async(
    agent_id: str,
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """异步版本的统一 AI 调用入口"""
    resolved_model = _resolve_model(model)
    call_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    try:
        response = await litellm.acompletion(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        latency = time.perf_counter() - start_time

        usage = response.usage or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

        cost_info = _calculate_cost(
            resolved_model, prompt_tokens, completion_tokens,
            cache_hit_tokens, cache_miss_tokens
        )

        reply_text = response.choices[0].message.content if response.choices else ""

        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": resolved_model,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": cache_miss_tokens,
            "latency_seconds": round(latency, 4),
            "cost_cny": cost_info["cost_cny"],
            "cost_usd": cost_info["cost_usd"],
            "cost_currency": cost_info["currency"],
            "status": "success",
            "error": None,
        }

        _append_log(log_record)

        print(
            f"[AI-async] call_id={call_id} agent={agent_id} model={resolved_model} "
            f"tokens={total_tokens} cost=¥{cost_info['cost_cny']:.4f} latency={latency:.2f}s"
        )

        return reply_text

    except Exception as e:
        latency = time.perf_counter() - start_time

        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": resolved_model,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "latency_seconds": round(latency, 4),
            "cost_cny": 0.0,
            "cost_usd": 0.0,
            "cost_currency": "CNY",
            "status": "error",
            "error": str(e),
        }

        _append_log(log_record)
        print(f"[AI-async] call_id={call_id} agent={agent_id} ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# 配置函数
# ---------------------------------------------------------------------------

def set_log_file(path: str | Path) -> None:
    global LOG_FILE
    LOG_FILE = Path(path)


def set_usd_rate(rate: float) -> None:
    """设置 USD 到 CNY 的汇率"""
    global USD_TO_CNY
    USD_TO_CNY = rate


def add_model_pricing(model: str, pricing: dict) -> None:
    """添加/更新模型定价
    pricing 格式: {"input": 2.5, "output": 10.0, "currency": "USD"}
    或 DeepSeek 格式: {"input_cache_hit": 0.5, "input_cache_miss": 2.0, "output": 8.0, "currency": "CNY"}
    """
    MODEL_PRICING[model] = pricing


def get_stats(agent_id: Optional[str] = None, hours: int = 24) -> dict:
    """获取指定 Agent 或全部 Agent 的统计信息"""
    if not LOG_FILE.exists():
        return {}

    records = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    records.append(r)
                except json.JSONDecodeError:
                    continue

    cutoff = time.time() - hours * 3600
    filtered = [r for r in records if r.get("timestamp", 0) >= cutoff]
    if agent_id:
        filtered = [r for r in filtered if r.get("agent_id") == agent_id]

    if not filtered:
        return {"calls": 0, "tokens": 0, "cost_cny": 0.0, "cost_usd": 0.0}

    return {
        "calls": len(filtered),
        "tokens": sum(r.get("total_tokens", 0) for r in filtered),
        "cost_cny": sum(r.get("cost_cny", 0) for r in filtered),
        "cost_usd": sum(r.get("cost_usd", 0) for r in filtered),
        "avg_latency": sum(r.get("latency_seconds", 0) for r in filtered) / len(filtered),
    }
