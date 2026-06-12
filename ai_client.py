"""
ai_client.py — 统一 AI 调用中间件
所有 Agent 的 AI 接口调用必须经过此模块，实现统一监控与日志记录。
依赖：pip install litellm
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import litellm
from litellm import completion

# ---------------------------------------------------------------------------
# 配置区域（可按需修改）
# ---------------------------------------------------------------------------

# 日志文件路径（JSONL 格式，逐行追加）
LOG_FILE: Path = Path("api_logs.jsonl")

# 成本换算常量：$1 = COST_PER_DOLLAR_TOKENS 个 tokens（默认 100 万）
COST_PER_DOLLAR_TOKENS: float = 1_000_000

# 默认模型（调用方未指定时使用）
DEFAULT_MODEL: str = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 内部：写入一条日志到 JSONL 文件
# ---------------------------------------------------------------------------

def _append_log(record: dict[str, Any]) -> None:
    """将一条日志记录追加写入 api_logs.jsonl"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    统一 AI 调用入口。

    参数：
        agent_id    : 必填，标识调用来源的 Agent（如 "agent_search", "agent_summary"）
        messages    : 必填，OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]
        model       : 模型名称，支持 litellm 的所有模型别名（默认 gpt-4o-mini）
        temperature : 生成温度（默认 0.7）
        max_tokens  : 最大生成 token 数（可选）
        **kwargs    : 其他透传给 litellm.completion 的参数

    返回：
        模型生成的文本内容（str）

    示例：
        from ai_client import call_ai
        reply = call_ai(
            agent_id="my_agent",
            messages=[{"role": "user", "content": "你好"}],
            model="gpt-4o",
        )
    """
    call_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    try:
        # 调用 litellm（统一 OpenAI / Anthropic / 本地模型等）
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        latency = time.perf_counter() - start_time

        # 提取 usage 信息
        usage = response.usage or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        # 提取回复文本
        reply_text = response.choices[0].message.content if response.choices else ""

        # 构建日志记录
        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": model,
            "timestamp": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": round(latency, 4),
            "estimated_cost_usd": round(total_tokens / COST_PER_DOLLAR_TOKENS, 6),
            "status": "success",
            "error": None,
        }

        _append_log(log_record)

        # 控制台简要输出（可选）
        print(
            f"[AI] call_id={call_id} agent={agent_id} model={model} "
            f"tokens={total_tokens} latency={latency:.2f}s"
        )

        return reply_text

    except Exception as e:
        latency = time.perf_counter() - start_time

        # 记录失败日志
        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": model,
            "timestamp": time.time(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(latency, 4),
            "estimated_cost_usd": 0.0,
            "status": "error",
            "error": str(e),
        }

        _append_log(log_record)

        print(f"[AI] call_id={call_id} agent={agent_id} ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# 便捷：异步版本（可选）
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
    """
    异步版本的统一 AI 调用入口，参数与 call_ai 完全一致。
    """
    call_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    try:
        response = await litellm.acompletion(
            model=model,
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

        reply_text = response.choices[0].message.content if response.choices else ""

        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": model,
            "timestamp": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": round(latency, 4),
            "estimated_cost_usd": round(total_tokens / COST_PER_DOLLAR_TOKENS, 6),
            "status": "success",
            "error": None,
        }

        _append_log(log_record)

        print(
            f"[AI-async] call_id={call_id} agent={agent_id} model={model} "
            f"tokens={total_tokens} latency={latency:.2f}s"
        )

        return reply_text

    except Exception as e:
        latency = time.perf_counter() - start_time

        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": model,
            "timestamp": time.time(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(latency, 4),
            "estimated_cost_usd": 0.0,
            "status": "error",
            "error": str(e),
        }

        _append_log(log_record)

        print(f"[AI-async] call_id={call_id} agent={agent_id} ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# 模块级配置函数
# ---------------------------------------------------------------------------

def set_log_file(path: str | Path) -> None:
    """修改日志输出路径"""
    global LOG_FILE
    LOG_FILE = Path(path)


def set_cost_rate(tokens_per_dollar: float) -> None:
    """修改成本换算比率：$1 = tokens_per_dollar 个 tokens"""
    global COST_PER_DOLLAR_TOKENS
    COST_PER_DOLLAR_TOKENS = tokens_per_dollar
