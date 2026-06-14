"""
litellm_proxy.py — AI API 透明代理（监控所有调用）
支持 OpenAI 兼容格式，应用只需修改 API base_url

部署：python3 litellm_proxy.py
监听：http://0.0.0.0:4000
日志：api_logs.jsonl（与 dashboard 同目录）

应用改：base_url = "http://127.0.0.1:4000" 即可
"""

import json
import time
import uuid
import os
import asyncio
from datetime import datetime
from typing import Optional

import litellm
from litellm import completion as litellm_completion
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
LOG_FILE = os.environ.get("LOG_FILE", "/usr/share/nginx/html/api_logs.jsonl")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "4000"))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 定价（真实成本计算）
MODEL_PRICING = {
    "deepseek/deepseek-chat": {
        "input_cache_hit": 0.5, "input_cache_miss": 2.0, "output": 8.0, "currency": "CNY",
    },
    "deepseek/deepseek-reasoner": {
        "input_cache_hit": 1.0, "input_cache_miss": 4.0, "output": 16.0, "currency": "CNY",
    },
    "openai/gpt-4o": {"input": 2.5, "output": 10.0, "currency": "USD"},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6, "currency": "USD"},
    "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0, "currency": "USD"},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25, "currency": "USD"},
}

USD_TO_CNY = 7.2

app = FastAPI(title="AI Proxy Monitor")

# ---------------------------------------------------------------------------
# Agent 识别：从 API Key 或 Header 提取
# ---------------------------------------------------------------------------
AGENT_MAP = {}  # 动态注册：api_key -> agent_id


def detect_agent(request: Request) -> str:
    """从请求中识别 agent_id"""
    # 优先从 header 取
    agent = request.headers.get("X-Agent-Id", "")
    if agent:
        return agent
    
    # 从 API Key 映射
    auth = request.headers.get("Authorization", "")
    api_key = auth.replace("Bearer ", "").strip()
    if api_key:
        mapped = AGENT_MAP.get(api_key, "")
        if mapped:
            return mapped
    
    return "unknown"


def resolve_model(model: str) -> str:
    """解析模型名称为完整格式"""
    aliases = {
        "deepseek-chat": "deepseek/deepseek-chat",
        "deepseek-reasoner": "deepseek/deepseek-reasoner",
        "deepseek-v4-flash": "deepseek/deepseek-chat",
        "deepseek-v4-pro": "deepseek/deepseek-reasoner",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "claude-3-haiku": "anthropic/claude-3-haiku",
    }
    if model in aliases:
        return aliases[model]
    if not model.startswith(("deepseek/", "openai/", "anthropic/")):
        return f"deepseek/{model}" if "deepseek" in model.lower() else model
    return model


def calc_cost(model, prompt_tokens, completion_tokens, cache_hit_tokens=0, cache_miss_tokens=0):
    """计算真实成本"""
    resolved = resolve_model(model)
    pricing = MODEL_PRICING.get(resolved, MODEL_PRICING.get("deepseek/deepseek-chat"))
    currency = pricing.get("currency", "CNY")
    
    if "input_cache_hit" in pricing:
        hit = cache_hit_tokens or 0
        miss = cache_miss_tokens or max(0, prompt_tokens - hit)
        input_cost = (hit * pricing["input_cache_hit"] + miss * pricing["input_cache_miss"]) / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        total = input_cost + output_cost
        if currency == "CNY":
            return round(total, 6), round(total / USD_TO_CNY, 6)
        else:
            return round(total * USD_TO_CNY, 6), round(total, 6)
    else:
        input_cost = prompt_tokens * pricing["input"] / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        total = input_cost + output_cost
        if currency == "USD":
            return round(total * USD_TO_CNY, 6), round(total, 6)
        else:
            return round(total, 6), round(total / USD_TO_CNY, 6)


def append_log(record):
    """追加一行日志"""
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model"},
            {"id": "deepseek-reasoner", "object": "model"},
            {"id": "deepseek-v4-flash", "object": "model"},
            {"id": "deepseek-v4-pro", "object": "model"},
        ]
    }


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """透明代理：接收 OpenAI 格式请求，转发 DeepSeek，记录日志"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    model = body.get("model", "deepseek-chat")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    
    # 提取认证信息和 agent
    auth = request.headers.get("Authorization", "")
    api_key = auth.replace("Bearer ", "").strip() or DEEPSEEK_API_KEY
    agent_id = detect_agent(request)
    call_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    
    resolved_model = resolve_model(model)
    
    try:
        # 转发请求
        kwargs = {
            "model": f"deepseek/{model}" if not model.startswith(("deepseek/", "openai/", "anthropic/")) else model,  # litellm 要求格式
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if api_key:
            kwargs["api_key"] = api_key
        
        response = litellm_completion(**kwargs)
        
        if stream:
            # 流式处理：收完整响应后记录日志
            return await handle_stream(response, call_id, agent_id, resolved_model, messages, api_key, start_time)
        
        latency = time.perf_counter() - start_time
        
        # 提取 usage
        usage = response.usage or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
        
        cost_cny, cost_usd = calc_cost(resolved_model, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens)
        
        # 记录日志
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
            "cost_cny": cost_cny,
            "cost_usd": cost_usd,
            "cost_currency": "CNY",
            "status": "success",
            "error": None,
        }
        append_log(log_record)
        
        print(f"[PROXY] agent={agent_id} model={resolved_model} tokens={total_tokens} cost=¥{cost_cny:.4f} latency={latency:.2f}s")
        
        return JSONResponse(content=response.model_dump() if hasattr(response, 'model_dump') else response.json())
    
    except Exception as e:
        latency = time.perf_counter() - start_time
        error_msg = str(e)
        
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
            "error": error_msg,
        }
        append_log(log_record)
        
        print(f"[PROXY] ERROR agent={agent_id} model={resolved_model} err={error_msg[:100]}")
        raise HTTPException(status_code=502, detail=f"Proxy error: {error_msg}")


async def handle_stream(response, call_id, agent_id, resolved_model, messages, api_key, start_time):
    """处理流式响应，收齐后记录日志"""
    collected_chunks = []
    
    async def generate():
        total_prompt = 0
        total_completion = 0
        total_cache_hit = 0
        total_cache_miss = 0
        
        async for chunk in response:
            collected_chunks.append(chunk)
            
            # 尝试提取 usage（最后一个 chunk 通常带）
            usage = getattr(chunk, "usage", None) if hasattr(chunk, "usage") else None
            if usage:
                total_prompt = getattr(usage, "prompt_tokens", 0) or 0
                total_completion = getattr(usage, "completion_tokens", 0) or 0
                total_cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
                total_cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
            
            # 转回原始格式
            if hasattr(chunk, "model_dump"):
                yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        yield "data: [DONE]\n\n"
        
        # 流结束后记录日志
        latency = time.perf_counter() - start_time
        cost_cny, cost_usd = calc_cost(resolved_model, total_prompt, total_completion, total_cache_hit, total_cache_miss)
        
        log_record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "model": resolved_model,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "cache_hit_tokens": total_cache_hit,
            "cache_miss_tokens": total_cache_miss,
            "latency_seconds": round(latency, 4),
            "cost_cny": cost_cny,
            "cost_usd": cost_usd,
            "cost_currency": "CNY",
            "status": "success",
            "error": None,
        }
        append_log(log_record)
        
        print(f"[PROXY][stream] agent={agent_id} model={resolved_model} tokens={total_prompt + total_completion} cost=¥{cost_cny:.4f} latency={latency:.2f}s")
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Agent 注册 API
# ---------------------------------------------------------------------------

@app.post("/register_agent")
async def register_agent(data: dict):
    """将 API Key 映射到 Agent 名称
    用法：curl -X POST http://127.0.0.1:4000/register_agent \
              -H "Content-Type: application/json" \
              -d '{"api_key":"sk-xxx","agent_id":"workbuddy"}'
    """
    api_key = data.get("api_key", "")
    agent_id = data.get("agent_id", "")
    if not api_key or not agent_id:
        raise HTTPException(status_code=400, detail="api_key and agent_id required")
    AGENT_MAP[api_key] = agent_id
    return {"status": "ok", "agent_id": agent_id}


@app.post("/unregister_agent")
async def unregister_agent(data: dict):
    api_key = data.get("api_key", "")
    if api_key in AGENT_MAP:
        del AGENT_MAP[api_key]
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"🚀 AI Proxy 启动于 :{PROXY_PORT}")
    print(f"📝 日志: {LOG_FILE}")
    print(f"🔑 DeepSeek API Key: {'已配置' if DEEPSEEK_API_KEY else '未配置'}")
    print(f"")
    print(f"应用修改 base_url 为: http://127.0.0.1:{PROXY_PORT}")
    print(f"例如: python openai.OpenAI(api_key='...', base_url='http://127.0.0.1:{PROXY_PORT}')")
    print(f"")
    print(f"Agent 注册示例:")
    print(f"  curl -X POST http://127.0.0.1:{PROXY_PORT}/register_agent \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"api_key\":\"sk-your-key\",\"agent_id\":\"workbuddy\"}}'")
    print(f"")
    print(f"或者设置 HTTP Header: X-Agent-Id: workbuddy")
    
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
