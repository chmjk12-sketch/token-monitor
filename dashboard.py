"""
dashboard.py — AI Agent 调用监管数据看板
基于 Streamlit 构建，读取 api_logs.jsonl 展示消耗情况。
依赖：pip install streamlit pandas
运行：streamlit run dashboard.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Agent 调用监控",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 配置区域
# ---------------------------------------------------------------------------

LOG_FILE = st.sidebar.text_input("日志文件路径", value="api_logs.jsonl")
COST_PER_DOLLAR_TOKENS = st.sidebar.number_input(
    "成本换算：$1 = ? tokens",
    value=1_000_000,
    step=100_000,
    format="%d",
    help="用于估算成本的 token 单价",
)

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)  # 缓存 10 秒，自动刷新
def load_logs(filepath: str) -> pd.DataFrame:
    """从 JSONL 文件加载日志数据"""
    path = Path(filepath)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "call_id", "agent_id", "model", "timestamp",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "latency_seconds", "estimated_cost_usd", "status", "error",
            ]
        )

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 确保数值列正确
    for col in ["prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 重新计算成本（使用当前配置的换算率）
    df["estimated_cost_usd"] = df["total_tokens"] / COST_PER_DOLLAR_TOKENS

    # 时间转换
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        df["datetime"] = pd.NaT

    return df


df = load_logs(LOG_FILE)

# ---------------------------------------------------------------------------
# 标题
# ---------------------------------------------------------------------------

st.title("📊 AI Agent 调用监控面板")
st.caption(f"数据来源：`{LOG_FILE}`  |  共 {len(df)} 条记录")

if df.empty:
    st.warning("暂无日志数据。请先通过 `ai_client.py` 发起 AI 调用以生成日志。")
    st.stop()

# ---------------------------------------------------------------------------
# 第一部分：总览卡片
# ---------------------------------------------------------------------------

st.header("总览")

total_tokens = int(df["total_tokens"].sum())
total_prompt_tokens = int(df["prompt_tokens"].sum())
total_completion_tokens = int(df["completion_tokens"].sum())
total_cost_usd = df["estimated_cost_usd"].sum()
total_calls = len(df)
success_calls = len(df[df["status"] == "success"])
error_calls = len(df[df["status"] == "error"])
avg_latency = df["latency_seconds"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("总调用次数", f"{total_calls:,}", delta=f"成功 {success_calls:,} / 失败 {error_calls:,}")
col2.metric("总 Token 消耗", f"{total_tokens:,}", delta=f"输入 {total_prompt_tokens:,} + 输出 {total_completion_tokens:,}")
col3.metric("估算总成本", f"${total_cost_usd:.4f}", help=f"按 $1 = {COST_PER_DOLLAR_TOKENS:,.0f} tokens 计算")
col4.metric("平均延迟", f"{avg_latency:.2f}s")

# ---------------------------------------------------------------------------
# 第二部分：Top 5 Agent 排行
# ---------------------------------------------------------------------------

st.header("排行 — Token 消耗 Top 5 Agent")

agent_stats = (
    df.groupby("agent_id")
    .agg(
        total_tokens=("total_tokens", "sum"),
        total_cost=("estimated_cost_usd", "sum"),
        call_count=("call_id", "count"),
        avg_latency=("latency_seconds", "mean"),
    )
    .reset_index()
    .sort_values("total_tokens", ascending=False)
    .head(5)
)

agent_stats["total_tokens"] = agent_stats["total_tokens"].astype(int)
agent_stats["call_count"] = agent_stats["call_count"].astype(int)

st.dataframe(
    agent_stats.rename(
        columns={
            "agent_id": "Agent ID",
            "total_tokens": "总 Tokens",
            "total_cost": "估算成本 ($)",
            "call_count": "调用次数",
            "avg_latency": "平均延迟 (s)",
        }
    ).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# 第三部分：24 小时趋势折线图
# ---------------------------------------------------------------------------

st.header("趋势 — 过去 24 小时 Token 消耗")

now = datetime.now()
hours_24_ago = now - timedelta(hours=24)

# 过滤最近 24 小时的数据
df_recent = df[df["datetime"] >= hours_24_ago].copy()

if df_recent.empty:
    st.info("过去 24 小时内暂无调用记录。")
else:
    # 按小时分组
    df_recent["hour"] = df_recent["datetime"].dt.floor("h")

    hourly = (
        df_recent.groupby("hour")
        .agg(
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            call_count=("call_id", "count"),
        )
        .reset_index()
    )

    # 补全缺失的小时（确保 24 小时完整展示）
    all_hours = pd.date_range(start=hours_24_ago, end=now.floor("h"), freq="h")
    hourly_full = hourly.set_index("hour").reindex(all_hours, fill_value=0).reset_index()
    hourly_full.rename(columns={"index": "hour"}, inplace=True)

    # 绘制折线图
    st.line_chart(
        data=hourly_full,
        x="hour",
        y=["prompt_tokens", "completion_tokens", "total_tokens"],
        y_label="Token 数量",
        height=400,
    )

    # 小时明细表
    with st.expander("查看每小时明细数据"):
        hourly_display = hourly_full.copy()
        hourly_display["hour"] = hourly_display["hour"].dt.strftime("%Y-%m-%d %H:00")
        hourly_display.columns = ["时间", "输入 Tokens", "输出 Tokens", "总 Tokens", "调用次数"]
        st.dataframe(hourly_display.reset_index(drop=True), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 侧边栏：模型分布 & 最近调用记录
# ---------------------------------------------------------------------------

st.sidebar.header("模型分布")
if "model" in df.columns and not df.empty:
    model_counts = df["model"].value_counts().reset_index()
    model_counts.columns = ["模型", "调用次数"]
    st.sidebar.dataframe(model_counts, use_container_width=True, hide_index=True)

st.sidebar.header("最近 10 条调用")
recent_logs = df.nlargest(10, "timestamp") if "timestamp" in df.columns else df.tail(10)
if not recent_logs.empty:
    display_cols = ["call_id", "agent_id", "model", "total_tokens", "latency_seconds", "status"]
    available_cols = [c for c in display_cols if c in recent_logs.columns]
    st.sidebar.dataframe(
        recent_logs[available_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# 自动刷新
# ---------------------------------------------------------------------------

auto_refresh = st.sidebar.checkbox("自动刷新（每 30 秒）", value=False)
if auto_refresh:
    st_autorefresh = st.sidebar.number_input("刷新间隔（秒）", value=30, min_value=5, max_value=300)
    # Streamlit 原生不支持自动刷新，这里用 meta refresh 实现
    html(f"""<meta http-equiv="refresh" content="{st_autorefresh}">""", height=0)
    st.sidebar.caption(f"页面将每 {st_autorefresh} 秒自动刷新")
