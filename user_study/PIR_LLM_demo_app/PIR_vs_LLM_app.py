import json
import os
import random
import re
from html import escape

import streamlit as st

from utils.model_client import ModelClient
from utils.load_dataset import load_dataset
from utils.user_management import (
    check_login_status,
    do_logout,
    get_user_dir,
    load_user_progress,
    get_user_output_path,
)
from utils.math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd

# ================= 配置与常量 =================
PIR_SYSTEM_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
)

TRADITIONAL_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

user_trigger_prompt = ''' Please reason step by step, and put your final answer within \\boxed{}.'''
input_file = "/data/home/chenxin/verl_interactive/datasets/mip/gsm8k.json"

pir_model_url = "http://localhost:1136"
pir_model_name = "Proactive-Interactive-R1-Math-7B-Max"

traditional_model_url = "http://localhost:1140"
traditional_model_name = "Qwen2.5-7B-Instruct"

MODEL_CONFIGS = {
    "pir": dict(
        model_path=pir_model_name,
        base_url=pir_model_url,
        stop_tokens=["</asking>", "<｜end▁of▁sentence｜>"],
        system_prompt=PIR_SYSTEM_PROMPT,
        reasoning_model=True,
    ),
    "traditional": dict(
        model_path=traditional_model_name,
        base_url=traditional_model_url,
        stop_tokens=["<｜end▁of▁sentence｜>"],
        system_prompt=TRADITIONAL_SYSTEM_PROMPT,
        reasoning_model=False,
    ),
}

MODEL_KEYS = ("a", "b")
MODEL_DEFAULT_FIELDS = [
    ("messages", []),
    ("client", None),
    ("step_state", "IDLE"),
    ("current_recommendations", []),
    ("completed", False),
    ("result", None),
    ("current_output", None),
    ("error", None),
    ("auto_completed", False),
]


def clone_default(value):
    return value.copy() if isinstance(value, list) else value


# ================= Model State 辅助函数 =================
def ms(key, field):
    return st.session_state.get(f"model_{key}_{field}")

def ms_set(key, field, value):
    st.session_state[f"model_{key}_{field}"] = value

def ms_has(key, field):
    return f"model_{key}_{field}" in st.session_state

def ms_del(key, field):
    st.session_state.pop(f"model_{key}_{field}", None)

def get_model_type(key):
    return st.session_state.model_order[0 if key == "a" else 1]

def get_user_simulator(key):
    return st.session_state[f"user_simulator_{key}"]

def create_model_client(model_type):
    cfg = MODEL_CONFIGS[model_type]
    return ModelClient(
        model_path=cfg["model_path"],
        base_url=cfg["base_url"],
        stop_tokens=cfg["stop_tokens"],
        reasoning_model=cfg.get("reasoning_model", False),
    )

def generate_recommendations(user_simulator, content, _current_item):
    response = user_simulator.chat(user_message=content)
    recs = [response.strip()]
    print("Generated recommendations:", recs)
    return recs


# ================= 页面设置 =================
st.set_page_config(page_title="Model Comparison Evaluation", layout="wide", initial_sidebar_state="collapsed")

# ================= 样式 =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; }
    .card { background: #fff; border: 1px solid #e8e8e8; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); transition: box-shadow 0.2s ease; }
    .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    .user-chat-container { display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 10px; }
    .user-chat-bubble { background: linear-gradient(135deg, #e0f2fe, #dbeafe); color: #1e293b; padding: 12px 18px; border-radius: 18px 4px 18px 18px; margin-right: 10px; max-width: 75%; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 0.95rem; line-height: 1.6; word-wrap: break-word; }
    .user-avatar { font-size: 28px; margin-top: -5px; }
    .model-header { text-align: center; padding: 12px 16px; border-radius: 12px; margin-bottom: 16px; font-weight: 600; font-size: 1.05rem; letter-spacing: 0.5px; transition: all 0.3s ease; }
    .model-a-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .model-b-header { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .header-active { box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { box-shadow: 0 0 15px rgba(102, 126, 234, 0.3); } 50% { box-shadow: 0 0 25px rgba(102, 126, 234, 0.6); } }
    .header-inactive { opacity: 0.45; filter: grayscale(30%); }
    .header-done { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important; box-shadow: 0 0 12px rgba(56, 239, 125, 0.3); }
    .header-error { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important; }
    .waiting-panel { background: linear-gradient(135deg, #f8f9fa, #e9ecef); border: 2px dashed #dee2e6; border-radius: 12px; padding: 40px 20px; text-align: center; color: #6c757d; min-height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .waiting-panel h3 { margin-bottom: 8px; color: #adb5bd; }
    .chat-container { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; min-height: 400px; max-height: 600px; overflow-y: auto; background-color: #fafafa; }
    .error-panel { background-color: #fff5f5; border: 1px solid #ffcccc; border-radius: 10px; padding: 20px; margin: 10px 0; }
    .auto-complete-panel { background: linear-gradient(135deg, #f0fff4, #c6f6d5); border: 1px solid #9ae6b4; border-radius: 12px; padding: 16px 20px; margin: 10px 0; text-align: center; }
    .auto-complete-panel h4 { margin: 0 0 4px 0; color: #22543d; }
    .auto-complete-panel p { margin: 0; color: #276749; font-size: 0.9rem; }
    .stButton > button { border-radius: 8px; font-weight: 500; transition: transform 0.15s ease, box-shadow 0.15s ease; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .stButton > button:active { transform: translateY(0); }
    .recommendation-box { background: #f0f7ff; border: 1px solid #bee3f8; border-radius: 10px; padding: 14px; margin: 12px 0; }
    .recommendation-box .rec-title { font-size: 0.8rem; color: #2b6cb0; font-weight: 600; margin-bottom: 8px; }
    .verification-card { border-radius: 10px; padding: 12px 16px; margin: 8px 0; }
    .verification-correct { background: #d4edda; border: 1px solid #c3e6cb; }
    .verification-wrong { background: #f8d7da; border: 1px solid #f5c6cb; }
    .verification-unknown { background: #fff3cd; border: 1px solid #ffeeba; }
    .verification-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge-correct { background: #a3d9a5; color: #155724; }
    .badge-wrong { background: #f1aeb5; color: #721c24; }
    .badge-unknown { background: #ffe69c; color: #856404; }
    .top-bar { display: flex; justify-content: space-between; align-items: center; padding: 8px 0 16px 0; border-bottom: 1px solid #eee; margin-bottom: 16px; }
    .top-bar-left { display: flex; align-items: center; gap: 12px; }
    .top-bar-title { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    .top-bar-subtitle { font-size: 0.8rem; color: #888; }
    .top-bar-user { background: #f0f0f0; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #555; }
    @keyframes thinkPulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.15); } }
    .thinking-box { display: flex; align-items: center; gap: 12px; padding: 16px 20px; background: linear-gradient(135deg, #fefce8, #fef9c3); border: 1px solid #fde68a; border-radius: 12px; margin: 10px 0; }
    .thinking-emoji { font-size: 1.5rem; animation: thinkPulse 1.5s ease-in-out infinite; display: inline-block; }
    .thinking-title { font-weight: 600; color: #92400e; font-size: 0.95rem; }
    .thinking-desc { color: #a16207; font-size: 0.8rem; }
    .empty-chat { text-align: center; padding: 40px 0; color: #adb5bd; }
    .empty-chat-icon { font-size: 2rem; margin-bottom: 8px; }
    .pref-card { text-align: center; border-radius: 12px; padding: 20px; background: #fff; transition: all 0.2s; }
    .pref-card-a { border: 2px solid #667eea; }
    .pref-card-b { border: 2px solid #f5576c; }
    .pref-card-icon { font-size: 2rem; margin-bottom: 8px; }
    .pref-card-label { font-size: 1rem; font-weight: 600; }
    .pref-card-result { margin: 12px 0; font-size: 1.5rem; }
    .pref-card-info { font-size: 0.85rem; color: #666; }
    .onboarding-box { background: linear-gradient(135deg, #e0f2fe, #dbeafe); border: 1px solid #93c5fd; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    .onboarding-box h4 { color: #1e40af; margin: 0 0 12px 0; }
    .onboarding-box p { color: #1e3a5f; margin: 6px 0; font-size: 0.95rem; line-height: 1.6; }
    .onboarding-step { display: flex; align-items: flex-start; gap: 10px; margin: 8px 0; }
    .onboarding-num { background: #3b82f6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0; margin-top: 2px; }
    .round-counter { font-size: 0.8rem; color: #888; text-align: right; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ================= 登录检查 =================
is_logged_in, username = check_login_status()
from utils.page import login_page

if not is_logged_in:
    login_page()
    st.stop()

# ================= Session State 初始化 =================
def init_session_state():
    defaults = {
        "current_index": 0, "dataset": [], "data_loaded": False,
        "load_message": None, "load_error": None, "dataset_seed": None,
        "current_item": None, "model_order": None,
        "user_simulator_a": None, "user_simulator_b": None,
        "overall_state": "IDLE", "onboarding_dismissed": False, "confirm_retry": False,
    }
    for k in MODEL_KEYS:
        for field, val in MODEL_DEFAULT_FIELDS:
            defaults[f"model_{k}_{field}"] = clone_default(val)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()
user_output_dir = get_user_dir(username)


# ================= UI 渲染辅助 =================

def render_top_bar(username, current_index, total):
    current_display = min(current_index + 1, total)
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-left">
            <span style="font-size:1.8rem;">🤖</span>
            <div>
                <div class="top-bar-title">Model Comparison Evaluation</div>
                <div class="top-bar-subtitle">Question {current_display} of {total}</div>
            </div>
        </div>
        <div><span class="top-bar-user">👤 {username}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_question_card(item, expanded=True):
    question = item.get('question', 'N/A').replace(user_trigger_prompt, '').strip()
    user_intent = item.get('user_intent', 'N/A')
    with st.expander("📋 Current Question", expanded=expanded):
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid #667eea;">
            <div style="font-size:0.75rem; text-transform:uppercase; color:#888; margin-bottom:8px; letter-spacing:1px;">Question</div>
            <div style="font-size:1rem; line-height:1.8; color:#2d3748;">{question}</div>
        </div>
        <div class="card" style="border-left: 4px solid #f5576c; background:#fff5f7;">
            <div style="font-size:0.75rem; text-transform:uppercase; color:#888; margin-bottom:8px; letter-spacing:1px;">🔒 User Intent (Hidden from Model)</div>
            <div style="font-size:0.95rem; color:#4a5568; font-style:italic;">{user_intent}</div>
        </div>
        """, unsafe_allow_html=True)


def render_model_header(key, state="active"):
    label, emoji = ("A", "🅰️") if key == "a" else ("B", "🅱️")
    color_class = "model-a-header" if key == "a" else "model-b-header"
    state_map = {
        "active": ("header-active", "ACTIVE"),
        "waiting": ("header-inactive", "WAITING"),
        "done": ("header-done", "COMPLETED ✅"),
        "error": ("header-error", "ERROR"),
    }
    state_class, status = state_map[state]
    st.markdown(f'<div class="model-header {color_class} {state_class}">{emoji} Model {label} — {status}</div>', unsafe_allow_html=True)


def render_round_counter(messages):
    n = len([m for m in messages if m['role'] == 'user'])
    st.markdown(f'<div class="round-counter">💬 {n} interaction round(s)</div>', unsafe_allow_html=True)


def render_chat_history(messages):
    if not messages:
        st.markdown('<div class="empty-chat"><div class="empty-chat-icon">💬</div><div>Conversation will appear here...</div></div>', unsafe_allow_html=True)
        return
    for msg in messages:
        if msg['role'] == 'user':
            c = escape(msg['content']).replace("\n", "<br>")
            st.markdown(f'<div class="user-chat-container"><div class="user-chat-bubble">{c}</div><div class="user-avatar">👤</div></div>', unsafe_allow_html=True)
        elif msg['role'] == 'assistant_reasoning':
            with st.chat_message("assistant", avatar="🧠"):
                with st.expander("💭 Reasoning Process", expanded=False):
                    t = msg['content'][:2000] + ("\n... (truncated)" if len(msg['content']) > 2000 else "")
                    st.warning(t)
        elif msg['role'] == 'assistant_response':
            with st.chat_message("assistant", avatar="💬"):
                st.info(f"**Model Response:** {msg['content']}")
        elif msg['role'] == 'assistant_output':
            with st.chat_message("assistant", avatar="✅"):
                st.success(msg['content'])


def render_thinking_indicator(label):
    st.markdown(f"""
    <div class="thinking-box">
        <div class="thinking-emoji">🧠</div>
        <div>
            <div class="thinking-title">{label} is thinking...</div>
            <div class="thinking-desc">The model is processing the problem. This may take a moment.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_box(recommendations, model_key, prefix_label="Responses"):
    if not recommendations:
        return None
    st.markdown(f'<div class="recommendation-box"><div class="rec-title">💡 AI-Suggested {prefix_label} — click to send directly, or write your own below</div></div>', unsafe_allow_html=True)
    for idx, rec in enumerate(recommendations):
        display = rec
        if st.button(f"📎 {display}", key=f"{model_key}_rec_btn_{idx}", use_container_width=True):
            return rec
    return None


def render_verification_status(verification):
    if verification is None:
        return
    reward, answer, gold = verification.get("reward"), verification.get("answer"), verification.get("gold")
    if reward is True:
        st.markdown(f'<div class="verification-card verification-correct"><span class="verification-badge badge-correct">✅ CORRECT</span><span style="margin-left:8px; color:#155724;">Model answered: <b>{answer}</b></span></div>', unsafe_allow_html=True)
    elif reward is False:
        st.markdown(f"""
        <div class="verification-card verification-wrong">
            <span class="verification-badge badge-wrong">❌ INCORRECT</span>
            <span style="margin-left:8px; color:#721c24;">Model: <b>{answer}</b> &nbsp;|&nbsp; Expected: <b>{gold}</b></span>
            <div style="margin-top:6px; font-size:0.85rem; color:#856404;">💡 You can provide feedback to help the model correct its answer, or click Finish to accept as-is.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verification-card verification-unknown"><span class="verification-badge badge-unknown">❓ UNVERIFIED</span><span style="margin-left:8px; color:#856404;">Could not automatically verify: <b>{answer}</b></span></div>', unsafe_allow_html=True)


def render_model_result(key):
    result = ms(key, "result")
    if not result:
        return
    if result.get("error"):
        st.warning(f"⚠️ Completed with error: {result['error']}")
    if result["reward"]:
        st.success(f"✅ Correct: {result['answer']}")
    elif result["reward"] is False:
        st.error(f"❌ Wrong: {result['answer']} (Expected: {result['gold']})")
    else:
        st.warning("❓ Could not verify")


def render_onboarding():
    st.markdown("""
    <div class="onboarding-box">
        <h4>👋 Welcome to Model Comparison Evaluation!</h4>
        <p>Here's how this works:</p>
        <div class="onboarding-step"><div class="onboarding-num">1</div><div>You'll see a math question and interact with two anonymous models (A then B). Their identities are hidden.</div></div>
        <div class="onboarding-step"><div class="onboarding-num">2</div><div>Models may respond and ask for your input — answer them using the suggested responses or type your own.</div></div>
        <div class="onboarding-step"><div class="onboarding-num">3</div><div>After both models finish, compare their performance and pick your preference.</div></div>
        <div class="onboarding-step"><div class="onboarding-num">4</div><div>Your interactions and preferences are saved automatically. You can resume anytime.</div></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✨ Got it! Let's start →", type="primary"):
        st.session_state.onboarding_dismissed = True
        st.rerun()


def render_preference_selection():
    st.markdown("## 🏆 Which model performed better?")
    st.caption("Consider: accuracy, reasoning quality, helpfulness of interactions, and how many rounds were needed.")

    col1, col2, col3 = st.columns([2, 1, 2])
    choices = {}
    for col, key, side in [(col1, "a", "👈 Prefer A"), (col3, "b", "👉 Prefer B")]:
        with col:
            result = ms(key, "result")
            reward = result["reward"] if result else None
            emoji = "✅" if reward else ("❌" if reward is False else "❓")
            rounds = len([m for m in ms(key, "messages") if m['role'] == 'user'])
            auto = " (auto)" if ms(key, "auto_completed") else ""
            label, icon, card_cls = ("A", "🅰️", "pref-card-a") if key == "a" else ("B", "🅱️", "pref-card-b")
            st.markdown(f'<div class="pref-card {card_cls}"><div class="pref-card-icon">{icon}</div><div class="pref-card-label">Model {label}</div><div class="pref-card-result">{emoji}</div><div class="pref-card-info">{rounds} round(s){auto}</div></div>', unsafe_allow_html=True)
            choices[key] = st.button(side, key=f"pref_{key}", use_container_width=True)

    with col2:
        st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)
        choices["tie"] = st.button("🤝 Tie", key="pref_tie", use_container_width=True)

    return choices["a"], choices["tie"], choices["b"]


def scroll_to_bottom():
    st.markdown("""<script>
        const m = window.parent.document.querySelector('section.main');
        if (m) m.scrollTo({top: m.scrollHeight, behavior: 'smooth'});
    </script>""", unsafe_allow_html=True)


# ================= 核心业务逻辑 =================

def _create_user_simulator(user_intent, user_question):
    from openai import OpenAI
    from utils.user_simulator_client import UserSimulatorClient
    client = OpenAI(
        api_key=os.getenv("USER_SIMULATOR_API_KEY", "YOUR_API_KEY_HERE"),
        base_url="https://api.ai-gaochao.cn/v1",
    )
    return UserSimulatorClient(
        client=client, model_name="gpt-4o-mini",
        task_name="question answering",
        user_intent=user_intent, user_question=user_question,
    )


def init_user_simulator(key, user_intent, user_question):
    st.session_state[f"user_simulator_{key}"] = _create_user_simulator(user_intent, user_question)


def verify_answer(output_content, current_item):
    try:
        print(f"current item: {current_item}")
        gold = current_item.get("answer") or extract_answer(current_item['solution'])
        answer = extract_answer(output_content)
        print(f"Extracted Answer: {answer}, Gold: {gold}")
        try:
            reward = grade_answer_mathd(answer, gold) or grade_answer_sympy(answer, gold)
        except Exception:
            reward = False
        return reward, answer, gold
    except Exception:
        return None, None, None


def _reset_model(key):
    for field, val in MODEL_DEFAULT_FIELDS:
        ms_set(key, field, clone_default(val))
    ms_del(key, "current_verification")


def _init_model(key):
    """初始化指定模型的 client、消息和用户模拟器，成功返回 True"""
    model_type = get_model_type(key)
    item = st.session_state.current_item
    init_user_simulator(key, user_intent=item['user_intent'], user_question=item['question'])
    try:
        ms_set(key, "client", create_model_client(model_type))
    except Exception as e:
        label = "A" if key == "a" else "B"
        ms_set(key, "error", f"Failed to initialize Model {label}: {e}")
        ms_set(key, "step_state", "ERROR")
        return False
    ms_set(key, "messages", [{"role": "user", "content": item['question']}])
    ms_set(key, "step_state", "REASONING")
    ms_set(key, "completed", False)
    ms_set(key, "current_recommendations", [])
    ms_set(key, "result", None)
    ms_set(key, "current_output", None)
    ms_set(key, "error", None)
    ms_set(key, "auto_completed", False)
    return True


def _extract_output(client, model_type=None):
    """从 client.completion 中提取最终输出。
    - PIR 模型：输出在 </think> 之后
    - Traditional 模型：整个 completion 就是输出（无 <think> 标签）
    """
    if not client or not client.completion:
        return None
    if model_type == "traditional":
        return client.completion.strip()
    if "</think>" in client.completion:
        return client.completion.split('</think>')[-1].strip()
    return client.completion.strip()


def finalize_output(key):
    client = ms(key, "client")
    messages = ms(key, "messages")
    model_type = get_model_type(key)
    output = _extract_output(client, model_type)
    if output is None:
        result = {"reward": None, "answer": "No output", "gold": None, "output": "Model failed to generate output"}
    else:
        messages.append({"role": "assistant_output", "content": output})
        reward, answer, gold = verify_answer(output, st.session_state.current_item)
        result = {"reward": reward, "answer": answer, "gold": gold, "output": output}
    ms_set(key, "result", result)
    ms_set(key, "completed", True)
    ms_set(key, "step_state", "DONE")


def finalize_with_error(key):
    client, messages, error_msg = ms(key, "client"), ms(key, "messages"), ms(key, "error")
    model_type = get_model_type(key)
    output = _extract_output(client, model_type) or "Error occurred - no valid output"
    if output and client and client.completion:
        messages.append({"role": "assistant_output", "content": output})
    reward, answer, gold = verify_answer(output, st.session_state.current_item) if output else (None, None, None)
    ms_set(key, "result", {"reward": reward, "answer": answer, "gold": gold, "output": output, "error": error_msg})
    ms_set(key, "completed", True)
    ms_set(key, "step_state", "DONE")
    ms_set(key, "error", None)


def process_pir_reasoning(key):
    client, messages = ms(key, "client"), ms(key, "messages")
    if not client:
        ms_set(key, "error", "Model client not initialized")
        ms_set(key, "step_state", "ERROR")
        return False
    try:
        input_msgs = [
            {"role": "system", "content": PIR_SYSTEM_PROMPT},
            {"role": "user", "content": st.session_state.current_item['question']},
            {"role": "assistant", "content": client.completion},
        ]
        model_response = client.chat(messages=input_msgs)

        if client.stop_reason == "</asking>":
            m = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL)
            if m:
                ask = m.group(1).strip()
                messages.append({"role": "assistant_response", "content": ask})
                print(f"Asking content extracted: {ask}")
                recs = generate_recommendations(get_user_simulator(key), ask, st.session_state.current_item)
                ms_set(key, "current_recommendations", recs)
                ms_set(key, "step_state", "WAITING_USER_INPUT")
                return True

        ms_set(key, "step_state", "ASSISTANT_OUTPUT")
        return True
    except Exception as e:
        ms_set(key, "error", f"Model call error: {e}")
        ms_set(key, "step_state", "ERROR")
        return False


def process_traditional_reasoning(key):
    """处理 Traditional LLM 的推理。
    与 LRM 的区别：
    - 不期望输出中有 <think>...</think> 标签
    - 直接把整个输出作为最终回答
    - 如果答案正确则自动完成，否则进入用户反馈环节
    """
    client, messages = ms(key, "client"), ms(key, "messages")
    if not client:
        ms_set(key, "error", "Model client not initialized")
        ms_set(key, "step_state", "ERROR")
        return False
    try:
        input_msgs = []
        for m in messages:
            if m['role'] == 'user':
                input_msgs.append({"role": "user", "content": m['content']})
            elif m['role'] in ('assistant_output', 'assistant_reasoning', 'assistant_response'):
                input_msgs.append({"role": "assistant", "content": m['content']})

        client.chat(messages=input_msgs, continue_mode=False)

        output = client.completion.strip()
        if not output:
            raise ValueError("Model returned empty output.")

        item = st.session_state.current_item
        reward, answer, gold = verify_answer(output, item)

        if reward is True:
            messages.append({"role": "assistant_output", "content": output})
            ms_set(key, "result", {
                "reward": reward, "answer": answer, "gold": gold,
                "output": output, "auto_completed": True
            })
            ms_set(key, "completed", True)
            ms_set(key, "auto_completed", True)
            ms_set(key, "step_state", "DONE")
            return True

        recs = generate_recommendations(get_user_simulator(key), output, item)
        ms_set(key, "current_output", output)
        ms_set(key, "current_recommendations", recs)
        ms_set(key, "current_verification", {"reward": reward, "answer": answer, "gold": gold})
        ms_set(key, "step_state", "WAITING_USER_INPUT")
        return True
    except Exception as e:
        ms_set(key, "error", f"Model call error: {e}")
        ms_set(key, "step_state", "ERROR")
        return False


def _send_pir_response(key, response):
    ms(key, "client").completion += f"\n<response>{response}</response>"
    ms(key, "messages").append({"role": "user", "content": response})
    ms_set(key, "current_recommendations", [])
    ms_set(key, "step_state", "REASONING")
    st.rerun()


def _send_traditional_feedback(key, feedback):
    ms(key, "messages").append({"role": "assistant_response", "content": ms(key, "current_output")})
    ms(key, "messages").append({"role": "user", "content": feedback})
    ms(key, "client").completion = ""
    ms_set(key, "current_recommendations", [])
    ms_set(key, "current_output", None)
    ms_set(key, "step_state", "REASONING")
    st.rerun()


# ================= 统一交互面板渲染 =================

def render_error_panel(key):
    label = "A" if key == "a" else "B"
    st.markdown(f'<div class="error-panel"><h4>⚠️ Error Occurred</h4><p>{ms(key, "error")}</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"🔄 Retry Model {label}", key=f"retry_{key}", use_container_width=True):
            try:
                ms_set(key, "client", create_model_client(get_model_type(key)))
                ms_set(key, "error", None)
                ms_set(key, "step_state", "REASONING")
                ms_set(key, "messages", [{"role": "user", "content": st.session_state.current_item['question']}])
                st.rerun()
            except Exception as e:
                st.error(f"Retry failed: {e}")
    with c2:
        if st.button("⏭️ Skip to Next Step", key=f"skip_{key}", use_container_width=True):
            finalize_with_error(key)
            st.rerun()


def render_active_model_interaction(key):
    """渲染活跃模型的交互区域（PIR / Traditional 通用）"""
    model_type = get_model_type(key)
    step = ms(key, "step_state")
    label = "A" if key == "a" else "B"
    is_last = (key == "b")

    if step == "ERROR":
        render_error_panel(key)
        return

    # === REASONING 状态 ===
    if step == "REASONING":
        render_thinking_indicator(f"Model {label}")
        with st.spinner("Processing..."):
            if model_type == "pir":
                process_pir_reasoning(key)
            else:
                process_traditional_reasoning(key)
            st.rerun()

    # === WAITING_USER_INPUT 状态 ===
    elif step == "WAITING_USER_INPUT":
        # 对于 Traditional LLM，显示当前输出和验证状态
        if model_type == "traditional" and ms(key, "current_output"):
            with st.chat_message("assistant", avatar="💬"):
                st.info(f"**Model Response:** {ms(key, 'current_output')}")
            if ms_has(key, "current_verification"):
                render_verification_status(ms(key, "current_verification"))

        st.markdown("---")
        st.markdown("##### 💬 Model is waiting for your input")

        sel = render_recommendation_box(ms(key, "current_recommendations"), f"model_{key}", "Responses")
        if sel:
            if model_type == "pir":
                _send_pir_response(key, sel)
            else:
                _send_traditional_feedback(key, sel)

        st.caption("Or type your own response:")
        user_input = st.text_input("💬 Your response:", key=f"model_{key}_user_input", placeholder="Type your response and press Enter...")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤 Send", key=f"model_{key}_send_btn", use_container_width=True):
                if user_input:
                    if model_type == "pir":
                        _send_pir_response(key, user_input)
                    else:
                        _send_traditional_feedback(key, user_input)
        with c2:
            # Traditional LLM 需要 Finish 按钮（PIR 会自动结束）
            if model_type == "traditional":
                if st.button("✅ Finish", key=f"model_{key}_finish_btn", type="primary", use_container_width=True):
                    finalize_output(key)
                    if is_last:
                        st.session_state.overall_state = "DONE"
                    st.rerun()

    # === ASSISTANT_OUTPUT 状态 - 仅 PIR 使用 ===
    elif step == "ASSISTANT_OUTPUT":
        finalize_output(key)
        if is_last:
            st.session_state.overall_state = "DONE"
        st.rerun()

    # === DONE 状态 ===
    elif step == "DONE":
        render_model_result(key)
        if is_last:
            st.session_state.overall_state = "DONE"
            st.rerun()
        else:
            st.markdown("---")
            st.success(f"✅ Model {label} completed!")
            if st.button("➡️ Proceed to Model B", type="primary", use_container_width=True):
                start_model_b()


# ================= 流程控制 =================

def start_next_question():
    idx = st.session_state.current_index
    if idx >= len(st.session_state.dataset):
        st.session_state.overall_state = "ALL_DONE"
        return
    item = st.session_state.dataset[idx]
    if user_trigger_prompt not in item['question']:
        item['question'] += user_trigger_prompt
    st.session_state.current_item = item

    order = ["pir", "traditional"]
    random.shuffle(order)
    st.session_state.model_order = order

    _reset_model("b")
    st.session_state.user_simulator_b = None
    _init_model("a")

    st.session_state.overall_state = "RUNNING_A"
    st.rerun()


def start_model_b():
    _init_model("b")
    st.session_state.overall_state = "RUNNING_B"
    st.rerun()


def reset_for_next_question():
    st.session_state.overall_state = "IDLE"
    st.session_state.model_order = None
    st.session_state.confirm_retry = False
    st.session_state.user_simulator_a = None
    st.session_state.user_simulator_b = None
    for key in ("a", "b"):
        _reset_model(key)


def load_data():
    try:
        user_progress = load_user_progress(username, input_file, pir_model_name)
        if user_progress:
            st.session_state.dataset = user_progress
            st.session_state.current_index = next(
                (i for i, item in enumerate(user_progress) if not item.get("pir_output") or not item.get("normal_output")),
                len(user_progress)
            )
            st.session_state.load_message = f"✅ Loaded progress: {len(user_progress)} items, continuing from question {st.session_state.current_index + 1}"
        else:
            ds = list({item['question']: item for item in load_dataset(input_file)}.values())
            if st.session_state.dataset_seed is None:
                st.session_state.dataset_seed = hash(username) % (2 ** 32)
            random.seed(st.session_state.dataset_seed)
            random.shuffle(ds)
            st.session_state.dataset = ds
            st.session_state.current_index = 0
            st.session_state.load_message = f"✅ Loaded {len(ds)} new items"
        st.session_state.data_loaded = True
        return True
    except Exception as e:
        import traceback
        st.session_state.load_error = f"Failed to load data: {e}\n\nDetails:\n{traceback.format_exc()}"
        st.session_state.data_loaded = True
        return False


def save_current_result():
    if not st.session_state.current_item:
        return False
    order = st.session_state.model_order
    if order:
        for key, idx in [("a", 0), ("b", 1)]:
            client = ms(key, "client")
            model_type = order[idx]

            if client:
                field = "pir_output" if model_type == "pir" else "normal_output"
                st.session_state.current_item[field] = client.completion

            messages = ms(key, "messages")
            if messages:
                history_field = "pir_messages" if model_type == "pir" else "normal_messages"
                st.session_state.current_item[history_field] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                ]

            result = ms(key, "result")
            if result:
                result_field = "pir_result" if model_type == "pir" else "normal_result"
                st.session_state.current_item[result_field] = result

    output_path = get_user_output_path(username, input_file, pir_model_name)
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(st.session_state.dataset, f, indent=4, ensure_ascii=False)
            st.toast("✅ Progress saved!", icon="💾")
            return True
        except Exception as e:
            st.toast(f"❌ Save failed: {e}", icon="⚠️")
    else:
        st.toast("❌ Save failed: Unable to get user directory", icon="⚠️")
    return False


# ================= 统一运行态页面渲染 =================

def render_running_state(active_key):
    item = st.session_state.current_item
    if not item:
        st.error("No current item found")
        st.session_state.overall_state = "IDLE"
        st.stop()

    render_question_card(item, expanded=(active_key == "a"))
    col_a, col_div, col_b = st.columns([5, 0.1, 5])

    with col_a:
        if active_key == "a":
            step = ms("a", "step_state")
            render_model_header("a", "error" if step == "ERROR" else "active")
            render_round_counter(ms("a", "messages"))
            with st.container():
                render_chat_history(ms("a", "messages"))
            render_active_model_interaction("a")
        else:
            render_model_header("a", "done")
            render_round_counter(ms("a", "messages"))
            with st.container():
                render_chat_history(ms("a", "messages"))
            render_model_result("a")

    with col_div:
        st.markdown('<div style="border-left: 2px solid #ddd; height: 600px; margin: 0 auto;"></div>', unsafe_allow_html=True)

    with col_b:
        if active_key == "a":
            render_model_header("b", "waiting")
            st.markdown('<div class="waiting-panel"><h3>⏳ Waiting...</h3><p>Please complete the interaction with Model A first.</p><p>Model B will start after Model A is done.</p></div>', unsafe_allow_html=True)
        else:
            step = ms("b", "step_state")
            render_model_header("b", "error" if step == "ERROR" else "active")
            render_round_counter(ms("b", "messages"))
            with st.container():
                render_chat_history(ms("b", "messages"))
            render_active_model_interaction("b")

    scroll_to_bottom()


# ================= 侧边栏 =================
with st.sidebar:
    st.markdown("### 📊 Session Stats")
    completed = st.session_state.current_index
    total_q = len(st.session_state.dataset) if st.session_state.dataset else 0
    if total_q > 0:
        st.metric("Completed", f"{completed}/{total_q}")
        if completed > 0:
            pref_counts = {"A": 0, "B": 0, "Tie": 0}
            for i in range(completed):
                p = st.session_state.dataset[i].get("user_preference", "")
                if p in pref_counts:
                    pref_counts[p] += 1
            c1, c2, c3 = st.columns(3)
            c1.metric("Prefer A", pref_counts["A"])
            c2.metric("Tie", pref_counts["Tie"])
            c3.metric("Prefer B", pref_counts["B"])
    st.markdown("---")
    if st.session_state.overall_state == "DONE" and st.session_state.model_order:
        st.markdown("### 🔍 Model Identity (This Round)")
        order = st.session_state.model_order
        st.markdown(f"**Model A** = `{order[0].upper()}`")
        st.markdown(f"**Model B** = `{order[1].upper()}`")
        st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        do_logout()
        st.rerun()


# ================= 数据加载 =================
if not st.session_state.data_loaded:
    with st.spinner("Loading data..."):
        if os.path.exists(input_file):
            load_data()
        else:
            st.session_state.load_error = f"⚠️ File not found: {input_file}"
            st.session_state.data_loaded = True

if st.session_state.load_message:
    st.toast(st.session_state.load_message, icon="✅")
    st.session_state.load_message = None
if st.session_state.load_error:
    st.error(st.session_state.load_error)
    st.session_state.load_error = None

if not st.session_state.dataset:
    st.warning("No data loaded. Please check the input file path.")
    st.info(f"Looking for: `{input_file}`")
    st.stop()

# ================= 主界面 =================
total = min(len(st.session_state.dataset), 100)
current = min(st.session_state.current_index + 1, total)
render_top_bar(username, st.session_state.current_index, total)
st.progress(current / total)

state = st.session_state.overall_state

# ===== IDLE =====
if state == "IDLE":
    if st.session_state.current_index == 0 and not st.session_state.onboarding_dismissed:
        render_onboarding()
        st.stop()
    if st.session_state.current_index >= len(st.session_state.dataset):
        st.session_state.overall_state = "ALL_DONE"
        st.rerun()
    else:
        st.markdown("---")
        if st.button("▶️ Start Next Question", type="primary", use_container_width=True):
            start_next_question()

# ===== RUNNING_A / RUNNING_B =====
elif state == "RUNNING_A":
    render_running_state("a")

elif state == "RUNNING_B":
    render_running_state("b")

# ===== DONE =====
elif state == "DONE":
    item = st.session_state.current_item
    with st.expander("📋 Question & Solution", expanded=False):
        st.markdown(f"**Question:** {item.get('question', 'N/A')}")
        st.markdown(f"**User Intent:** {item.get('user_intent', 'N/A')}")
        st.markdown(f"**Solution:** {item.get('solution', 'N/A')}")

    st.markdown("## 📊 Comparison Results")
    rc1, rc2 = st.columns(2)
    for col, key, title in [(rc1, "a", "🅰️ Model A"), (rc2, "b", "🅱️ Model B")]:
        with col:
            st.markdown(f"### {title}")
            render_model_result(key)
            with st.expander("💬 Chat History", expanded=False):
                for msg in ms(key, "messages"):
                    role_label = msg['role'].replace('_', ' ').title()
                    st.write(f"**{role_label}**: {msg['content']}")

    st.markdown("---")
    prefer_a, prefer_tie, prefer_b = render_preference_selection()

    if prefer_a or prefer_tie or prefer_b:
        item['user_preference'] = "A" if prefer_a else ("B" if prefer_b else "Tie")
        item['model_order'] = st.session_state.model_order
        item['model_a_auto_completed'] = ms("a", "auto_completed")
        item['model_b_auto_completed'] = ms("b", "auto_completed")
        save_current_result()
        st.session_state.current_index += 1
        reset_for_next_question()
        st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save Progress Only", use_container_width=True):
            save_current_result()
    with c2:
        if not st.session_state.confirm_retry:
            if st.button("🔄 Retry This Question", use_container_width=True):
                st.session_state.confirm_retry = True
                st.rerun()
        else:
            st.warning("⚠️ Are you sure? This will discard current progress on this question.")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes, retry", type="primary", use_container_width=True):
                    st.session_state.confirm_retry = False
                    reset_for_next_question()
                    st.rerun()
            with cc2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_retry = False
                    st.rerun()

# ===== ALL_DONE =====
elif state == "ALL_DONE":
    st.markdown("""
    <div style="text-align:center; padding:60px 0;">
        <div style="font-size:4rem; margin-bottom:16px;">🎉</div>
        <div style="font-size:1.5rem; font-weight:700; color:#2d3748; margin-bottom:8px;">Congratulations!</div>
        <div style="font-size:1rem; color:#718096; margin-bottom:24px;">You have completed all questions. Thank you for your participation!</div>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()
    st.info(f"📊 Total completed: {st.session_state.current_index} / {len(st.session_state.dataset)}")
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.current_index = 0
        st.session_state.data_loaded = False
        reset_for_next_question()
        st.rerun()
