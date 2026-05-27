from datetime import datetime, timedelta
import hashlib
import json
import os
import uuid
import streamlit as st

# 用户数据存储路径
USERS_DATA_DIR = "user_data"
USERS_DB_FILE = os.path.join(USERS_DATA_DIR, "users.json")
SESSIONS_DB_FILE = os.path.join(USERS_DATA_DIR, "sessions.json")

if not os.path.exists(USERS_DATA_DIR):
    os.makedirs(USERS_DATA_DIR)

# ================= 密码哈希 =================
def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

# ================= 用户数据库操作 =================
def load_users_db():
    """加载用户数据库"""
    if os.path.exists(USERS_DB_FILE):
        with open(USERS_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users_db(users_db):
    """保存用户数据库"""
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users_db, f, indent=4)

# ================= Session 管理 =================
def load_sessions_db():
    """加载会话数据库"""
    if os.path.exists(SESSIONS_DB_FILE):
        try:
            with open(SESSIONS_DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions_db(sessions_db):
    """保存会话数据库"""
    with open(SESSIONS_DB_FILE, 'w') as f:
        json.dump(sessions_db, f, indent=4)

def create_session(username):
    """创建新会话，返回 session_token"""
    sessions_db = load_sessions_db()
    
    # 生成唯一 token
    session_token = str(uuid.uuid4())
    
    # 设置过期时间（7天）
    expire_time = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    
    sessions_db[session_token] = {
        "username": username,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expires_at": expire_time
    }
    
    save_sessions_db(sessions_db)
    return session_token

def validate_session(session_token):
    """验证会话是否有效，返回 (is_valid, username)"""
    if not session_token:
        return False, None
    
    sessions_db = load_sessions_db()
    
    if session_token not in sessions_db:
        return False, None
    
    session = sessions_db[session_token]
    expire_time = datetime.strptime(session["expires_at"], "%Y-%m-%d %H:%M:%S")
    
    if datetime.now() > expire_time:
        # 会话过期，删除
        del sessions_db[session_token]
        save_sessions_db(sessions_db)
        return False, None
    
    return True, session["username"]

def delete_session(session_token):
    """删除会话（登出）"""
    if not session_token:
        return
    
    sessions_db = load_sessions_db()
    if session_token in sessions_db:
        del sessions_db[session_token]
        save_sessions_db(sessions_db)

def cleanup_expired_sessions():
    """清理过期会话"""
    sessions_db = load_sessions_db()
    now = datetime.now()
    
    expired_tokens = []
    for token, session in sessions_db.items():
        expire_time = datetime.strptime(session["expires_at"], "%Y-%m-%d %H:%M:%S")
        if now > expire_time:
            expired_tokens.append(token)
    
    for token in expired_tokens:
        del sessions_db[token]
    
    if expired_tokens:
        save_sessions_db(sessions_db)

# ================= 用户管理函数 =================
def register_user(username, password):
    """注册新用户"""
    users_db = load_users_db()
    if username in users_db:
        return False, "Username already exists"
    
    users_db[username] = {
        "password": hash_password(password),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_dir": os.path.join(USERS_DATA_DIR, username)
    }
    
    # 创建用户专属目录
    user_dir = users_db[username]["user_dir"]
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    save_users_db(users_db)
    return True, "Registration successful"

def login_user(username, password):
    """用户登录，返回 (success, message, session_token)"""
    users_db = load_users_db()
    if username not in users_db:
        return False, "Username does not exist", None
    
    if users_db[username]["password"] != hash_password(password):
        return False, "Incorrect password", None
    
    # 创建会话
    session_token = create_session(username)
    return True, "Login successful", session_token

def logout_user(session_token):
    """用户登出"""
    delete_session(session_token)

def get_user_dir(username):
    """获取用户数据目录"""
    users_db = load_users_db()
    if username in users_db:
        return users_db[username]["user_dir"]
    return None

def get_user_output_path(username, input_file, model_name):
    """获取用户特定的输出文件路径"""
    user_dir = get_user_dir(username)
    if not user_dir:
        return None
    
    file_name = input_file.split("/")[-1].split(".json")[0].replace(".parquet", "")
    output_filename = f"{model_name}_human_interactive_{file_name}_generation_result.json"
    return os.path.join(user_dir, output_filename)

def load_user_progress(username, input_file, model_name):
    """加载用户的进度"""
    output_path = get_user_output_path(username, input_file, model_name)
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return json.load(f)[0:50]
        except:
            return None
    return None

# ================= Streamlit 集成函数 =================
def init_auth_state():
    """初始化认证状态"""
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

def check_login_status():
    """
    检查登录状态
    返回 (is_logged_in, username)
    """
    init_auth_state()
    
    # 清理过期会话
    cleanup_expired_sessions()
    
    # 如果有 session_token，验证它
    if st.session_state.session_token:
        is_valid, username = validate_session(st.session_state.session_token)
        if is_valid:
            st.session_state.logged_in = True
            st.session_state.username = username
            return True, username
        else:
            # Token 无效或过期
            st.session_state.session_token = None
            st.session_state.logged_in = False
            st.session_state.username = None
            return False, None
    
    return False, None

def do_login(username, password):
    """执行登录"""
    success, message, session_token = login_user(username, password)
    if success:
        st.session_state.session_token = session_token
        st.session_state.logged_in = True
        st.session_state.username = username
    return success, message

def do_logout():
    """执行登出"""
    logout_user(st.session_state.session_token)
    st.session_state.session_token = None
    st.session_state.logged_in = False
    st.session_state.username = None
    # 清除其他状态
    keys_to_keep = ["session_token", "logged_in", "username"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]