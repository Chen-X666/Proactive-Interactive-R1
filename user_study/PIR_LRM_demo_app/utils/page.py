from utils.user_management import (
    do_login,
    do_logout,
    check_login_status,
    register_user,
)

def login_page():
    import streamlit as st
    st.title("Login")
        
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("🔐 Login to your account")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn", type="primary"):
            if login_username and login_password:
                success, message = do_login(login_username, login_password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter both username and password")

    with tab2:
        st.subheader("Create a new account")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
        
        if st.button("Register", key="register_btn", type="primary"):
            if reg_username and reg_password and reg_password_confirm:
                if reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    success, message = register_user(reg_username, reg_password)
                    if success:
                        st.success(message + ", please login")
                    else:
                        st.error(message)
            else:
                st.warning("Please fill in all fields")

    st.stop()