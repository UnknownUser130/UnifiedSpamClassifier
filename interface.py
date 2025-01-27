import streamlit as st
import sqlite3
import hashlib
from datetime import datetime
import app_ver2

def adapt_datetime(ts):
    return ts.isoformat()

# Register the adapter
sqlite3.register_adapter(datetime, adapt_datetime)

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?',
              (username, hash_password(password)))
    data = c.fetchone()
    conn.close()
    return data is not None

def signup_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)',
                 (username, hash_password(password), datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
        
def logout():
    st.session_state.authenticated = False
    st.rerun()

def main():
    init_db()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Welcome to SMS Spam Detection System")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if login_user(login_username, login_password):
                    st.session_state.authenticated = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            signup_username = st.text_input("Choose Username", key="signup_username")
            signup_password = st.text_input("Choose Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Sign Up"):
                if signup_password != confirm_password:
                    st.error("Passwords don't match!")
                elif signup_user(signup_username, signup_password):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists!")
    else:
        col1, col2 = st.columns([0.9, 0.2])
        with col2:
            if st.button("Logout"):
                logout()

        app_ver2.run_app()

if __name__ == "__main__":
    main()