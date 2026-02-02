import streamlit as st
import os
import json
import numpy as np
import time
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pandas as pd
import db_manager as db
import random # Needed for robust retry logic

# ==============================================================
# CONFIGURATION
# ==============================================================
load_dotenv()
# Set Streamlit's default theme to dark for best results with custom CSS
st.set_page_config(page_title="⚖️ LegalBot", layout="wide")

# Model selection - using flash-lite for resilience
MODEL_NAME = 'gemini-2.5-flash-lite' 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_FILE = "legal_vector_db.json"
NUM_RAG_RESULTS = 3

# Initialize Gemini client
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    client = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    client = None
    

# ==============================================================
# PAGE STYLE (UPDATED & OPTIMIZED FOR DARK THEME)
# ==============================================================
st.markdown("""
<style>
/* Main Background and Containers */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: linear-gradient(180deg, #0B1A3C 0%, #1E3A8A 100%) !important;
    color: #E2E8F0 !important;
}
[data-testid="stHeader"] { box-shadow: none !important; height: 0 !important; }

/* Sidebar - Professional & Dark */
[data-testid="stSidebar"] {
    background: #0D1B38 !important;
    color: #F8FAFC !important;
    border-right: 1px solid rgba(255,255,255,0.1);
    padding-top: 2rem;
}
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }

/* Headers and Primary Text */
h1,h2,h3,h4 { color: #F8FAFC !important; font-weight: 600; }

/* Button Styles */
.stButton>button {
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.2);
    background-color: rgba(255,255,255,0.05);
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: rgba(255,255,255,0.15);
}
.stButton.primary>button { /* Targeted primary button for the sidebar (New Query) */
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
    color: white !important;
}
.stButton.primary>button:hover {
    background-color: #1D4ED8 !important;
}


/* Chat Styling */
.chat-container { max-width:95%; padding:1rem 2rem; margin:auto; }
.user-msg, .bot-msg {
  border-radius:14px; padding:1rem 1.2rem; margin:1rem 0; font-size:15px;
  line-height:1.6; width:fit-content; display:inline-block;
}
.user-msg {
  background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
  color:#F8FAFC; margin-left:auto; float:right; clear:both;
}
.bot-msg {
  background-color: rgba(255,255,255,0.08); color:#F1F5F9;
  width:90%; float:left; clear:both; border-left:3px solid #60A5FA; padding-left:1.5rem;
  margin-bottom: 0.25rem !important;
}

/* Chat Input Styling */
.stChatInput textarea {
  background: rgba(255,255,255,0.12) !important;
  border:1px solid rgba(255,255,255,0.3) !important;
  border-radius:14px !important; color:#F8FAFC !important; padding:0.8rem 1rem !important;
}

/* Quick Action Buttons Styling */
.quick-actions-container {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    float: left;
    clear: both;
    width: 90%;
    margin-left: 1.5rem;
    display: flex; /* Use flexbox for button layout */
    gap: 10px; /* Spacing between buttons */
}
.quick-action-button button {
    background-color: #1A56CC !important;
    color: #F8FAFC !important;
    border: none;
    padding: 0.4rem 0.8rem !important;
    font-size: 13px;
    height: auto !important; /* Fix for button height */
}
.quick-action-button button:hover {
    background-color: #3B82F6 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# DATABASE + EMBEDDINGS UTILITIES
# ==============================================================
@st.cache_resource
def load_vector_db():
    if not os.path.exists(VECTOR_DB_FILE):
        return []
    try:
        with open(VECTOR_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error reading {VECTOR_DB_FILE}. Please run create_embeddings.py again.")
        return []

@st.cache_resource
def load_local_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_embedding_local(text: str):
    return load_local_embedding_model().encode(text)

def search_relevant_sections_semantic(user_query: str, db_data: List[Dict]):
    if not db_data:
        return []
    query_embedding = get_embedding_local(user_query)
    sims = cosine_similarity([query_embedding],
                             [np.array(item["embedding"]) for item in db_data])[0]
    top_k = np.argsort(sims)[::-1][:NUM_RAG_RESULTS]
    return [(db_data[i].get("section_number", f"N/A-{i}"), db_data[i].get("text","")) for i in top_k]

# ==============================================================
# LLM RESPONSE WITH EXPONENTIAL BACKOFF + FALLBACK
# ==============================================================
def generate_llm_response(prompt: str, results: List[Tuple[str, str]]) -> str:
    """
    Try to generate using Gemini with robust Exponential Backoff.
    Saves references to session state for quick action buttons.
    """
    if client is None:
        st.session_state.last_references = []
        return "⚠️ Gemini client not initialized. Showing retrieved sections below.\n\n" + \
               "\n\n".join([f"**Section {s}:** {t[:400]}..." for s,t in results])

    context = "\n\n".join([f"Section {s}: {t}" for s,t in results])
    instruction = (
        "You are LegalBot, an expert in Indian law (Bharatiya Nyaya Sanhita, 2023). "
        "Provide a clear, user-friendly explanation (3–5 short paragraphs). "
        "Mention referenced sections naturally."
    )

    base_delay = 3
    max_attempts = 5
    last_error_message = ""
    st.session_state.last_references = [] # Reset references

    for attempt in range(max_attempts):
        try:
            response = client.generate_content(
                f"{instruction}\n\nContext:\n{context}\n\nQuestion:\n{prompt}"
            )
            reply = response.text.strip()
            if results:
                # Save references to session state for quick action buttons
                st.session_state.last_references = [s for s,_ in results]
                reply += "\n\n**📚 References:** " + ", ".join(f"Section {s}" for s,_ in results)
            return reply
            
        except Exception as e:
            emsg = str(e)
            last_error_message = emsg
            
            if "503" in emsg or "UNAVAILABLE" in emsg or "429" in emsg:
                if attempt < max_attempts - 1:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 2)
                    st.warning(f"Server overloaded (503/429). Retrying in {wait_time:.1f}s... (Attempt {attempt + 2}/{max_attempts})")
                    time.sleep(wait_time)
                    continue
                else:
                    break
            
            if "NOT_FOUND" in emsg or "404" in emsg:
                return "⚠️ The selected Gemini model is not available for your account. Showing retrieved sections below.\n\n" + \
                       "\n\n".join([f"**Section {s}:** {t[:400]}..." for s,t in results])
            
            break

    fallback = "⚠️ Gemini servers are temporarily unavailable or highly overloaded. Showing retrieved legal references:\n\n"
    fallback += "\n\n".join([f"**Section {s}:** {t[:400]}..." for s,t in results])
    return fallback


# ==============================================================
# CHAT UTILITIES (QUICK ACTIONS & FEEDBACK)
# ==============================================================

def handle_quick_action(action_type: str):
    """Handles the logic for quick action buttons."""
    if action_type == "view_refs":
        refs = st.session_state.get('last_references', [])
        ref_list = ", ".join(f"Section {r}" for r in refs) if refs else "None."
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**📚 Quick Action:** You requested the references cited. The sections are: {ref_list}. \n\n(In a future version, this button will directly link to the full text.)"
        })
    elif action_type == "follow_up":
        st.session_state.messages.append({
            "role": "assistant",
            "content": "What follow-up question do you have regarding the last response? Feel free to ask about definitions, implications, or related laws."
        })
    elif action_type == "new_query":
        # Reset the chat history and references
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👩‍⚖️ New conversation started. Ask me any legal question."
        }]
    
    st.session_state.last_references = [] # Clear refs after action
    st.session_state.view_mode = "chat"
    st.rerun()


def add_quick_action_buttons(key_prefix: str):
    """Adds Quick Action buttons below the bot message."""
    
    # Quick Action Buttons (Uses custom CSS container)
    st.markdown('<div class="quick-actions-container">', unsafe_allow_html=True)
    
    # Removed View Cited Sections and Ask a Follow-up buttons per user request
    
    st.markdown('</div>', unsafe_allow_html=True)


def add_feedback_buttons(username: str, user_q: str, bot_a: str, results: List, key_prefix: str):
    """Adds the 👍/👎 feedback buttons."""
    
    st.markdown(f'<div style="width: 90%; float: left; clear: both; margin-left: 1.5rem; padding-bottom: 1rem;">', unsafe_allow_html=True)
    fcols = st.columns([0.08, 0.08, 0.84]) # Small columns for thumbs

    # Feedback Logic
    with fcols[0]:
        if st.button("👍", key=f"{key_prefix}_like", help="Response was helpful and accurate."):
            db.log_interaction(username, user_q, bot_a, results, rating=1)
            st.toast("Feedback logged! Thank you.", icon="✅")
    with fcols[1]:
        if st.button("👎", key=f"{key_prefix}_dislike", help="Response was unhelpful or inaccurate."):
            db.log_interaction(username, user_q, bot_a, results, rating=0)
            st.toast("Feedback logged! We'll review this response.", icon="❌")
            
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================
# AUTH PAGE (centered)
# ==============================================================
def show_auth_page():
    st.markdown("<h2 style='text-align:center;'>⚖️ LegalBot Login / Register</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Initialize session state for mode if not exists
        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "Login"
        
        # Mode selection with session state
        mode_choice = st.radio("Select Mode:", ["🔐Login", "👤Register", "🔵Google"], horizontal=True, key="auth_mode_radio")
        st.session_state.auth_mode = mode_choice
        
        if st.session_state.auth_mode == "🔵Google":
            st.markdown("### Sign in with Google")
            st.info("📧 To sign in with Google, enter your Google email address below. An account will be automatically created if it doesn't exist.")
            
            google_email = st.text_input("Google Email", placeholder="your.email@gmail.com", key="google_email")
            google_name = st.text_input("Full Name (optional)", placeholder="Your Name", key="google_name")
            
            col_google1, col_google2 = st.columns(2)
            with col_google1:
                if st.button("🔵 Continue with Google", use_container_width=True, key="google_signin_btn"):
                    if not google_email:
                        st.warning("⚠️ Please enter your email address.")
                    elif "@" not in google_email:
                        st.warning("⚠️ Please enter a valid email address.")
                    else:
                        # Create or get user
                        name = google_name if google_name else google_email.split("@")[0]
                        if db.add_or_update_google_user(google_email, name, google_email):
                            st.session_state.authenticated = True
                            st.session_state.username = google_email
                            st.session_state.name = name
                            st.success("✅ Google Sign-In successful!")
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please try again.")
            
            with col_google2:
                if st.button("← Back", use_container_width=True, key="back_from_google"):
                    st.session_state.auth_mode = "🔐Login"
                    st.rerun()
        
        else:
            username = st.text_input("Username (Email)", key="auth_username")
            password = st.text_input("Password", type="password", key="auth_password")

            if st.session_state.auth_mode == "👤Register":
                name = st.text_input("Full Name", key="auth_name")
                col_reg1, col_reg2 = st.columns(2)
                with col_reg1:
                    if st.button("📝 Register", use_container_width=True, key="register_btn"):
                        if not username or not password or not name:
                            st.warning("⚠️ Please fill all fields.")
                        elif db.get_user(username):
                            st.error("❌ User already exists. Try logging in.")
                        else:
                            if db.add_user(username, name, password):
                                st.success("✅ Registered successfully! Please login now.")
                                st.session_state.auth_mode = "🔐Login"
                                st.rerun()
                            else:
                                st.error("❌ Registration failed. Please try again.")
                with col_reg2:
                    if st.button("← Back to Login", use_container_width=True, key="back_to_login"):
                        st.session_state.auth_mode = "🔐Login"
                        st.rerun()
            else:
                col_login1, col_login2 = st.columns(2)
                with col_login1:
                    if st.button("🔐 Login", use_container_width=True, key="login_btn"):
                        if not username or not password:
                            st.warning("⚠️ Please fill all fields.")
                        elif db.verify_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            user_data = db.get_user(username)
                            st.session_state.name = user_data["name"] if user_data else username
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password.")
                with col_login2:
                    if st.button("→ Go to Register", use_container_width=True, key="go_to_register"):
                        st.session_state.auth_mode = "👤Register"
                        st.rerun()

# ==============================================================
# MAIN APP INTERFACE (UPDATED SIDEBAR & LOGS VIEW)
# ==============================================================
def main_app_interface():
    username = st.session_state.username
    name = st.session_state.name

    # (File upload feature removed; no session-level uploaded docs necessary)

    # --- SIDEBAR (Updated UI/UX) ---
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {username}")
        st.markdown("---") 

        st.markdown("#### 💬 AI Legal Workflow")
        
        # New Query Button (Primary Call to Action)
        if st.button("➕ New Legal Query", key="new_query_sidebar", type="primary", use_container_width=True):
            handle_quick_action("new_query")
        
        # Navigation Buttons (using session state to manage view)
        if st.button("⏱️ Review History", key="review_history_sidebar", use_container_width=True):
            st.session_state.view_mode = "logs"
        if st.button("⚖️ Current Chat", key="current_chat_sidebar", use_container_width=True):
            st.session_state.view_mode = "chat"
        
        st.markdown("---")

        # Document Vault removed per request (file upload moved into chat area)
        
        st.markdown("---")
        
        # Logout moved to the bottom
        if st.button("🚪 Logout", key="logout_sidebar", use_container_width=True, help="Log out of the LegalBot application."):
            st.session_state.authenticated = False
            st.session_state.clear()
            st.rerun()
            
    # --- MAIN CONTENT ---
    db_data = load_vector_db()

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "chat"
    
    if "last_references" not in st.session_state:
        st.session_state.last_references = []
    
    # Review logs view (FIXED TO SHOW RELEVANT SECTIONS AND LONGER SNIPPETS)
    if st.session_state.view_mode == "logs":
        conn = db.create_connection()
        if conn:
            # Fetch all columns including retrieved_sections (which is a JSON string)
            df = pd.read_sql("SELECT * FROM feedback WHERE username = ? ORDER BY timestamp DESC", conn, params=(username,))
            st.subheader("📜 Your Previous Interactions")
            if df.empty:
                st.info("No queries found for your account.")
            else:
                # 1. PARSE RETRIEVED SECTIONS: Extract the section numbers from the JSON string
                def parse_sections(json_str):
                    if not json_str: return "N/A"
                    try:
                        data = json.loads(json_str)
                        if not data: return "N/A"
                        
                        sections = []
                        for item in data:
                            # Case 1: New, correct logging (list of section numbers: ["302"])
                            if isinstance(item, str):
                                sections.append(item)
                            # Case 2: Old logging (list of tuples: [("302", "text...")])
                            elif isinstance(item, tuple) or isinstance(item, list) and len(item) > 0:
                                sections.append(str(item[0])) # Extract the first element (the section number)
                            # Case 3: Dict logging (if any)
                            elif isinstance(item, dict) and 'section_number' in item:
                                sections.append(item['section_number'])
                        
                        return ", ".join(sections) if sections else "N/A"
                    
                    except Exception as e:
                        # Print error to terminal for debugging
                        print(f"DEBUG LOG PARSE ERROR: {e} on data: {json_str[:50]}...") 
                        return "N/A (Parse Error)"
                
                # Display each interaction with a delete button
                for idx, row in df.iterrows():
                    col1, col2 = st.columns([0.95, 0.05])
                    
                    with col1:
                        sections = parse_sections(row['retrieved_sections'])
                        rating_text = "👍 Helpful" if row['rating'] == 1 else ("👎 Needs Review" if row['rating'] == 0 else "N/A")
                        
                        st.markdown(f"""
                        **Time:** {row['timestamp']}  
                        **Query:** {row['query'][:150]}...  
                        **Response:** {row['response_snippet'][:150]}...  
                        **Sections:** {sections} | **Rating:** {rating_text}
                        """)
                        st.divider()
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{row['id']}", help="Delete this interaction"):
                            if db.delete_single_interaction(row['id']):
                                st.toast("✅ Chat deleted!", icon="🗑️")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
            conn.close()
        else:
            st.warning("Unable to load logs.")
        return

    # Chat interface
    st.title("⚖️ LegalBot: AI-Powered Reference System")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👩‍⚖️ Hello! I’m LegalBot. Ask me any legal question."
        }]

    for i, msg in enumerate(st.session_state.messages):
        key_prefix = f"msg_{i}"
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            
            # Add buttons only to the *last* assistant message
            if i == len(st.session_state.messages) - 1 and msg["role"] == "assistant":
                
                user_q = st.session_state.messages[i-1]['content'] if i > 0 and st.session_state.messages[i-1]['role'] == 'user' else ""
                full_bot_a = msg['content']
                references = st.session_state.get('last_references', [])

                add_quick_action_buttons(key_prefix)
                add_feedback_buttons(username, user_q, full_bot_a, references, key_prefix)
                
                # Clear the last references *after* adding the buttons for the current message
                st.session_state.last_references = [] 

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat search input: pressing Enter submits via on_change handler (no visible Send button)
    if "chat_text" not in st.session_state:
        st.session_state.chat_text = ""

    def submit_search():
        prompt = st.session_state.get("chat_text", "").strip()
        if not prompt:
            return
        st.session_state.last_references = []
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show non-blocking thinking message (UI stays responsive)
        st.info("🤔 Analyzing legal statutes...")
        
        results = search_relevant_sections_semantic(prompt, db_data)
        response = generate_llm_response(prompt, results)
        st.session_state.messages.append({"role": "assistant", "content": response})
        db.log_interaction(username, prompt, response, [s for s,_ in results])

        # Clear input after submission
        st.session_state.chat_text = ""

    # Single wide input — Enter will trigger `submit_search`
    col_full = st.columns([1])[0]
    with col_full:
        user_query = st.text_input("Search", placeholder="Type your legal question here...", key="chat_text", label_visibility="collapsed", on_change=submit_search)

# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    if not os.path.exists(db.DATABASE_FILE):
        conn = db.create_connection()
        if conn:
            db.create_tables(conn)
            conn.close()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_auth_page()
    else:
        main_app_interface()