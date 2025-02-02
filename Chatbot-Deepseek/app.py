# Import required libraries
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# --------------------------
# 1. ENHANCED STYLING & UX
# --------------------------
# Added code block styling and improved dark theme
st.markdown("""
<style>
    /* Base styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    
    /* Code block styling */
    pre {
        background-color: #2d2d2d !important;
        border-radius: 8px !important;
        padding: 15px !important;
        border: 1px solid #4d4d4d !important;
    }
    
    /* Improved dropdown styling */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg { fill: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# --------------------------
# 2. CONTEXT MANAGEMENT
# --------------------------
# Added context trimming and system prompt optimization
def trim_context(messages, max_tokens=4000):
    """Smart context trimming with token awareness"""
    token_count = 0
    trimmed = []
    system_prompt_found = False
    
    for msg in reversed(messages):
        content = msg["content"]
        tokens = len(content.split())  # Simplified token estimation
        
        # Preserve system prompt if not found yet
        if msg["role"] == "system" and not system_prompt_found:
            trimmed.insert(0, msg)
            system_prompt_found = True
            continue
            
        if token_count + tokens > max_tokens:
            break
            
        token_count += tokens
        trimmed.insert(0, msg)
        
    return trimmed

# --------------------------
# 3. IMPROVED CONFIGURATION
# --------------------------
# Added model parameters and server configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:32b"],
        index=0
    )
    
    # Server configuration
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_context = st.number_input("Max Context (tokens)", 1000, 8000, 4000)
    
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# --------------------------
# 4. ERROR HANDLING
# --------------------------
# Wrapped LLM initialization in error handler
try:
    llm_engine = ChatOllama(
        model=selected_model,
        base_url=ollama_url,
        temperature=temperature,
        timeout=30  # Added timeout
    )
except Exception as e:
    st.error(f"Failed to initialize model: {str(e)}")
    st.stop()

# --------------------------
# 5. IMPROVED PROMPT ENGINEERING
# --------------------------
# Enhanced system prompt with debugging focus
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert AI coding assistant specializing in Python development and debugging. Follow these rules:
1. Provide concise, correct solutions with strategic print statements
2. Highlight potential error points using comments
3. Format code blocks with proper syntax
4. Include brief explanations of complex logic
5. Always respond in English"""
)

# --------------------------
# 6. SESSION STATE MANAGEMENT
# --------------------------
# Added conversation reset and message formatting
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "system", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
    ]

if st.button("🧹 Clear Conversation"):
    st.session_state.message_log = [
        {"role": "system", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
    ]
    st.rerun()

# --------------------------
# 7. IMPROVED CHAT DISPLAY
# --------------------------
# Added code formatting and message styling
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        if message["role"] == "system":
            continue  # Skip system greeting in display
            
        with st.chat_message(message["role"]):
            content = message["content"]
            if "```" in content:
                code_block = content.split("```")[1]
                st.code(code_block, language="python")
                if len(content.split("```")) > 2:
                    st.markdown(content.split("```")[2])
            else:
                st.markdown(content)

# --------------------------
# 8. ENHANCED PROCESSING PIPELINE
# --------------------------
def build_prompt_chain():
    """Build context-aware prompt with smart trimming"""
    prompt_sequence = [system_prompt]
    relevant_history = trim_context(
        st.session_state.message_log,
        max_tokens=max_context
    )
    
    for msg in relevant_history:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
            
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain):
    """Robust response generation with error handling"""
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return "⚠️ Sorry, I'm having trouble connecting. Please try again."

# --------------------------
# 9. CHAT PROCESSING
# --------------------------
user_query = st.chat_input("Type your coding question here...")
if user_query:
    # Validate input
    if not user_query.strip():
        st.warning("Please enter a valid question")
        st.stop()
    
    # Add to message log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Process response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Store and display response
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()