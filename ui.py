
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Beast Reasoning LLM 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://0.0.0.0:8000"

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_methods() -> Dict[str, Any]:
    """Get available reasoning methods from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/methods")
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def solve_problem(prompt: str, method: str = "auto", use_context: bool = True) -> Dict[str, Any]:
    """Send problem to API for solving."""
    try:
        payload = {
            "prompt": prompt,
            "method": method,
            "use_context": use_context
        }
        response = requests.post(f"{API_BASE_URL}/solve", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def chat_with_model(message: str, method: str = "auto", session_id: str = "default") -> Dict[str, Any]:
    """Send chat message to API."""
    try:
        payload = {
            "message": message,
            "method": method,
            "session_id": session_id
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def add_documents(documents: list) -> Dict[str, Any]:
    """Add knowledge documents via API."""
    try:
        payload = {"documents": documents}
        response = requests.post(f"{API_BASE_URL}/add_documents", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

# Main UI
def main():
    st.title("🧠 Beast Reasoning LLM")
    st.markdown("*A state-of-the-art reasoning language model with multiple reasoning strategies*")
    
    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("⚠️ API server is not running. Please start the server with: `python server.py`")
        st.stop()
    
    st.success("✅ API server is running")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Get available methods
        methods_info = get_available_methods()
        method_options = methods_info.get("methods", ["auto", "cot", "pal", "self_consistency", "tot"])
        method_descriptions = methods_info.get("descriptions", {})
        
        # Method selection
        selected_method = st.selectbox(
            "Reasoning Method",
            method_options,
            help="Choose the reasoning strategy"
        )
        
        # Show method description
        if selected_method in method_descriptions:
            st.info(f"**{selected_method.upper()}**: {method_descriptions[selected_method]}")
        
        # Context usage
        use_context = st.checkbox("Use Knowledge Context", value=True, help="Include relevant context from knowledge base")
        
        st.markdown("---")
        
        # Knowledge Management
        st.header("📚 Knowledge Base")
        
        with st.expander("Add Documents"):
            new_doc = st.text_area("Enter new knowledge document:")
            if st.button("Add Document"):
                if new_doc.strip():
                    result = add_documents([new_doc.strip()])
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Document added successfully!")
                        st.experimental_rerun()
        
        # Session management
        if st.button("End Session"):
            try:
                requests.post(f"{API_BASE_URL}/end_session")
                st.success("Session ended")
            except:
                st.error("Failed to end session")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Problem Solver", "💬 Chat Interface", "📊 Reasoning Trace"])
    
    with tab1:
        st.header("Problem Solver")
        st.markdown("Enter a problem that requires reasoning. The system will analyze and solve it step by step.")
        
        # Problem input
        problem = st.text_area(
            "Enter your problem:",
            placeholder="Example: A farmer has 17 sheep. All but 9 die. How many sheep are left?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            solve_button = st.button("🚀 Solve Problem", type="primary")
        
        if solve_button and problem.strip():
            with st.spinner("🤔 Reasoning..."):
                result = solve_problem(problem.strip(), selected_method, use_context)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display results
                st.success("✅ Problem Solved!")
                
                # Answer
                st.markdown("### 🎯 Answer")
                st.markdown(f"**{result['answer']}**")
                
                # Method and confidence
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Method Used", result['method'].replace('_', ' ').title())
                with col2:
                    confidence_pct = int(result['confidence'] * 100)
                    st.metric("Confidence", f"{confidence_pct}%")
                
                # Store in session state for trace view
                st.session_state['last_result'] = result
    
    with tab2:
        st.header("Chat Interface")
        st.markdown("Have a conversation with the reasoning model.")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        chat_input = st.chat_input("Ask me anything...")
        
        if chat_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                result = chat_with_model(chat_input, selected_method)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Add assistant response to history
                assistant_msg = {
                    "role": "assistant", 
                    "content": result["response"],
                    "reasoning": result["reasoning"],
                    "method": result["method"],
                    "confidence": result["confidence"]
                }
                st.session_state.chat_history.append(assistant_msg)
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if message["role"] == "assistant" and "method" in message:
                    with st.expander("View Reasoning"):
                        st.text(message["reasoning"])
                        st.caption(f"Method: {message['method']} | Confidence: {int(message['confidence']*100)}%")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    with tab3:
        st.header("Reasoning Trace")
        st.markdown("Explore the step-by-step reasoning process.")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Reasoning details
            st.markdown("### 🧠 Detailed Reasoning")
            st.text_area("Reasoning Process:", result['reasoning'], height=200, disabled=True)
            
            # Reasoning trace
            if result.get('trace'):
                st.markdown("### 🔍 Reasoning Trace")
                for i, step in enumerate(result['trace']):
                    with st.expander(f"Step {i+1}"):
                        st.text(step)
            
            # Context used
            if result.get('context_used'):
                st.markdown("### 📚 Context Used")
                st.text_area("Retrieved Context:", result['context_used'], height=150, disabled=True)
            
            # Method analysis
            st.markdown("### 📈 Method Analysis")
            method_info = {
                "Method": result['method'].replace('_', ' ').title(),
                "Confidence": f"{int(result['confidence'] * 100)}%",
                "Steps": len(result.get('trace', [])),
                "Context Used": "Yes" if result.get('context_used') else "No"
            }
            
            for key, value in method_info.items():
                st.metric(key, value)
        
        else:
            st.info("Solve a problem first to see the reasoning trace.")

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using the **Build a Reasoning Model (From Scratch)** architecture | "
        "[GitHub Repository](https://github.com/rasbt/reasoning-from-scratch)"
    )

if __name__ == "__main__":
    main()
