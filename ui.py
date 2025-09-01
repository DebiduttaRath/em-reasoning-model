
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

def start_auto_learning() -> Dict[str, Any]:
    """Start auto-learning process."""
    try:
        response = requests.post(f"{API_BASE_URL}/start_learning")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_learning_status() -> Dict[str, Any]:
    """Get auto-learning status."""
    try:
        response = requests.get(f"{API_BASE_URL}/learning_status")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def learn_from_url(url: str) -> Dict[str, Any]:
    """Learn from specific URL."""
    try:
        payload = {"url": url}
        response = requests.post(f"{API_BASE_URL}/learn_from_url", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def add_learning_topic(topic: str) -> Dict[str, Any]:
    """Add learning topic."""
    try:
        payload = {"topic": topic}
        response = requests.post(f"{API_BASE_URL}/add_topic", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def submit_feedback(question: str, answer: str, rating: int, feedback: str = "") -> Dict[str, Any]:
    """Submit user feedback."""
    try:
        payload = {
            "question": question,
            "answer": answer,
            "rating": rating,
            "feedback": feedback
        }
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/performance_stats")
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
        
        # Domain Experts Status
        st.header("🎯 Domain Experts")
        try:
            domain_info = requests.get(f"{API_BASE_URL}/domain_experts").json()
            if "error" not in domain_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expert Queries", domain_info.get("total_expert_queries", 0))
                with col2:
                    st.metric("Available Domains", len(domain_info.get("available_domains", [])))
                
                # Show domain distribution
                if domain_info.get("expert_distribution"):
                    st.write("**Domain Usage Distribution:**")
                    for domain, percentage in domain_info["expert_distribution"].items():
                        if percentage > 0:
                            st.write(f"• {domain}: {percentage:.1f}%")
            else:
                st.warning("Domain experts not available")
        except:
            st.warning("Could not load domain expert status")
        
        st.markdown("---")
        
        # Auto-Learning Status
        st.header("🧠 Auto-Learning")
        learning_status = get_learning_status()
        
        if "error" not in learning_status and learning_status.get("status") != "not_available":
            # Display learning status
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Items Learned", learning_status.get("total_items_learned", 0))
            with col2:
                st.metric("Topics", learning_status.get("learning_topics_count", 0))
            
            # Learning controls
            if learning_status.get("is_learning", False):
                st.warning("🔄 Currently learning...")
            else:
                if st.button("🚀 Start Learning Cycle"):
                    with st.spinner("Starting auto-learning..."):
                        result = start_auto_learning()
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"Learning started! Will process {result.get('sources_processed', 0)} sources.")
                        st.experimental_rerun()
            
            # Add custom learning topic
            with st.expander("Add Learning Topic"):
                new_topic = st.text_input("Enter topic to learn about:")
                if st.button("Add Topic"):
                    if new_topic.strip():
                        result = add_learning_topic(new_topic.strip())
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"Added topic: {new_topic}")
                            st.experimental_rerun()
            
            # Learn from URL
            with st.expander("Learn from URL"):
                url = st.text_input("Enter URL to learn from:")
                if st.button("Learn from URL"):
                    if url.strip():
                        with st.spinner("Learning from URL..."):
                            result = learn_from_url(url.strip())
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success("Successfully learned from URL!")
        else:
            st.warning("Auto-learning not available")
        
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Problem Solver", "💬 Chat Interface", "📊 Reasoning Trace", "📈 Performance", "🏢 Domain Analysis"])
    
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
                
                # Method, confidence, and domain info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Method Used", result['method'].replace('_', ' ').title())
                with col2:
                    confidence_pct = int(result['confidence'] * 100)
                    st.metric("Confidence", f"{confidence_pct}%")
                with col3:
                    domain = result.get('domain', 'general')
                    st.metric("Domain", domain.title())
                
                # Domain enhancement and compliance
                if result.get('domain_enhanced'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("✅ Domain Expert Enhanced")
                    with col2:
                        compliance = result.get('compliance_score', 0.7)
                        st.metric("Compliance Score", f"{int(compliance * 100)}%")
                    
                    # Show domain validation details
                    if result.get('domain_validation'):
                        validation = result['domain_validation']
                        with st.expander("Domain Validation Details"):
                            st.write(f"**Valid:** {validation.get('is_valid', True)}")
                            st.write(f"**Confidence:** {validation.get('confidence', 0.8):.2f}")
                            if validation.get('validation_notes'):
                                st.write("**Notes:**")
                                for note in validation['validation_notes']:
                                    st.write(f"• {note}")
                
                # Store in session state for trace view
                st.session_state['last_result'] = result
                
                # Feedback system
                st.markdown("### 💬 Rate this Response")
                col1, col2 = st.columns([1, 3])
                with col1:
                    rating = st.slider("Rating", 1, 5, 3, help="Rate the quality of this response")
                with col2:
                    feedback_text = st.text_input("Optional feedback:")
                
                if st.button("Submit Feedback"):
                    feedback_result = submit_feedback(problem, result['answer'], rating, feedback_text)
                    if "error" in feedback_result:
                        st.error(feedback_result["error"])
                    else:
                        st.success("Thank you for your feedback!")
                        st.experimental_rerun()
    
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
                        domain_info = f"Method: {message['method']} | Confidence: {int(message['confidence']*100)}%"
                        if message.get('domain') and message['domain'] != 'general':
                            domain_info += f" | Domain: {message['domain'].title()}"
                        if message.get('domain_enhanced'):
                            domain_info += " | Expert Enhanced ✅"
                        if message.get('compliance_score'):
                            domain_info += f" | Compliance: {int(message['compliance_score']*100)}%"
                        st.caption(domain_info)
        
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
    
    with tab4:
        st.header("Performance Monitoring")
        st.markdown("Track the system's learning progress and performance.")
        
        # Get performance stats
        perf_stats = get_performance_stats()
        
        if "error" not in perf_stats:
            # Overall performance
            st.markdown("### 📊 Overall Performance")
            
            overall = perf_stats.get("overall", {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Feedback", overall.get("total_feedback", 0))
            with col2:
                st.metric("Satisfaction Rate", f"{overall.get('satisfaction_rate', 0)*100:.1f}%")
            with col3:
                st.metric("Average Rating", f"{overall.get('average_rating', 0):.1f}/5")
            with col4:
                method_usage = perf_stats.get("method_usage", {})
                most_used = max(method_usage.items(), key=lambda x: x[1]) if method_usage else ("N/A", 0)
                st.metric("Most Used Method", most_used[0].upper())
            
            # Method performance
            st.markdown("### 🔧 Method Performance")
            method_perf = perf_stats.get("method_performance", {})
            
            if method_perf:
                method_data = []
                for method, stats in method_perf.items():
                    method_data.append({
                        "Method": method.upper(),
                        "Success Rate": f"{stats.get('success_rate', 0)*100:.1f}%",
                        "Avg Confidence": f"{stats.get('average_confidence', 0)*100:.1f}%",
                        "Total Attempts": stats.get('total_attempts', 0)
                    })
                
                st.dataframe(method_data)
            else:
                st.info("No method performance data available yet.")
            
            # Topic performance
            st.markdown("### 📚 Topic Performance")
            topic_perf = perf_stats.get("topic_performance", {})
            
            if topic_perf:
                topic_data = []
                for topic, stats in topic_perf.items():
                    topic_data.append({
                        "Topic": topic.title(),
                        "Success Rate": f"{stats.get('rate', 0)*100:.1f}%",
                        "Attempts": stats.get('attempts', 0),
                        "Successes": stats.get('successes', 0)
                    })
                
                st.dataframe(topic_data)
            else:
                st.info("No topic performance data available yet.")
            
            # Recent feedback
            st.markdown("### 📝 Recent Feedback")
            user_feedback = perf_stats.get("user_feedback", [])
            
            if user_feedback:
                recent_feedback = user_feedback[-5:]  # Show last 5
                for feedback in reversed(recent_feedback):
                    with st.expander(f"Rating: {feedback['rating']}/5 - {feedback['timestamp'][:19]}"):
                        st.write(f"**Question:** {feedback['question'][:100]}...")
                        st.write(f"**Answer:** {feedback['answer'][:100]}...")
                        if feedback.get('feedback'):
                            st.write(f"**Feedback:** {feedback['feedback']}")
            else:
                st.info("No feedback data available yet.")
        
        else:
            st.error("Could not load performance statistics.")
    
    with tab5:
        st.header("Domain Expert Analysis")
        st.markdown("Analyze questions with domain-specific expertise.")
        
        # Domain expert testing
        st.markdown("### 🧪 Test Domain Expert")
        test_question = st.text_area(
            "Enter a question to test domain expert selection:",
            placeholder="Example: What is the credit risk of a borrower with 700 credit score?",
            height=100
        )
        
        if st.button("🔍 Analyze Domain") and test_question.strip():
            try:
                # Get domain prompt
                payload = {"question": test_question.strip()}
                response = requests.post(f"{API_BASE_URL}/domain_prompt", json=payload)
                
                if response.status_code == 200:
                    domain_result = response.json()
                    
                    if domain_result["has_domain_expert"]:
                        st.success(f"✅ Domain Expert Available: **{domain_result['domain']}**")
                        
                        with st.expander("View Domain-Specific Prompt"):
                            st.text_area("Generated Prompt:", domain_result["prompt"], height=300, disabled=True)
                    else:
                        st.info("ℹ️ No specific domain expert available - will use general reasoning")
                else:
                    st.error("Error analyzing domain")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Domain expert statistics
        st.markdown("### 📊 Domain Expert Statistics")
        try:
            domain_stats = requests.get(f"{API_BASE_URL}/domain_experts").json()
            
            if "error" not in domain_stats:
                # Overall stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Expert Queries", domain_stats.get("total_expert_queries", 0))
                with col2:
                    st.metric("Available Domains", len(domain_stats.get("available_domains", [])))
                
                # Available domains
                st.markdown("### 🎯 Available Domain Experts")
                domains = domain_stats.get("available_domains", [])
                for i, domain in enumerate(domains):
                    usage = domain_stats.get("expert_usage", {}).get(domain, 0)
                    percentage = domain_stats.get("expert_distribution", {}).get(domain, 0)
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{domain}**")
                    with col2:
                        st.write(f"{usage} queries")
                    with col3:
                        st.write(f"{percentage:.1f}%")
                
                # Usage distribution chart
                if domain_stats.get("expert_distribution"):
                    st.markdown("### 📈 Domain Usage Distribution")
                    usage_data = []
                    for domain, percentage in domain_stats["expert_distribution"].items():
                        if percentage > 0:
                            usage_data.append({"Domain": domain, "Usage %": percentage})
                    
                    if usage_data:
                        st.bar_chart(data={row["Domain"]: row["Usage %"] for row in usage_data})
                    else:
                        st.info("No domain expert usage data yet")
            else:
                st.error("Could not load domain expert statistics")
        except Exception as e:
            st.error(f"Error loading domain statistics: {str(e)}")
        
        # Compliance testing
        st.markdown("### ⚖️ Compliance Checker")
        with st.expander("Test Reasoning Compliance"):
            compliance_question = st.text_input("Question:")
            compliance_reasoning = st.text_area("Reasoning:", height=150)
            compliance_answer = st.text_input("Answer:")
            
            if st.button("Check Compliance") and all([compliance_question, compliance_reasoning, compliance_answer]):
                try:
                    payload = {
                        "question": compliance_question,
                        "reasoning": compliance_reasoning,
                        "answer": compliance_answer
                    }
                    response = requests.post(f"{API_BASE_URL}/compliance_check", json=payload)
                    
                    if response.status_code == 200:
                        validation = response.json()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            status = "✅ Valid" if validation.get("is_valid", True) else "❌ Invalid"
                            st.write(f"**Validation:** {status}")
                        with col2:
                            compliance_score = validation.get("compliance_score", 0.7)
                            st.metric("Compliance Score", f"{int(compliance_score * 100)}%")
                        
                        if validation.get("validation_notes"):
                            st.write("**Validation Notes:**")
                            for note in validation["validation_notes"]:
                                st.write(f"• {note}")
                    else:
                        st.error("Error checking compliance")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using the **Build a Reasoning Model (From Scratch)** architecture | "
        "[GitHub Repository](https://github.com/rasbt/reasoning-from-scratch)"
    )

if __name__ == "__main__":
    main()
