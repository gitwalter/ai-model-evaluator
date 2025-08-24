#!/usr/bin/env python3
"""
Refactored Streamlit app for interactive testing of Gemini models.
Features model selection, custom prompts, system prompts, and response evaluation.
Uses modular architecture with object-oriented design.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# Import modular components
from models.model_manager import ModelManager
from evaluation.response_evaluator import ResponseEvaluator
from execution.code_executor import CodeExecutor
from ui.components import UIComponents
from utils.prompt_manager import PromptManagerUtils


class AIModelEvaluatorApp:
    """Main application class for AI Model Evaluator."""
    
    def __init__(self):
        """Initialize the application with all required components."""
        self.model_manager = ModelManager()
        self.evaluator = ResponseEvaluator()
        self.code_executor = CodeExecutor()
        self.ui_components = UIComponents()
        self.prompt_utils = PromptManagerUtils()
        
        # Initialize session state
        if "test_results" not in st.session_state:
            st.session_state.test_results = []
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = []
    
    def setup_page(self):
        """Setup the Streamlit page configuration."""
        st.set_page_config(
            page_title="AI Model Evaluator",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better response formatting
        st.markdown("""
        <style>
        .response-container {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
        }
        .response-container pre {
            background-color: #f1f3f4;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
        }
        .response-container code {
            background-color: #f1f3f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        .metric-container {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🤖 AI Model Evaluator")
        st.markdown("Interactive testing and evaluation of AI models (currently Gemini)")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("Configuration")
        
        # API Key Management Section
        self._render_api_key_section()
        
        # Get available models dynamically
        available_models = self.model_manager.get_available_models()
        
        # Model selection with multi-select capability
        st.sidebar.subheader("🤖 Model Selection")
        
        # Check if we have models available
        if not available_models:
            st.sidebar.warning("⚠️ No models available. Please configure your API key first.")
            return [], "", None
        
        # Test mode selection
        test_mode = st.sidebar.radio(
            "Test Mode",
            ["Single Model", "Multiple Models"],
            help="Choose between testing a single model or multiple models"
        )
        
        # Show current mode status
        if test_mode == "Single Model":
            st.sidebar.info("🔄 Single Model Mode")
        else:
            st.sidebar.info("🔄 Multiple Models Mode")
        
        if test_mode == "Single Model":
            col1, col2 = st.columns([3, 1])
            with col1:
                # Set default model to models/gemini-2.5-flash if available, otherwise use first available
                default_model_index = 0
                if "models/gemini-2.5-flash" in available_models:
                    default_model_index = available_models.index("models/gemini-2.5-flash")
                    st.sidebar.success(f"✅ Default model set to: models/gemini-2.5-flash")
                else:
                    st.sidebar.info(f"ℹ️ models/gemini-2.5-flash not available, using: {available_models[0] if available_models else 'None'}")
                
                selected_models = [st.selectbox(
                    "Select Model",
                    available_models,
                    index=default_model_index,
                    help="Choose the Gemini model to test (models/gemini-2.5-flash is recommended)"
                )]
            with col2:
                if st.button("🔄", help="Refresh model list and clear cache"):
                    # Clear cached models to force refresh
                    self.model_manager._available_models = None
                    self.model_manager._last_model_fetch = 0
                    st.rerun()
        else:
            # Multi-model selection
            # Initialize session state for selected models
            if "selected_models" not in st.session_state:
                st.session_state.selected_models = []
                
            if not st.session_state.selected_models:
                # Prioritize models/gemini-2.5-flash as first model if available
                default_models = []
                if "models/gemini-2.5-flash" in available_models:
                    default_models.append("models/gemini-2.5-flash")
                
                # Add other models up to 3 total
                remaining_models = [m for m in available_models if m != "models/gemini-2.5-flash"]
                default_models.extend(remaining_models[:3 - len(default_models)])
                
                st.session_state.selected_models = default_models
            
            selected_models = st.multiselect(
                "Select Models",
                available_models,
                default=st.session_state.selected_models,
                help="Choose multiple models to test simultaneously"
            )
            
            # Update session state
            st.session_state.selected_models = selected_models
            
            # Quick selection buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Select All", help="Select all available models"):
                    st.session_state.selected_models = available_models
                    st.rerun()
            with col2:
                if st.button("Select Fast", help="Select fast models (Flash variants)"):
                    fast_models = [m for m in available_models if "flash" in m.lower()]
                    st.session_state.selected_models = fast_models
                    st.rerun()
            with col3:
                if st.button("Clear All", help="Clear all selections"):
                    st.session_state.selected_models = []
                    st.rerun()
            
            # Show selected models count
            if selected_models:
                st.sidebar.success(f"Selected {len(selected_models)} model(s)")
            else:
                st.sidebar.warning("No models selected")
        
        # Prompt Management Section
        st.sidebar.header("📝 Prompt Management")
        
        # Initialize prompt database
        db = self.prompt_utils.get_prompt_database()
        
        # Get system prompt selection
        system_prompt = self.prompt_utils.get_system_prompt_selection(db)
        
        # Get test prompt selection
        self.prompt_utils.get_test_prompt_selection(db)
        
        return selected_models, system_prompt, db
    
    def _render_api_key_section(self):
        """Render the API key management section."""
        st.sidebar.subheader("🔑 API Key Configuration")
        
        # Get current API key status
        api_status = self.model_manager.get_api_key_status()
        
        if api_status['has_api_key'] and api_status['is_valid']:
            st.sidebar.success("✅ API Key configured and valid")
            
            # Show API key management options
            with st.sidebar.expander("🔧 API Key Options", expanded=False):
                if st.button("🔄 Refresh API Key", help="Re-validate the current API key"):
                    # Force refresh by clearing cache
                    self.model_manager._available_models = None
                    self.model_manager._last_model_fetch = 0
                    st.rerun()
                
                if st.button("🗑️ Clear API Key", help="Clear the current API key"):
                    # Clear the API key from session state
                    if "api_key" in st.session_state:
                        del st.session_state.api_key
                    st.rerun()
                
                if st.button("🧹 Clear Session State", help="Clear all session state and refresh"):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        else:
            st.sidebar.warning("⚠️ API Key not configured")
            
            # API key input section
            with st.sidebar.expander("🔑 Enter API Key", expanded=True):
                st.write("**Google Gemini API Key Required**")
                st.write("Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
                
                # Use session state to persist the API key
                if "api_key" not in st.session_state:
                    st.session_state.api_key = ""
                
                api_key = st.text_input(
                    "API Key",
                    value=st.session_state.api_key,
                    type="password",
                    help="Enter your Google Gemini API key",
                    placeholder="AIzaSyC..."
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("✅ Test & Save", help="Test the API key and save it"):
                        if api_key.strip():
                            if self.model_manager.set_api_key(api_key):
                                st.session_state.api_key = api_key
                                st.success("✅ API Key is valid and saved!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid API key. Please check and try again.")
                        else:
                            st.error("❌ Please enter a valid API key.")
                
                with col2:
                    if st.button("📋 Load from Secrets", help="Try to load API key from Streamlit secrets"):
                        # Try to reload from secrets
                        self.model_manager._configure_api()
                        if self.model_manager._api_key:
                            st.session_state.api_key = self.model_manager._api_key
                            st.success("✅ API Key loaded from Streamlit secrets!")
                            st.rerun()
                        else:
                            st.error("❌ No API key found in Streamlit secrets")
                
                # Help text
                st.caption("💡 **Tip:** You can also configure secrets in Streamlit Cloud or create a `.streamlit/secrets.toml` file")
                
                # Show secrets file example
                with st.expander("📄 Secrets Configuration", expanded=False):
                    st.write("**Option 1: Streamlit Cloud Secrets**")
                    st.write("1. Go to your app in Streamlit Cloud")
                    st.write("2. Navigate to Settings → Secrets")
                    st.write("3. Add: `GEMINI_API_KEY = 'your_actual_api_key_here'`")
                    
                    st.write("**Option 2: Local Development**")
                    st.write("Create `.streamlit/secrets.toml` file:")
                    st.code("""
# Create .streamlit/secrets.toml file
GEMINI_API_KEY = "AIzaSyC_your_actual_api_key_here"
                    """, language="toml")
                    
                    st.write("**Option 3: Environment Variable (Legacy)**")
                    st.write("Create `.env` file in project root:")
                    st.code("""
# Create .env file in project root
GEMINI_API_KEY=AIzaSyC_your_actual_api_key_here
                    """, language="bash")
    
    def render_input_section(self):
        """Render the input section for prompts."""
        st.header("📝 Input")
        
        # Prompt input with more space
        default_prompt = "Write a Python function to calculate the factorial of a number and explain how it works."
        
        # Check if we have a pre-filled prompt from user prompt selection
        if "prefill_prompt" in st.session_state:
            default_prompt = st.session_state.prefill_prompt
            # Clear the pre-filled prompt after using it
            del st.session_state.prefill_prompt
        
        prompt = st.text_area(
            "Enter your prompt",
            value=default_prompt,
            height=150,
            help="Enter the prompt you want to send to the model"
        )
        
        return prompt
    
    def render_test_button(self, selected_models: List[str], prompt: str, system_prompt: str):
        """Render the test button and handle testing."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if len(selected_models) == 1:
                button_text = f"🚀 Test {selected_models[0]}"
            else:
                button_text = f"🚀 Test {len(selected_models)} Models" if selected_models else "🚀 Test Models"
            
            if st.button(button_text, type="primary", use_container_width=True, disabled=not selected_models):
                if not prompt.strip():
                    st.error("Please enter a prompt")
                elif not selected_models:
                    st.error("Please select at least one model")
                else:
                    self.run_tests(selected_models, prompt, system_prompt)
    
    def run_tests(self, selected_models: List[str], prompt: str, system_prompt: str):
        """Run tests on selected models."""
        # Clear previous results for this test session
        st.session_state.test_results = []
        
        if len(selected_models) > 1:
            # Multi-model testing
            with st.spinner(f"Testing {len(selected_models)} models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Testing {model_name}... ({i+1}/{len(selected_models)})")
                    
                    result = self.model_manager.test_model(model_name, prompt, system_prompt)
                    results.append(result)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(selected_models))
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"Completed testing {len(selected_models)} models!")
                
                # Store results
                st.session_state.test_results = results
        else:
            # Single model test
            with st.spinner(f"Testing {selected_models[0]}..."):
                result = self.model_manager.test_model(selected_models[0], prompt, system_prompt)
                
                # Store result in session state
                st.session_state.test_results = [result]
                
                st.success(f"Test completed in {result.get('response_time', 0):.2f} seconds!")
    
    def render_results_section(self):
        """Render the results section with tabs."""
        # Initialize session state if not exists
        if "test_results" not in st.session_state:
            st.session_state.test_results = []
        
        if not st.session_state.test_results:
            return
        
        # Use all current test results (they're already filtered for this test session)
        latest_results = st.session_state.test_results
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Model Response", "📊 Evaluation", "📈 Comparison", "🚀 Code Execution", "📝 Prompt Manager"])
        
        with tab1:
            self.render_model_response_tab(latest_results)
        
        with tab2:
            self.render_evaluation_tab(latest_results)
        
        with tab3:
            self.render_comparison_tab()
        
        with tab4:
            self.render_code_execution_tab(latest_results)
        
        with tab5:
            self.render_prompt_manager_tab()
    
    def render_model_response_tab(self, latest_results: List[Dict[str, Any]]):
        """Render the model response tab."""
        if len(latest_results) == 1:
            # Single result display
            result = latest_results[0]
            st.header(f"Response from {result['model_name']}")
            
            if result["status"] == "success":
                self.ui_components.display_single_response(result, self.evaluator)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            # Multiple results display
            st.header(f"Responses from {len(latest_results)} Models")
            
            # Create tabs for each model response
            model_tabs = st.tabs([f"🤖 {result['model_name']}" for result in latest_results])
            
            for i, (result, tab) in enumerate(zip(latest_results, model_tabs)):
                with tab:
                    if result["status"] == "success":
                        self.ui_components.display_single_response(result, self.evaluator)
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    def render_evaluation_tab(self, latest_results: List[Dict[str, Any]]):
        """Render the evaluation tab."""
        if len(latest_results) == 1:
            # Single result evaluation
            result = latest_results[0]
            if result["status"] == "success":
                st.header(f"📊 Response Evaluation - {result['model_name']}")
                
                # Evaluation
                analysis = self.evaluator.analyze_response(result["response"])
                self.ui_components.display_evaluation_dashboard(result, analysis, self.evaluator)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            # Multiple results evaluation
            st.header("📊 Multi-Model Evaluation")
            self.ui_components.display_comparison_table(latest_results, self.evaluator)
    
    def render_comparison_tab(self):
        """Render the comparison tab."""
        # Initialize session state if not exists
        if "test_results" not in st.session_state:
            st.session_state.test_results = []
            
        if len(st.session_state.test_results) > 1:
            st.header("📈 Model Comparison")
            
            # Filter options
            col1, col2 = st.columns([2, 1])
            with col1:
                # Show last N results
                max_results = st.slider("Show last N results", 5, 50, 10, help="Number of recent results to compare")
            with col2:
                # Filter by status
                show_only_successful = st.checkbox("Show only successful", value=True, help="Filter out failed tests")
            
            # Create comparison table
            comparison_data = []
            recent_results = st.session_state.test_results[-max_results:]
            
            for result in recent_results:
                if show_only_successful and result["status"] != "success":
                    continue
                    
                analysis = self.evaluator.analyze_response(result["response"])
                overall_score = self.evaluator.calculate_overall_score(analysis)
                
                comparison_data.append({
                    "Model": result["model_name"],
                    "Status": result["status"],
                    "Time (s)": f"{result.get('response_time', 0):.2f}",
                    "Words": analysis["response_length"]["details"],
                    "Overall Score": f"{overall_score:.2f}",
                    "Code": "✅" if analysis["code_quality"]["score"] > 0 else "❌",
                    "Exec": "✅" if analysis["execution_attempt"]["score"] > 0 else "❌",
                    "Summary": "✅" if analysis["content_summary"]["score"] > 0 else "❌",
                    "Error Handling": "✅" if analysis["error_handling"]["score"] > 0 else "❌",
                    "Documentation": "✅" if analysis["documentation"]["score"] > 0 else "❌"
                })
            
            if comparison_data:
                st.dataframe(comparison_data, use_container_width=True)
                
                # Add some visual charts if we have enough data
                if len(comparison_data) >= 2:
                    st.subheader("📊 Performance Trends")
                    
                    # Extract data for visualization
                    models = [row["Model"] for row in comparison_data]
                    times = [float(row["Time (s)"]) for row in comparison_data]
                    scores = [float(row["Overall Score"]) for row in comparison_data]
                    
                    # Create a DataFrame for visualization
                    chart_data = pd.DataFrame({
                        "Model": models,
                        "Response Time (s)": times,
                        "Overall Score": scores
                    })
                    
                    # Response time bar chart
                    st.bar_chart(chart_data.set_index("Model")["Response Time (s)"])
                    
                    # Score comparison line chart
                    st.line_chart(chart_data.set_index("Model")["Overall Score"])
            else:
                st.info("No results to compare. Run some tests first.")
        else:
            st.info("Run multiple tests to see comparison data.")
    
    def render_code_execution_tab(self, latest_results: List[Dict[str, Any]]):
        """Render the code execution tab."""
        if len(latest_results) == 1:
            # Single result code execution
            result = latest_results[0]
            if result["status"] == "success":
                st.header(f"🚀 Code Execution - {result['model_name']}")
                self.ui_components.display_code_execution(result["response"], self.code_executor)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            # Multiple results code execution
            st.header(f"🚀 Code Execution - {len(latest_results)} Models")
            
            # Create tabs for each model's code execution
            exec_tabs = st.tabs([f"🚀 {result['model_name']}" for result in latest_results])
            
            for i, (result, tab) in enumerate(zip(latest_results, exec_tabs)):
                with tab:
                    if result["status"] == "success":
                        self.ui_components.display_code_execution(result["response"], self.code_executor)
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    def render_prompt_manager_tab(self):
        """Render the prompt manager tab."""
        st.header("📝 Prompt Manager")
        
        # Initialize prompt database
        db = self.prompt_utils.get_prompt_database()
        
        # Quick actions
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Initialize Default Prompts", help="Add default prompts if database is empty"):
                from prompt_manager import initialize_default_prompts
                initialize_default_prompts(db)
                st.success("Default prompts initialized!")
                st.rerun()
        
        with col2:
            if st.button("📊 Database Stats", help="Show database statistics"):
                system_count = len(db.get_system_prompts())
                user_count = len(db.get_user_prompts())
                st.info(f"System Prompts: {system_count} | User Prompts: {user_count}")
        
        # Prompt management interface
        prompt_mgmt_type = st.selectbox(
            "Manage",
            ["System Prompts", "Test Prompt Examples"],
            help="Choose which type of prompts to manage"
        )
        
        if prompt_mgmt_type == "System Prompts":
            self.prompt_utils.render_system_prompt_manager(db)
        else:
            self.prompt_utils.render_test_prompt_manager(db)
    
    def render_export_section(self):
        """Render the export section."""
        # Initialize session state if not exists
        if "test_results" not in st.session_state:
            st.session_state.test_results = []
            
        self.ui_components.display_export_section(st.session_state.test_results)
        
        if st.session_state.test_results and st.sidebar.button("🗑️ Clear Results"):
            st.session_state.test_results = []
            st.rerun()
    
    def run(self):
        """Run the main application."""
        self.setup_page()
        
        # Render sidebar and get configuration
        selected_models, system_prompt, db = self.render_sidebar()
        
        # Render input section
        prompt = self.render_input_section()
        
        # Render test button
        self.render_test_button(selected_models, prompt, system_prompt)
        
        # Render results section
        self.render_results_section()
        
        # Render export section
        self.render_export_section()


def main():
    """Main function to run the application."""
    app = AIModelEvaluatorApp()
    app.run()


if __name__ == "__main__":
    main()
