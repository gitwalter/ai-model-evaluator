#!/usr/bin/env python3
"""
Prompt Manager Utilities for AI Model Evaluator.
Contains utility functions for prompt management and database operations.
"""

import streamlit as st
from prompt_manager import PromptDatabase, initialize_default_prompts
from typing import Dict, Any, List


class PromptManagerUtils:
    """Utility functions for prompt management."""
    
    @staticmethod
    @st.cache_resource
    def get_prompt_database():
        """Get or create the prompt database with caching."""
        db = PromptDatabase()
        # Initialize with default prompts if database is empty
        system_prompts = db.get_system_prompts()
        if not system_prompts:
            initialize_default_prompts(db)
        return db
    
    @staticmethod
    def render_system_prompt_manager(db: PromptDatabase):
        """Render the system prompt management interface."""
        st.subheader("System Prompts Management")
        
        # Add new system prompt
        with st.expander("➕ Add New System Prompt", expanded=False):
            with st.form("add_system_prompt"):
                new_name = st.text_input("Prompt Name", help="Unique name for the system prompt")
                new_category = st.selectbox("Category", ["Development", "Documentation", "Security", "Performance", "Python", "Data Science", "Other"])
                new_description = st.text_input("Description", help="Brief description of the prompt's purpose")
                new_content = st.text_area("Prompt Content", height=200, help="The system prompt content")
                
                if st.form_submit_button("Add System Prompt"):
                    if new_name and new_content:
                        if db.add_system_prompt(new_name, new_content, new_category, new_description):
                            st.success(f"System prompt '{new_name}' added successfully!")
                            st.rerun()
                    else:
                        st.error("Please provide both name and content.")
        
        # Display existing system prompts
        st.subheader("Existing System Prompts")
        system_prompts = db.get_system_prompts()
        
        if system_prompts:
            # Group by category
            categories = db.get_categories("system")
            for category in categories:
                category_prompts = [p for p in system_prompts if p["category"] == category]
                if category_prompts:
                    st.write(f"**{category}**")
                    for prompt in category_prompts:
                        with st.expander(f"📝 {prompt['name']}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**Description:** {prompt['description']}")
                                st.write(f"**Updated:** {prompt['updated_at']}")
                            with col2:
                                if st.button("🗑️ Delete", key=f"del_sys_{prompt['id']}"):
                                    if db.delete_system_prompt(prompt['id']):
                                        st.success("Prompt deleted!")
                                        st.rerun()
                            
                            # Edit form
                            with st.form(f"edit_sys_{prompt['id']}"):
                                edit_name = st.text_input("Name", value=prompt['name'], key=f"name_sys_{prompt['id']}")
                                edit_category = st.selectbox("Category", categories, index=categories.index(prompt['category']), key=f"cat_sys_{prompt['id']}")
                                edit_description = st.text_input("Description", value=prompt['description'], key=f"desc_sys_{prompt['id']}")
                                edit_content = st.text_area("Content", value=prompt['content'], height=150, key=f"content_sys_{prompt['id']}")
                                
                                if st.form_submit_button("Update"):
                                    if edit_name and edit_content:
                                        if db.update_system_prompt(prompt['id'], edit_name, edit_content, edit_category, edit_description):
                                            st.success("Prompt updated!")
                                            st.rerun()
                                    else:
                                        st.error("Please provide both name and content.")
        else:
            st.info("No system prompts found. Add some using the form above.")
    
    @staticmethod
    def render_test_prompt_manager(db: PromptDatabase):
        """Render the test prompt examples management interface."""
        st.subheader("Test Prompt Examples Management")
        
        # Add new test prompt example
        with st.expander("➕ Add New Test Prompt Example", expanded=False):
            with st.form("add_test_prompt"):
                new_name = st.text_input("Prompt Name", help="Unique name for the test prompt example")
                new_category = st.selectbox("Category", ["LangChain", "LangGraph", "RAG", "MCP", "Pandas", "NumPy", "BeautifulSoup", "Scikit-learn", "TensorFlow", "Selenium", "FastAPI", "OpenCV", "Other"])
                new_description = st.text_input("Description", help="Brief description of the test prompt's purpose")
                new_tags = st.text_input("Tags", help="Comma-separated tags for easy searching")
                new_content = st.text_area("Prompt Content", height=200, help="The ready-to-use test prompt content")
                
                if st.form_submit_button("Add Test Prompt Example"):
                    if new_name and new_content:
                        if db.add_user_prompt(new_name, new_content, new_category, new_description, new_tags):
                            st.success(f"Test prompt example '{new_name}' added successfully!")
                            st.rerun()
                    else:
                        st.error("Please provide both name and content.")
        
        # Display existing test prompt examples
        st.subheader("Existing Test Prompt Examples")
        test_prompts = db.get_user_prompts()
        
        if test_prompts:
            # Group by category
            categories = db.get_categories("user")
            for category in categories:
                category_prompts = [p for p in test_prompts if p["category"] == category]
                if category_prompts:
                    st.write(f"**{category}**")
                    for prompt in category_prompts:
                        with st.expander(f"🧪 {prompt['name']}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**Description:** {prompt['description']}")
                                if prompt['tags']:
                                    st.write(f"**Tags:** {prompt['tags']}")
                                st.write(f"**Updated:** {prompt['updated_at']}")
                            with col2:
                                if st.button("🗑️ Delete", key=f"del_test_{prompt['id']}"):
                                    if db.delete_user_prompt(prompt['id']):
                                        st.success("Test prompt example deleted!")
                                        st.rerun()
                            
                            # Edit form
                            with st.form(f"edit_test_{prompt['id']}"):
                                edit_name = st.text_input("Name", value=prompt['name'], key=f"name_test_{prompt['id']}")
                                edit_category = st.selectbox("Category", categories, index=categories.index(prompt['category']), key=f"cat_test_{prompt['id']}")
                                edit_description = st.text_input("Description", value=prompt['description'], key=f"desc_test_{prompt['id']}")
                                edit_tags = st.text_input("Tags", value=prompt['tags'], key=f"tags_test_{prompt['id']}")
                                edit_content = st.text_area("Content", value=prompt['content'], height=150, key=f"content_test_{prompt['id']}")
                                
                                if st.form_submit_button("Update"):
                                    if edit_name and edit_content:
                                        if db.update_user_prompt(prompt['id'], edit_name, edit_content, edit_category, edit_description, edit_tags):
                                            st.success("Test prompt example updated!")
                                            st.rerun()
                                    else:
                                        st.error("Please provide both name and content.")
        else:
            st.info("No test prompt examples found. Add some using the form above.")
    
    @staticmethod
    def get_system_prompt_selection(db: PromptDatabase) -> str:
        """Get system prompt selection from sidebar."""
        st.sidebar.subheader("🔧 System Prompt")
        system_prompts = db.get_system_prompts()
        
        if system_prompts:
            # Create a dictionary for easy lookup
            system_prompt_dict = {prompt["name"]: prompt["content"] for prompt in system_prompts}
            system_prompt_dict["None"] = ""
            
            # System prompt selection with default
            # Set "Code Producer (Default)" as default if available, fallback to "Python Development Expert"
            default_index = 0
            if "Code Producer (Default)" in system_prompt_dict:
                default_index = list(system_prompt_dict.keys()).index("Code Producer (Default)")
            elif "Python Development Expert" in system_prompt_dict:
                default_index = list(system_prompt_dict.keys()).index("Python Development Expert")
            
            selected_system_prompt_name = st.sidebar.selectbox(
                "Select System Prompt",
                list(system_prompt_dict.keys()),
                index=default_index,
                help="Choose a system prompt or select 'None'. Code Producer (Default) is set as default for faster responses."
            )
            
            system_prompt = system_prompt_dict[selected_system_prompt_name]
            
            # Show selected prompt details
            if selected_system_prompt_name != "None":
                selected_prompt = next((p for p in system_prompts if p["name"] == selected_system_prompt_name), None)
                if selected_prompt:
                    with st.sidebar.expander("📋 System Prompt Details", expanded=False):
                        st.write(f"**Category:** {selected_prompt['category']}")
                        st.write(f"**Description:** {selected_prompt['description']}")
                        st.write(f"**Updated:** {selected_prompt['updated_at']}")
        else:
            system_prompt = ""
            st.sidebar.info("No system prompts available")
        
        # Custom system prompt option
        if st.sidebar.checkbox("Use custom system prompt"):
            system_prompt = st.sidebar.text_area(
                "Custom System Prompt",
                value=system_prompt,
                height=150,
                help="Enter a custom system prompt for the model"
            )
        
        return system_prompt
    
    @staticmethod
    def get_test_prompt_selection(db: PromptDatabase) -> str:
        """Get test prompt selection from sidebar."""
        st.sidebar.subheader("🧪 Test Prompt Examples")
        test_prompts = db.get_user_prompts()
        
        if test_prompts:
            # Create a dictionary for easy lookup
            test_prompt_dict = {prompt["name"]: prompt["content"] for prompt in test_prompts}
            
            # Test prompt selection
            selected_test_prompt_name = st.sidebar.selectbox(
                "Select Test Prompt Example",
                ["None"] + list(test_prompt_dict.keys()),
                help="Choose a ready-to-use test prompt example or select 'None'"
            )
            
            if selected_test_prompt_name != "None":
                # Pre-fill the main prompt input
                st.session_state.prefill_prompt = test_prompt_dict[selected_test_prompt_name]
                
                # Show selected prompt details
                selected_prompt = next((p for p in test_prompts if p["name"] == selected_test_prompt_name), None)
                if selected_prompt:
                    with st.sidebar.expander("📋 Example Details", expanded=False):
                        st.write(f"**Category:** {selected_prompt['category']}")
                        st.write(f"**Description:** {selected_prompt['description']}")
                        if selected_prompt['tags']:
                            st.write(f"**Tags:** {selected_prompt['tags']}")
                        st.write(f"**Updated:** {selected_prompt['updated_at']}")
        else:
            st.sidebar.info("No test prompt examples available")
