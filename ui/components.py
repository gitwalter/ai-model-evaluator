#!/usr/bin/env python3
"""
UI Components for AI Model Evaluator.
Contains Streamlit UI components for displaying results and interactions.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import json


class UIComponents:
    """UI Components for the Streamlit app."""
    
    @staticmethod
    def display_single_response(result: Dict[str, Any], evaluator) -> None:
        """Display a single model response with all its details."""
        # Response metadata at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", f"{result.get('response_time', 0):.2f}s")
        with col2:
            word_count = len(result["response"].split())
            st.metric("Word Count", word_count)
        with col3:
            char_count = len(result["response"])
            st.metric("Character Count", char_count)
        with col4:
            # Calculate response quality score
            analysis = evaluator.analyze_response(result["response"])
            overall_score = evaluator.calculate_overall_score(analysis)
            st.metric("Quality Score", f"{overall_score:.2f}")
        
        # Main formatted response display
        st.subheader("🤖 AI Response")
        
        # Create a styled container for the response with custom CSS
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        
        # Format and display the response
        formatted_response = UIComponents.format_response_for_display(result["response"])
        st.markdown(formatted_response)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a divider
        st.divider()
        
        # Response details in expandable section
        with st.expander("📋 Response Details & Context", expanded=False):
            # Show prompt used
            st.subheader("📝 Prompt Used")
            st.code(result["prompt"], language="text")
            
            if result.get("system_prompt"):
                st.subheader("⚙️ System Prompt")
                st.code(result["system_prompt"], language="text")
            
            # Raw response option
            st.subheader("📄 Raw Response")
            st.text_area(
                "Raw Response Text",
                value=result["response"],
                height=200,
                disabled=True,
                help="Raw text response without formatting"
            )
    
    @staticmethod
    def display_evaluation_dashboard(result: Dict[str, Any], analysis: Dict[str, Any], evaluator) -> None:
        """Display the evaluation dashboard for a single result."""
        # Create a comprehensive evaluation dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Quality Metrics")
            
            # Performance metrics
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric(
                    "Response Time", 
                    f"{result.get('response_time', 0):.2f}s",
                    help="Time taken to generate the response"
                )
            with perf_col2:
                st.metric(
                    "Word Count", 
                    analysis["response_length"]["details"],
                    help="Number of words in the response"
                )
            with perf_col3:
                response_length_score = analysis["response_length"]["score"]
                st.metric(
                    "Length Score", 
                    f"{response_length_score:.2f}",
                    help="Normalized length score (0-1)"
                )
            
            # Quality indicators
            st.subheader("✅ Quality Indicators")
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                code_score = analysis["code_quality"]["score"]
                st.metric(
                    "Code Quality", 
                    "✅ Present" if code_score > 0 else "❌ Missing",
                    help=analysis["code_quality"]["details"]
                )
                
                exec_score = analysis["execution_attempt"]["score"]
                st.metric(
                    "Execution Attempt", 
                    "✅ Present" if exec_score > 0 else "❌ Missing",
                    help=analysis["execution_attempt"]["details"]
                )
                
                error_score = analysis["error_handling"]["score"]
                st.metric(
                    "Error Handling", 
                    "✅ Present" if error_score > 0 else "❌ Missing",
                    help=analysis["error_handling"]["details"]
                )
            
            with quality_col2:
                summary_score = analysis["content_summary"]["score"]
                st.metric(
                    "Content Summary", 
                    "✅ Present" if summary_score > 0 else "❌ Missing",
                    help=analysis["content_summary"]["details"]
                )
                
                doc_score = analysis["documentation"]["score"]
                st.metric(
                    "Documentation", 
                    "✅ Present" if doc_score > 0 else "❌ Missing",
                    help=analysis["documentation"]["details"]
                )
        
        with col2:
            st.subheader("📈 Overall Score")
            
            # Calculate overall score
            overall_score = evaluator.calculate_overall_score(analysis)
            
            # Display overall score with visual indicator
            quality_indicator = evaluator.get_quality_indicator(overall_score)
            
            st.metric(
                "Overall Quality",
                f"{quality_indicator['color']} {quality_indicator['text']}",
                f"{overall_score:.2f}/1.0"
            )
            
            # Progress bar for overall score
            st.progress(overall_score)
            
            # Score breakdown
            st.subheader("📊 Score Breakdown")
            breakdown = evaluator.get_score_breakdown(analysis)
            for item in breakdown:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{item['criterion']}**")
                with col2:
                    st.write(f"{item['score']:.2f}")
                
                st.progress(item['score'])
                st.caption(item['details'])
        
        # Detailed analysis in expandable section
        with st.expander("🔍 Detailed Analysis", expanded=False):
            st.subheader("Detailed Criterion Analysis")
            
            for criterion, data in analysis.items():
                score = data.get("score", 0)
                details = data.get("details", "")
                status = "✅" if score > 0 else "❌"
                criterion_name = criterion.replace('_', ' ').title()
                
                st.write(f"### {criterion_name}")
                st.write(f"**Status:** {status}")
                st.write(f"**Score:** {score:.2f}")
                st.write(f"**Details:** {details}")
                st.divider()
    
    @staticmethod
    def display_code_execution(response_text: str, code_executor, unique_id: str = None) -> None:
        """Display enhanced code execution interface with editing and debugging capabilities."""
        st.subheader("🚀 Code Execution & Debugging")
        
        # Extract code blocks from response
        code_blocks = code_executor.extract_code_from_response(response_text)
        
        if not code_blocks:
            st.info("No executable code found in the response.")
            return
        
        # Take only the first code block for execution
        if code_blocks:
            block = code_blocks[0]  # Use only the first block
            language = block['language']
            
            # Create tabs for different execution modes
            exec_tab, edit_tab, debug_tab = st.tabs(["▶️ Execute", "✏️ Edit & Execute", "🐛 Debug"])
            
            with exec_tab:
                UIComponents._display_simple_execution(block, code_executor, unique_id)
            
            with edit_tab:
                UIComponents._display_editable_execution(block, code_executor, unique_id)
            
            with debug_tab:
                UIComponents._display_debug_execution(block, code_executor, unique_id)
            
            # Show additional blocks info if there are more
            if len(code_blocks) > 1:
                st.info(f"Note: {len(code_blocks)} code blocks found. Only the first block is shown for execution.")
        
        # Add a divider at the end
        st.divider()
    
    @staticmethod
    def _display_simple_execution(block: dict, code_executor, unique_id: str = None) -> None:
        """Display simple code execution (original functionality)."""
        language = block['language']
        
        st.write(f"**📝 Original Code ({language.upper()}):**")
        st.code(block['code'], language=block['language'])
        
        # Simple execution button with unique key
        unique_suffix = f"_{unique_id}" if unique_id else ""
        button_key = f"exec_simple_{language}_{hash(block['code'])}{unique_suffix}"
        
        if st.button(f"▶️ Execute {language.upper()} Code", key=button_key, use_container_width=True):
            if block['language'].lower() in ['python', 'py']:
                with st.spinner("Executing Python code..."):
                    execution_result = code_executor.execute_python_code(block['code'])
                    UIComponents._display_execution_results(execution_result, unique_id)
            else:
                UIComponents._display_unsupported_language(block['language'])
    
    @staticmethod
    def _display_editable_execution(block: dict, code_executor, unique_id: str = None) -> None:
        """Display editable code execution interface."""
        language = block['language']
        
        st.write(f"**✏️ Edit & Execute Code ({language.upper()}):**")
        
        # Create an editable text area for the code
        edited_code = st.text_area(
            "Edit the code below:",
            value=block['code'],
            height=400,
            help="Modify the code before execution"
        )
        
        # Execution options with unique keys
        unique_suffix = f"_{unique_id}" if unique_id else ""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            execute_edited = st.button("▶️ Execute Edited Code", key=f"exec_edited_{language}{unique_suffix}", use_container_width=True)
        
        with col2:
            reset_code = st.button("🔄 Reset to Original", key=f"reset_{language}{unique_suffix}", use_container_width=True)
        
        with col3:
            format_code = st.button("🎨 Format Code", key=f"format_{language}{unique_suffix}", use_container_width=True)
        
        # Handle button actions
        if reset_code:
            st.rerun()
        
        if format_code:
            if language.lower() in ['python', 'py']:
                try:
                    import black
                    formatted_code = black.format_str(edited_code, mode=black.FileMode())
                    st.success("Code formatted successfully!")
                    st.code(formatted_code, language=language)
                    edited_code = formatted_code
                except ImportError:
                    st.warning("Black formatter not available. Install with: pip install black")
                except Exception as e:
                    st.error(f"Formatting failed: {e}")
            else:
                st.info("Code formatting is currently only supported for Python.")
        
        if execute_edited:
            if block['language'].lower() in ['python', 'py']:
                with st.spinner("Executing edited Python code..."):
                    execution_result = code_executor.execute_python_code(edited_code)
                    UIComponents._display_execution_results(execution_result, unique_id)
            else:
                UIComponents._display_unsupported_language(block['language'])
    
    @staticmethod
    def _display_debug_execution(block: dict, code_executor, unique_id: str = None) -> None:
        """Display simplified debugging interface for code execution."""
        language = block['language']
        
        st.write(f"**🐛 Simple Debug ({language.upper()}):**")
        
        if language.lower() not in ['python', 'py']:
            st.warning("Debugging is currently only supported for Python code.")
            return
        
        # Simple debug options
        debug_code = st.text_area(
            "Code to debug:",
            value=block['code'],
            height=300,
            help="Modify the code for debugging"
        )
        
        # Debug actions with unique keys
        unique_suffix = f"_{unique_id}" if unique_id else ""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            syntax_check = st.button("🔍 Check Syntax", key=f"syntax_{language}{unique_suffix}", use_container_width=True)
        
        with col2:
            dry_run = st.button("🚀 Dry Run", key=f"dryrun_{language}{unique_suffix}", use_container_width=True)
        
        with col3:
            debug_exec = st.button("🐛 Debug Execute", key=f"debug_exec_{language}{unique_suffix}", use_container_width=True)
        
        with col4:
            inspect_vars = st.button("🔍 Inspect Variables", key=f"inspect_{language}{unique_suffix}", use_container_width=True)
        
        # Handle debug actions
        if syntax_check:
            UIComponents._simple_syntax_check(debug_code)
        
        if dry_run:
            UIComponents._simple_dry_run(debug_code)
        
        if debug_exec:
            UIComponents._simple_debug_execute(debug_code, code_executor, unique_id)
        
        if inspect_vars:
            UIComponents._debug_execute_with_variables(debug_code, code_executor, unique_id)
    
    @staticmethod
    def _debug_execute_with_variables(code: str, code_executor, unique_id: str = None) -> None:
        """Debug execution with variable inspection capabilities."""
        st.subheader("🔍 Debug Execution with Variable Inspection")
        
        with st.spinner("Executing code with variable capture..."):
            execution_result = code_executor.execute_python_code_with_variables(code)
        
        if execution_result['success']:
            st.success("✅ Code executed successfully!")
            
            # Display execution metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Execution Time", f"{execution_result['execution_time']:.3f}s")
            with col2:
                st.metric("Variables Captured", len(execution_result['variables']))
            
            # Display output
            if execution_result['output'].strip():
                # Remove variable capture markers from display
                clean_output = execution_result['output']
                start_marker = "=== VARIABLE_CAPTURE_START ==="
                end_marker = "=== VARIABLE_CAPTURE_END ==="
                
                start_idx = clean_output.find(start_marker)
                end_idx = clean_output.find(end_marker)
                
                if start_idx != -1 and end_idx != -1:
                    clean_output = clean_output[:start_idx] + clean_output[end_idx + len(end_marker):]
                
                if clean_output.strip():
                    st.write("**📤 Output:**")
                    st.text_area(
                        "Execution Output",
                        value=clean_output.strip(),
                        height=200,
                        disabled=True
                    )
            
            # Display variable inspection
            if execution_result['variables']:
                st.write("**🔍 Variable Inspection:**")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Details", "🔍 Raw Data"])
                
                with tab1:
                    UIComponents._display_variable_overview(execution_result)
                
                with tab2:
                    UIComponents._display_variable_details(execution_result)
                
                with tab3:
                    UIComponents._display_variable_raw_data(execution_result)
            else:
                st.info("ℹ️ No variables were captured during execution")
        else:
            st.error("❌ Code execution failed!")
            
            # Check if it's a missing library error
            if execution_result.get('missing_libraries'):
                UIComponents._display_library_management_info(execution_result)
            else:
                # Enhanced error display
                st.write("**🚨 Error Details:**")
                st.text_area(
                    "Error Output",
                    value=execution_result['error'],
                    height=200,
                    disabled=True
                )
                
                # Error analysis
                error_text = execution_result['error'].lower()
                if 'nameerror' in error_text:
                    st.info("💡 **Suggestion:** Check if all variables are defined before use")
                elif 'typeerror' in error_text:
                    st.info("💡 **Suggestion:** Check data types and function arguments")
                elif 'indexerror' in error_text:
                    st.info("💡 **Suggestion:** Check list/array indices")
                elif 'keyerror' in error_text:
                    st.info("💡 **Suggestion:** Check dictionary keys")
                elif 'attributeerror' in error_text:
                    st.info("💡 **Suggestion:** Check object attributes and methods")
                elif 'importerror' in error_text or 'modulenotfounderror' in error_text:
                    st.info("💡 **Suggestion:** Check if required modules are installed")
                else:
                    st.info("💡 **Suggestion:** Review the code logic and check for common programming mistakes")
    
    @staticmethod
    def _display_variable_overview(execution_result: dict) -> None:
        """Display an overview of captured variables."""
        variables = execution_result['variables']
        variable_types = execution_result['variable_types']
        variable_sizes = execution_result['variable_sizes']
        
        # Count variables by type
        type_counts = {}
        for var_type in variable_types.values():
            type_counts[var_type] = type_counts.get(var_type, 0) + 1
        
        # Display type distribution
        st.write("**📊 Variable Type Distribution:**")
        for var_type, count in type_counts.items():
            st.write(f"- {var_type}: {count} variables")
        
        # Display variables with their types and sizes
        st.write("**📋 Variable Summary:**")
        
        # Create a DataFrame-like display
        for var_name in sorted(variables.keys()):
            var_type = variable_types.get(var_name, 'Unknown')
            var_size = variable_sizes.get(var_name, 'N/A')
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{var_name}**")
            with col2:
                st.write(f"`{var_type}`")
            with col3:
                st.write(f"Size: {var_size}")
    
    @staticmethod
    def _display_variable_details(execution_result: dict) -> None:
        """Display detailed information about each variable."""
        variables = execution_result['variables']
        variable_types = execution_result['variable_types']
        variable_sizes = execution_result['variable_sizes']
        
        # Allow user to select a variable to inspect
        if variables:
            selected_var = st.selectbox(
                "Select variable to inspect:",
                options=sorted(variables.keys()),
                key="variable_inspector"
            )
            
            if selected_var:
                st.write(f"**🔍 Inspecting: `{selected_var}`**")
                
                # Variable type
                var_type = variable_types.get(selected_var, 'Unknown')
                st.write(f"**Type:** `{var_type}`")
                
                # Variable size
                var_size = variable_sizes.get(selected_var, 'N/A')
                st.write(f"**Size:** {var_size}")
                
                # Variable value
                var_value = variables[selected_var]
                st.write("**Value:**")
                
                # Display value based on type
                if var_type in ['str', 'string']:
                    st.text_area(
                        "String Value",
                        value=str(var_value),
                        height=150,
                        disabled=True
                    )
                elif var_type in ['list', 'tuple']:
                    st.write(f"```python\n{var_value}\n```")
                elif var_type in ['dict']:
                    st.write(f"```python\n{var_value}\n```")
                elif var_type in ['int', 'float']:
                    st.metric("Numeric Value", var_value)
                else:
                    # For other types, show as text
                    st.text_area(
                        "Object Value",
                        value=str(var_value),
                        height=200,
                        disabled=True
                    )
        else:
            st.info("No variables available for inspection")
    
    @staticmethod
    def _display_variable_raw_data(execution_result: dict) -> None:
        """Display raw variable data in JSON format."""
        st.write("**🔍 Raw Variable Data:**")
        
        # Create a clean data structure for display
        raw_data = {
            "variables": execution_result['variables'],
            "types": execution_result['variable_types'],
            "sizes": execution_result['variable_sizes']
        }
        
        st.json(raw_data)
    
    @staticmethod
    def _simple_syntax_check(code: str) -> None:
        """Simple syntax check for Python code."""
        st.subheader("🔍 Syntax Check")
        
        try:
            import ast
            ast.parse(code)
            st.success("✅ Syntax is valid!")
            
            # Show basic code structure
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
            
            st.write("**📊 Code Structure:**")
            if functions:
                st.write(f"- Functions: {', '.join(functions)}")
            if classes:
                st.write(f"- Classes: {', '.join(classes)}")
            if imports:
                st.write(f"- Imports: {', '.join(imports)}")
            
        except SyntaxError as e:
            st.error(f"❌ Syntax Error: {e}")
            st.write(f"**Line {e.lineno}:** {e.text}")
            st.write(f"**Error:** {e.msg}")
        except Exception as e:
            st.error(f"❌ Error during syntax check: {e}")
    
    @staticmethod
    def _simple_dry_run(code: str) -> None:
        """Simple dry run to identify potential issues."""
        st.subheader("🚀 Dry Run Analysis")
        
        # Check for common issues
        issues = []
        warnings = []
        
        # Check for undefined variables (simple heuristic)
        lines = code.split('\n')
        defined_vars = set()
        used_vars = set()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Simple variable definition detection
            if '=' in line and not line.startswith('if') and not line.startswith('elif'):
                var_name = line.split('=')[0].strip()
                if var_name and not var_name.startswith('#'):
                    defined_vars.add(var_name)
            
            # Simple variable usage detection
            for word in line.split():
                if word.isidentifier() and word not in ['if', 'else', 'elif', 'for', 'while', 'def', 'class', 'import', 'from', 'return', 'print']:
                    used_vars.add(word)
        
        undefined_vars = used_vars - defined_vars
        if undefined_vars:
            warnings.append(f"Potentially undefined variables: {', '.join(undefined_vars)}")
        
        # Check for common patterns
        if 'print(' in code:
            st.info("ℹ️ Code contains print statements")
        
        if 'import ' in code or 'from ' in code:
            st.info("ℹ️ Code contains imports")
        
        if 'def ' in code:
            st.info("ℹ️ Code contains function definitions")
        
        if 'class ' in code:
            st.info("ℹ️ Code contains class definitions")
        
        # Show results
        if warnings:
            st.warning("⚠️ Potential issues found:")
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.success("✅ No obvious issues detected")
    
    @staticmethod
    def _simple_debug_execute(code: str, code_executor, unique_id: str = None) -> None:
        """Simple debug execution with enhanced error reporting and library management."""
        st.subheader("🐛 Debug Execution")
        
        with st.spinner("Executing code in debug mode..."):
            execution_result = code_executor.execute_python_code(code)
        
        if execution_result['success']:
            st.success("✅ Code executed successfully!")
            
            # Enhanced output display
            if execution_result['output'].strip():
                st.write("**📤 Output:**")
                st.text_area(
                    "Execution Output",
                    value=execution_result['output'],
                    height=200,
                    disabled=True
                )
            else:
                st.info("ℹ️ Code executed but produced no output")
            
            # Show execution metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Execution Time", f"{execution_result['execution_time']:.3f}s")
            with col2:
                st.metric("Output Size", f"{len(execution_result['output'])} chars")
        else:
            st.error("❌ Code execution failed!")
            
            # Check if it's a missing library error
            if execution_result.get('missing_libraries'):
                UIComponents._display_library_management_info(execution_result)
            else:
                # Enhanced error display
                st.write("**🚨 Error Details:**")
                st.text_area(
                    "Error Output",
                    value=execution_result['error'],
                    height=200,
                    disabled=True
                )
                
                # Error analysis
                error_text = execution_result['error'].lower()
                if 'nameerror' in error_text:
                    st.info("💡 **Suggestion:** Check if all variables are defined before use")
                elif 'typeerror' in error_text:
                    st.info("💡 **Suggestion:** Check data types and function arguments")
                elif 'indexerror' in error_text:
                    st.info("💡 **Suggestion:** Check list/array indices")
                elif 'keyerror' in error_text:
                    st.info("💡 **Suggestion:** Check dictionary keys")
                elif 'attributeerror' in error_text:
                    st.info("💡 **Suggestion:** Check object attributes and methods")
                elif 'importerror' in error_text or 'modulenotfounderror' in error_text:
                    st.info("💡 **Suggestion:** Check if required modules are installed")
                else:
                    st.info("💡 **Suggestion:** Review the code logic and check for common programming mistakes")
    
    @staticmethod
    def _display_execution_results(execution_result: dict, unique_id: str = None) -> None:
        """Display execution results in a standardized format with library management."""
        st.write("**📋 Execution Results:**")
        
        if execution_result['success']:
            st.success("✅ Code executed successfully!")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", "Success")
                st.metric("Execution Time", f"{execution_result['execution_time']:.3f}s")
            with col2:
                st.metric("Return Code", "0")
                st.metric("Output Size", f"{len(execution_result['output'])} chars")
            
            # Output
            st.write("**📤 Output:**")
            if execution_result['output'].strip():
                st.text_area(
                    "Execution Output",
                    value=execution_result['output'],
                    height=300,
                    disabled=True,
                    help="Output from code execution"
                )
            else:
                st.info("No output generated (code executed silently)")
        else:
            st.error("❌ Code execution failed!")
            
            # Check if it's a missing library error
            if execution_result.get('missing_libraries'):
                UIComponents._display_library_management_info(execution_result, unique_id)
            else:
                # Regular error display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", "Failed")
                    st.metric("Execution Time", f"{execution_result['execution_time']:.3f}s")
                with col2:
                    st.metric("Error Type", "Runtime Error")
                    st.metric("Error Size", f"{len(execution_result['error'])} chars")
                
                # Error details
                st.write("**🚨 Error Details:**")
                st.text_area(
                    "Error Output",
                    value=execution_result['error'],
                    height=300,
                    disabled=True,
                    help="Error details from code execution"
                )
    
    @staticmethod
    def _display_library_management_info(execution_result: dict, unique_id: str = None) -> None:
        """Display library management information for missing libraries."""
        missing_libs = execution_result.get('missing_libraries', [])
        suggestions = execution_result.get('library_suggestions', [])
        alternatives = execution_result.get('alternative_solutions', [])
        
        # Library management section
        st.subheader("📦 Library Management")
        
        # Missing libraries
        st.error(f"**Missing Libraries:** {', '.join(missing_libs)}")
        
        # Installation suggestions
        if suggestions:
            st.write("**📥 Installation Commands:**")
            for i, suggestion in enumerate(suggestions, 1):
                st.code(suggestion, language="bash")
                
                # Add copy button functionality with unique key
                unique_suffix = f"_{unique_id}" if unique_id else ""
                if st.button(f"📋 Copy Command {i}", key=f"copy_cmd_{i}{unique_suffix}"):
                    st.write("✅ Command copied to clipboard!")
        
        # Alternative solutions
        if alternatives:
            st.write("**🔄 Alternative Solutions:**")
            for alternative in alternatives[:5]:  # Limit to 5 alternatives
                st.info(f"• {alternative}")
        
        # Quick actions with unique keys
        st.write("**⚡ Quick Actions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📦 Install All", key=f"install_all{unique_suffix}"):
                st.info("💡 Run these commands in your terminal:")
                for suggestion in suggestions:
                    st.code(suggestion, language="bash")
        
        with col2:
            if st.button("🔄 Show Alternatives", key=f"show_alternatives{unique_suffix}"):
                st.write("**Built-in Python Alternatives:**")
                for alternative in alternatives:
                    st.write(f"• {alternative}")
        
        with col3:
            if st.button("📚 Library Info", key=f"library_info{unique_suffix}"):
                st.write("**Library Information:**")
                for lib in missing_libs:
                    clean_lib = lib.split('.')[0]
                    st.write(f"• **{clean_lib}**: Popular Python library for data processing")
        
        # Error details (if any additional error info)
        if execution_result.get('error'):
            st.write("**🚨 Additional Error Details:**")
            st.text_area(
                "Error Output",
                value=execution_result['error'],
                height=200,
                disabled=True,
                help="Additional error details"
            )
    
    @staticmethod
    def _display_unsupported_language(language: str) -> None:
        """Display message for unsupported languages."""
        if language.lower() in ['bash', 'shell', 'sh']:
            st.warning("Shell command execution is not yet implemented for security reasons.")
            st.info("Shell execution requires additional security considerations and is not currently supported.")
        elif language.lower() in ['javascript', 'js', 'node']:
            st.warning("JavaScript execution is not yet implemented.")
            st.info("JavaScript execution requires a Node.js runtime and is not currently supported.")
        else:
            st.warning(f"Code execution is currently only supported for Python code. Found: {language}")
            st.info("Only Python code can be executed in this environment.")
    
    @staticmethod
    def format_response_for_display(response_text: str) -> str:
        """Format the response text for better display in the UI."""
        # Detect and format code blocks
        lines = response_text.split('\n')
        formatted_lines = []
        in_code_block = False
        code_block_lang = ""
        
        for line in lines:
            # Check for code block markers
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    # Extract language if specified
                    lang = line.strip()[3:].strip()
                    if lang:
                        code_block_lang = lang
                    formatted_lines.append(line)
                else:
                    # Ending a code block
                    in_code_block = False
                    code_block_lang = ""
                    formatted_lines.append(line)
            elif in_code_block:
                # Inside a code block - preserve formatting
                formatted_lines.append(line)
            else:
                # Regular text - apply some basic formatting
                # Convert single backticks to inline code
                if '`' in line and not line.strip().startswith('#'):
                    # Simple inline code detection
                    parts = line.split('`')
                    formatted_parts = []
                    for i, part in enumerate(parts):
                        if i % 2 == 1:  # Odd indices are code
                            formatted_parts.append(f'`{part}`')
                        else:
                            formatted_parts.append(part)
                    line = ''.join(formatted_parts)
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def display_comparison_table(results: List[Dict[str, Any]], evaluator) -> None:
        """Display a comparison table for multiple model results."""
        if not results:
            st.info("No results to compare.")
            return
        
        # Create comparison data
        comparison_data = []
        for result in results:
            if result["status"] == "success":
                analysis = evaluator.analyze_response(result["response"])
                overall_score = evaluator.calculate_overall_score(analysis)
                
                comparison_data.append({
                    "Model": result["model_name"],
                    "Response Time": f"{result.get('response_time', 0):.2f}s",
                    "Word Count": len(result["response"].split()),
                    "Overall Score": f"{overall_score:.2f}",
                    "Code Quality": "✅" if analysis["code_quality"]["score"] > 0 else "❌",
                    "Execution": "✅" if analysis["execution_attempt"]["score"] > 0 else "❌",
                    "Summary": "✅" if analysis["content_summary"]["score"] > 0 else "❌",
                    "Error Handling": "✅" if analysis["error_handling"]["score"] > 0 else "❌",
                    "Documentation": "✅" if analysis["documentation"]["score"] > 0 else "❌"
                })
        
        if comparison_data:
            st.dataframe(comparison_data, use_container_width=True)
        else:
            st.error("No successful results to compare")
    
    @staticmethod
    def display_export_section(results: List[Dict[str, Any]]) -> None:
        """Display export section for results."""
        if not results:
            return
        
        st.sidebar.header("Export")
        
        if st.sidebar.button("📁 Export Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export JSON
            json_data = {
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            st.sidebar.download_button(
                label="Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"gemini_test_results_{timestamp}.json",
                mime="application/json"
            )
            
            # Export Markdown report
            report_lines = [
                "# Gemini Model Test Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Test Results",
                ""
            ]
            
            for i, result in enumerate(results, 1):
                report_lines.extend([
                    f"### Test {i}: {result['model_name']}",
                    f"**Status:** {result['status']}",
                    f"**Response Time:** {result.get('response_time', 0):.2f}s",
                    "",
                    "**Prompt:**",
                    f"```",
                    result['prompt'],
                    "```",
                    ""
                ])
                
                if result.get('system_prompt'):
                    report_lines.extend([
                        "**System Prompt:**",
                        f"```",
                        result['system_prompt'],
                        "```",
                        ""
                    ])
                
                report_lines.extend([
                    "**Response:**",
                    f"```",
                    result['response'],
                    "```",
                    "",
                    "---",
                    ""
                ])
            
            report_content = "\n".join(report_lines)
            
            st.sidebar.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"gemini_test_report_{timestamp}.md",
                mime="text/markdown"
            )