#!/usr/bin/env python3
"""
Code Executor for AI Model Evaluator.
Handles Python code execution and analysis.
"""

import streamlit as st
import time
import subprocess
import sys
import tempfile
import traceback
from io import StringIO
import contextlib
import ast
import re
import json
from typing import Dict, Any, List


class CodeExecutor:
    """Handles Python code execution and analysis."""
    
    def __init__(self, timeout: int = 30):
        """Initialize the CodeExecutor with timeout settings."""
        self.timeout = timeout
    
    def extract_code_from_response(self, response_text: str) -> List[Dict[str, str]]:
        """Extract code blocks from AI response with support for executable commands."""
        code_blocks = []
        
        # Look for markdown code blocks
        code_pattern = r'```(?:(\w+)\n)?(.*?)```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        
        for lang, code in matches:
            if code.strip():
                # Split code into executable blocks based on command patterns
                executable_blocks = self.split_code_into_executable_blocks(code.strip(), lang or 'python')
                for i, block in enumerate(executable_blocks):
                    code_blocks.append({
                        'language': lang or 'python',
                        'code': block,
                        'block_index': i,
                        'total_blocks': len(executable_blocks)
                    })
        
        # Also look for inline code that might be Python functions
        python_patterns = [
            r'def\s+\w+\s*\([^)]*\)\s*:.*?(?=\n\S|\Z)',
            r'import\s+.*?(?=\n\S|\Z)',
            r'class\s+\w+.*?(?=\n\S|\Z)'
        ]
        
        for pattern in python_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                if match.strip() and len(match.strip()) > 10:
                    code_blocks.append({
                        'language': 'python',
                        'code': match.strip(),
                        'block_index': 0,
                        'total_blocks': 1
                    })
        
        return code_blocks
    
    def split_code_into_executable_blocks(self, code: str, language: str) -> List[str]:
        """Split code into executable blocks based on language-specific patterns."""
        if language.lower() in ['python', 'py']:
            return self.split_python_code_blocks(code)
        elif language.lower() in ['bash', 'shell', 'sh']:
            return self.split_shell_code_blocks(code)
        elif language.lower() in ['javascript', 'js', 'node']:
            return self.split_javascript_code_blocks(code)
        else:
            # For other languages, return the entire code as one block
            return [code]
    
    def split_python_code_blocks(self, code: str) -> List[str]:
        """Return the entire Python code as a single executable block."""
        def is_syntactically_valid(code_block: str) -> bool:
            """Check if a code block is syntactically valid Python."""
            try:
                ast.parse(code_block)
                return True
            except SyntaxError:
                return False
        
        # Clean up the code
        lines = [line.rstrip() for line in code.split('\n')]
        cleaned_code = '\n'.join(lines)
        
        # Check if the code is syntactically valid
        if is_syntactically_valid(cleaned_code):
            return [cleaned_code]
        else:
            # If not valid, return the original code anyway (user can fix)
            return [code]
    
    def split_shell_code_blocks(self, code: str) -> List[str]:
        """Return the entire shell code as a single executable block."""
        # Clean up the code
        lines = [line.rstrip() for line in code.split('\n')]
        cleaned_code = '\n'.join(lines)
        
        return [cleaned_code]
    
    def split_javascript_code_blocks(self, code: str) -> List[str]:
        """Return the entire JavaScript code as a single executable block."""
        # Clean up the code
        lines = [line.rstrip() for line in code.split('\n')]
        cleaned_code = '\n'.join(lines)
        
        return [cleaned_code]
    
    def execute_python_code_with_variables(self, code: str) -> Dict[str, Any]:
        """Execute Python code and capture variables for inspection."""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'variables': {},
            'variable_types': {},
            'variable_sizes': {},
            'missing_libraries': [],
            'library_suggestions': [],
            'alternative_solutions': []
        }
        
        start_time = time.time()
        
        # First, analyze the code for imports
        import_analysis = self._analyze_imports(code)
        result['missing_libraries'] = import_analysis['missing']
        result['library_suggestions'] = import_analysis['suggestions']
        result['alternative_solutions'] = import_analysis['alternatives']
        
        # If there are missing libraries, provide helpful information
        if import_analysis['missing']:
            result['error'] = self._format_missing_library_error(import_analysis)
            result['execution_time'] = time.time() - start_time
            return result
        
        try:
            # Create a modified version of the code that captures variables
            modified_code = self._add_variable_capture(code)
            
            # Create a temporary file for the modified code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(modified_code)
                temp_file_path = temp_file.name
            
            # Capture stdout and stderr
            output_buffer = StringIO()
            error_buffer = StringIO()
            
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                # Execute the code with timeout
                process = subprocess.run(
                    [sys.executable, temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
            
            result['execution_time'] = time.time() - start_time
            
            if process.returncode == 0:
                result['success'] = True
                result['output'] = output_buffer.getvalue() + process.stdout
                
                # Extract variables from the output
                variables_data = self._extract_variables_from_output(result['output'])
                result['variables'] = variables_data.get('variables', {})
                result['variable_types'] = variables_data.get('types', {})
                result['variable_sizes'] = variables_data.get('sizes', {})
            else:
                error_output = error_buffer.getvalue() + process.stderr
                result['error'] = error_output
                
                # Check if the error is due to missing libraries
                missing_libs = self._extract_missing_libraries_from_error(error_output)
                if missing_libs:
                    result['missing_libraries'] = missing_libs
                    result['library_suggestions'] = self._get_library_suggestions(missing_libs)
                    result['alternative_solutions'] = self._get_alternative_solutions(missing_libs)
        
        except subprocess.TimeoutExpired:
            result['error'] = f"Code execution timed out after {self.timeout} seconds"
            result['execution_time'] = self.timeout
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            result['execution_time'] = time.time() - start_time
        
        return result
    
    def _add_variable_capture(self, code: str) -> str:
        """Add variable capture code to the original code."""
        capture_code = '''
import json
import sys
import traceback
from io import StringIO

# Capture original stdout
original_stdout = sys.stdout
captured_output = StringIO()
sys.stdout = captured_output

try:
    # Execute the user code
    exec(compile("""{user_code}""", '<string>', 'exec'), globals(), locals())
    
    # Capture variables
    variables = {{}}
    variable_types = {{}}
    variable_sizes = {{}}
    
    # Get all variables in locals (excluding built-ins and special variables)
    # Create a copy of locals to avoid modification during iteration
    local_vars = dict(locals())
    excluded_vars = {{'captured_output', 'original_stdout', 'variables', 'variable_types', 'variable_sizes', 'local_vars', 'excluded_vars', 'StringIO'}}
    
    for var_name, var_value in local_vars.items():
        # Skip built-ins, special variables, and imported modules
        if (not var_name.startswith('_') and 
            var_name not in excluded_vars and
            not hasattr(var_value, '__file__') and  # Skip imported modules
            not str(type(var_value)).startswith("<class 'module'>")):
            try:
                # Store variable value (convert to string representation for complex objects)
                if hasattr(var_value, '__dict__'):
                    # For objects, try to get a string representation
                    variables[var_name] = str(var_value)
                else:
                    variables[var_name] = var_value
                
                # Store variable type
                variable_types[var_name] = str(type(var_value).__name__)
                
                # Store variable size information
                if hasattr(var_value, '__len__'):
                    try:
                        variable_sizes[var_name] = len(var_value)
                    except:
                        variable_sizes[var_name] = 'N/A'
                elif hasattr(var_value, 'shape'):  # For numpy arrays
                    variable_sizes[var_name] = str(var_value.shape)
                else:
                    variable_sizes[var_name] = 'N/A'
                    
            except Exception as e:
                variables[var_name] = f"<Error capturing: {{str(e)}}>"
                variable_types[var_name] = "Error"
                variable_sizes[var_name] = "N/A"
    
    # Output the captured variables as JSON
    print("\\n=== VARIABLE_CAPTURE_START ===")
    print(json.dumps({{
        "variables": variables,
        "types": variable_types,
        "sizes": variable_sizes
    }}, default=str))
    print("=== VARIABLE_CAPTURE_END ===")
    
except Exception as e:
    print(f"Error during execution: {{str(e)}}")
    traceback.print_exc()

finally:
    # Restore original stdout and print captured output
    sys.stdout = original_stdout
    print(captured_output.getvalue())
'''
        
        # Escape the user code for the exec statement
        escaped_code = code.replace('"', '\\"').replace('\n', '\\n')
        return capture_code.format(user_code=escaped_code)
    
    def _extract_variables_from_output(self, output: str) -> Dict[str, Any]:
        """Extract variable data from the execution output."""
        try:
            # Find the variable capture section
            start_marker = "=== VARIABLE_CAPTURE_START ==="
            end_marker = "=== VARIABLE_CAPTURE_END ==="
            
            start_idx = output.find(start_marker)
            end_idx = output.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                # Extract the JSON data
                json_start = start_idx + len(start_marker)
                json_data = output[json_start:end_idx].strip()
                
                # Parse the JSON
                variables_data = json.loads(json_data)
                return variables_data
            else:
                return {"variables": {}, "types": {}, "sizes": {}}
        except Exception as e:
            return {"variables": {}, "types": {}, "sizes": {}}
    
    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely and return results with enhanced library management."""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'missing_libraries': [],
            'library_suggestions': [],
            'alternative_solutions': []
        }
        
        start_time = time.time()
        
        # First, analyze the code for imports
        import_analysis = self._analyze_imports(code)
        result['missing_libraries'] = import_analysis['missing']
        result['library_suggestions'] = import_analysis['suggestions']
        result['alternative_solutions'] = import_analysis['alternatives']
        
        # If there are missing libraries, provide helpful information
        if import_analysis['missing']:
            result['error'] = self._format_missing_library_error(import_analysis)
            result['execution_time'] = time.time() - start_time
            return result
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Capture stdout and stderr
            output_buffer = StringIO()
            error_buffer = StringIO()
            
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                # Execute the code with timeout
                process = subprocess.run(
                    [sys.executable, temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
            
            result['execution_time'] = time.time() - start_time
            
            if process.returncode == 0:
                result['success'] = True
                result['output'] = output_buffer.getvalue() + process.stdout
            else:
                error_output = error_buffer.getvalue() + process.stderr
                result['error'] = error_output
                
                # Check if the error is due to missing libraries
                missing_libs = self._extract_missing_libraries_from_error(error_output)
                if missing_libs:
                    result['missing_libraries'] = missing_libs
                    result['library_suggestions'] = self._get_library_suggestions(missing_libs)
                    result['alternative_solutions'] = self._get_alternative_solutions(missing_libs)
        
        except subprocess.TimeoutExpired:
            result['error'] = f"Code execution timed out after {self.timeout} seconds"
            result['execution_time'] = self.timeout
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            result['execution_time'] = time.time() - start_time
        
        return result
    
    def _analyze_imports(self, code: str) -> Dict[str, Any]:
        """Analyze imports in the code and check for missing libraries."""
        try:
            tree = ast.parse(code)
            imports = []
            missing = []
            suggestions = []
            alternatives = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        if module:
                            imports.append(f"{module}.{alias.name}")
                        else:
                            imports.append(alias.name)
            
            # Check which imports are missing
            for imp in imports:
                if not self._is_library_available(imp):
                    missing.append(imp)
                    suggestions.extend(self._get_library_suggestions([imp]))
                    alternatives.extend(self._get_alternative_solutions([imp]))
            
            return {
                'imports': imports,
                'missing': missing,
                'suggestions': suggestions,
                'alternatives': alternatives
            }
        except SyntaxError:
            return {
                'imports': [],
                'missing': [],
                'suggestions': [],
                'alternatives': []
            }
    
    def _is_library_available(self, library_name: str) -> bool:
        """Check if a library is available in the current environment."""
        try:
            # Handle different import patterns
            if '.' in library_name:
                # For imports like 'pandas.DataFrame'
                base_module = library_name.split('.')[0]
                __import__(base_module)
            else:
                __import__(library_name)
            return True
        except ImportError:
            return False
    
    def _extract_missing_libraries_from_error(self, error_text: str) -> List[str]:
        """Extract missing library names from error messages."""
        missing_libs = []
        
        # Common patterns for missing library errors
        patterns = [
            r"No module named '([^']+)'",
            r"ModuleNotFoundError: No module named '([^']+)'",
            r"ImportError: No module named '([^']+)'",
            r"cannot import name '([^']+)' from '([^']+)'"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, error_text)
            for match in matches:
                if isinstance(match, tuple):
                    missing_libs.extend(match)
                else:
                    missing_libs.append(match)
        
        return list(set(missing_libs))  # Remove duplicates
    
    def _get_library_suggestions(self, missing_libraries: List[str]) -> List[str]:
        """Get installation suggestions for missing libraries."""
        suggestions = []
        
        # Common library mappings
        library_mappings = {
            'pandas': 'pip install pandas',
            'numpy': 'pip install numpy',
            'matplotlib': 'pip install matplotlib',
            'seaborn': 'pip install seaborn',
            'scikit-learn': 'pip install scikit-learn',
            'tensorflow': 'pip install tensorflow',
            'torch': 'pip install torch',
            'requests': 'pip install requests',
            'beautifulsoup4': 'pip install beautifulsoup4',
            'selenium': 'pip install selenium',
            'flask': 'pip install flask',
            'django': 'pip install django',
            'fastapi': 'pip install fastapi',
            'streamlit': 'pip install streamlit',
            'plotly': 'pip install plotly',
            'bokeh': 'pip install bokeh',
            'openpyxl': 'pip install openpyxl',
            'xlrd': 'pip install xlrd',
            'pillow': 'pip install pillow',
            'opencv-python': 'pip install opencv-python',
            'scipy': 'pip install scipy',
            'statsmodels': 'pip install statsmodels',
            'nltk': 'pip install nltk',
            'spacy': 'pip install spacy',
            'transformers': 'pip install transformers',
            'datasets': 'pip install datasets',
            'huggingface_hub': 'pip install huggingface_hub',
            'langchain': 'pip install langchain',
            'llama_index': 'pip install llama-index',
            'chromadb': 'pip install chromadb',
            'pinecone': 'pip install pinecone-client',
            'weaviate': 'pip install weaviate-client',
            'qdrant': 'pip install qdrant-client',
            'milvus': 'pip install pymilvus',
            'faiss': 'pip install faiss-cpu',
            'annoy': 'pip install annoy',
            'hnswlib': 'pip install hnswlib'
        }
        
        for lib in missing_libraries:
            # Clean up library name
            clean_lib = lib.split('.')[0]  # Remove submodules
            if clean_lib in library_mappings:
                suggestions.append(library_mappings[clean_lib])
            else:
                suggestions.append(f"pip install {clean_lib}")
        
        return suggestions
    
    def _get_alternative_solutions(self, missing_libraries: List[str]) -> List[str]:
        """Get alternative solutions for missing libraries."""
        alternatives = []
        
        # Alternative library mappings
        alternatives_mappings = {
            'pandas': [
                "Use built-in CSV module: import csv",
                "Use built-in JSON module: import json",
                "Use built-in sqlite3 for database operations"
            ],
            'numpy': [
                "Use built-in math module: import math",
                "Use built-in statistics module: import statistics",
                "Use built-in random module: import random"
            ],
            'matplotlib': [
                "Use built-in print statements for simple output",
                "Use ASCII art for simple visualizations",
                "Use tabular output with string formatting"
            ],
            'requests': [
                "Use built-in urllib: import urllib.request",
                "Use built-in http.client: import http.client"
            ],
            'beautifulsoup4': [
                "Use built-in re module for regex parsing",
                "Use built-in html.parser: import html.parser"
            ],
            'selenium': [
                "Use built-in urllib for web scraping",
                "Use built-in http.client for API calls"
            ],
            'flask': [
                "Use built-in http.server: python -m http.server",
                "Use built-in sockets for simple server"
            ],
            'streamlit': [
                "Use built-in tkinter for GUI",
                "Use command-line interface",
                "Use built-in http.server for web interface"
            ]
        }
        
        for lib in missing_libraries:
            clean_lib = lib.split('.')[0]
            if clean_lib in alternatives_mappings:
                alternatives.extend(alternatives_mappings[clean_lib])
            else:
                alternatives.append(f"Consider using built-in Python modules instead of {clean_lib}")
        
        return alternatives
    
    def _format_missing_library_error(self, import_analysis: Dict[str, Any]) -> str:
        """Format a helpful error message for missing libraries."""
        missing = import_analysis['missing']
        suggestions = import_analysis['suggestions']
        alternatives = import_analysis['alternatives']
        
        error_msg = f"❌ Missing Libraries Detected: {', '.join(missing)}\n\n"
        
        if suggestions:
            error_msg += "📦 Installation Commands:\n"
            for suggestion in suggestions:
                error_msg += f"   {suggestion}\n"
            error_msg += "\n"
        
        if alternatives:
            error_msg += "🔄 Alternative Solutions:\n"
            for alternative in alternatives[:5]:  # Limit to 5 alternatives
                error_msg += f"   • {alternative}\n"
            error_msg += "\n"
        
        error_msg += "💡 Tip: You can install libraries using pip or conda, or modify the code to use built-in Python modules."
        
        return error_msg
    
    def analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure and provide insights."""
        analysis = {
            'lines': len(code.split('\n')),
            'characters': len(code),
            'code_lines': 0,
            'comment_lines': 0,
            'language_specific': {}
        }
        
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                analysis['code_lines'] += 1
            elif stripped.startswith('#'):
                analysis['comment_lines'] += 1
        
        # Language-specific analysis
        if language.lower() in ['python', 'py']:
            analysis['language_specific'] = self._analyze_python_structure(code)
        elif language.lower() in ['bash', 'shell', 'sh']:
            analysis['language_specific'] = self._analyze_shell_structure(code)
        elif language.lower() in ['javascript', 'js', 'node']:
            analysis['language_specific'] = self._analyze_javascript_structure(code)
        
        return analysis
    
    def _analyze_python_structure(self, code: str) -> Dict[str, Any]:
        """Analyze Python code structure."""
        try:
            tree = ast.parse(code)
            analysis = {
                'functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                'classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                'imports': len([node for node in ast.walk(tree) if isinstance(node, ast.Import)]),
                'import_froms': len([node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]),
                'loops': len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]),
                'conditionals': len([node for node in ast.walk(tree) if isinstance(node, ast.If)]),
                'try_blocks': len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
            }
            return analysis
        except SyntaxError:
            return {'error': 'Invalid Python syntax'}
    
    def _analyze_shell_structure(self, code: str) -> Dict[str, Any]:
        """Analyze shell code structure."""
        lines = code.split('\n')
        analysis = {
            'commands': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comments': len([line for line in lines if line.strip().startswith('#')]),
            'file_operations': len(re.findall(r'\b(cp|mv|rm|mkdir|touch|cat|grep|find)\b', code)),
            'control_structures': len(re.findall(r'\b(if|for|while|case)\b', code))
        }
        return analysis
    
    def _analyze_javascript_structure(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code structure."""
        analysis = {
            'functions': len(re.findall(r'\bfunction\s+\w+\s*\(', code)),
            'arrow_functions': len(re.findall(r'=>', code)),
            'async_functions': len(re.findall(r'\basync\s+function\b', code)),
            'await_keywords': len(re.findall(r'\bawait\b', code)),
            'const_declarations': len(re.findall(r'\bconst\b', code)),
            'let_declarations': len(re.findall(r'\blet\b', code)),
            'var_declarations': len(re.findall(r'\bvar\b', code))
        }
        return analysis
