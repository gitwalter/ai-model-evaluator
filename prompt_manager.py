#!/usr/bin/env python3
"""
Prompt Manager for AI Model Evaluator
Provides SQLite database functionality for managing system prompts and user prompts.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import streamlit as st

class PromptDatabase:
    """SQLite database manager for prompts."""
    
    def __init__(self, db_path: str = "prompts.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create system prompts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user prompts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def add_system_prompt(self, name: str, content: str, category: str, description: str = "") -> bool:
        """Add a new system prompt to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO system_prompts (name, content, category, description, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, content, category, description, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error adding system prompt: {str(e)}")
            return False
    
    def add_user_prompt(self, name: str, content: str, category: str, description: str = "", tags: str = "") -> bool:
        """Add a new user prompt to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_prompts (name, content, category, description, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, content, category, description, tags, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error adding user prompt: {str(e)}")
            return False
    
    def get_system_prompts(self, category: Optional[str] = None) -> List[Dict]:
        """Get system prompts from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if category:
                    cursor.execute("""
                        SELECT id, name, content, category, description, created_at, updated_at
                        FROM system_prompts WHERE category = ? ORDER BY name
                    """, (category,))
                else:
                    cursor.execute("""
                        SELECT id, name, content, category, description, created_at, updated_at
                        FROM system_prompts ORDER BY category, name
                    """)
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error getting system prompts: {str(e)}")
            return []
    
    def get_user_prompts(self, category: Optional[str] = None) -> List[Dict]:
        """Get user prompts from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if category:
                    cursor.execute("""
                        SELECT id, name, content, category, description, tags, created_at, updated_at
                        FROM user_prompts WHERE category = ? ORDER BY name
                    """, (category,))
                else:
                    cursor.execute("""
                        SELECT id, name, content, category, description, tags, created_at, updated_at
                        FROM user_prompts ORDER BY category, name
                    """)
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error getting user prompts: {str(e)}")
            return []
    
    def update_system_prompt(self, prompt_id: int, name: str, content: str, category: str, description: str = "") -> bool:
        """Update an existing system prompt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE system_prompts 
                    SET name = ?, content = ?, category = ?, description = ?, updated_at = ?
                    WHERE id = ?
                """, (name, content, category, description, datetime.now(), prompt_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            st.error(f"Error updating system prompt: {str(e)}")
            return False
    
    def update_user_prompt(self, prompt_id: int, name: str, content: str, category: str, description: str = "", tags: str = "") -> bool:
        """Update an existing user prompt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_prompts 
                    SET name = ?, content = ?, category = ?, description = ?, tags = ?, updated_at = ?
                    WHERE id = ?
                """, (name, content, category, description, tags, datetime.now(), prompt_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            st.error(f"Error updating user prompt: {str(e)}")
            return False
    
    def delete_system_prompt(self, prompt_id: int) -> bool:
        """Delete a system prompt from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM system_prompts WHERE id = ?", (prompt_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            st.error(f"Error deleting system prompt: {str(e)}")
            return False
    
    def delete_user_prompt(self, prompt_id: int) -> bool:
        """Delete a user prompt from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_prompts WHERE id = ?", (prompt_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            st.error(f"Error deleting user prompt: {str(e)}")
            return False
    
    def get_categories(self, prompt_type: str = "both") -> List[str]:
        """Get unique categories for prompts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                categories = []
                
                if prompt_type in ["both", "system"]:
                    cursor.execute("SELECT DISTINCT category FROM system_prompts ORDER BY category")
                    categories.extend([row[0] for row in cursor.fetchall()])
                
                if prompt_type in ["both", "user"]:
                    cursor.execute("SELECT DISTINCT category FROM user_prompts ORDER BY category")
                    categories.extend([row[0] for row in cursor.fetchall()])
                
                return sorted(list(set(categories)))
        except Exception as e:
            st.error(f"Error getting categories: {str(e)}")
            return []

class DefaultPrompts:
    """Default prompts for development tasks."""
    
    @staticmethod
    def get_default_system_prompts() -> Dict[str, Dict[str, str]]:
        """Get default system prompts for development tasks."""
        return {
            "Code Producer (Default)": {
                "content": """You are a code producer focused on delivering working solutions quickly. 
Provide clean, functional code with minimal explanation but good documentation.

REQUIREMENTS:
- Write complete, runnable code that works immediately
- Include necessary imports and dependencies
- Add docstrings and key comments
- Handle errors appropriately
- Follow PEP 8 style guidelines
- Provide brief usage examples

FOCUS:
- Production-ready code that works
- Clean, readable implementation
- Essential documentation only
- Quick, effective solutions
- Minimal but clear explanations

DELIVER:
- Complete, executable code
- Brief summary of what the code does
- Key usage instructions
- Any important notes or limitations

Keep responses concise and code-focused.""",
                "category": "Development",
                "description": "Fast code production with minimal explanation but good documentation"
            },
            
            "Python Development Expert": {
                "content": """You are a Python development expert. Provide practical, production-ready solutions 
for Python applications. Focus on modern Python best practices, clean code, and real-world implementation.

CRITICAL REQUIREMENT: You MUST always deliver complete, runnable code examples unless explicitly asked not to do so. 
Your code should be ready to execute immediately without any additional modifications or setup.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your response, including:
- Overview of the solution provided
- Key features and functionality
- Important implementation details
- Usage instructions and examples
- Any limitations or considerations

Key areas:
- Modern Python features (3.8+) and best practices
- Clean, readable, and maintainable code
- Error handling and edge cases
- Performance optimization and efficiency
- Testing strategies and debugging
- Package management and dependencies
- Documentation and code comments
- Security best practices
- Integration with popular libraries and frameworks
- Deployment and production considerations

Code delivery requirements:
- ALWAYS provide complete, executable Python code that can run immediately
- Include ALL necessary imports and dependencies
- Add proper error handling and validation
- Include example usage and test cases
- Ensure code follows PEP 8 style guidelines
- Add comprehensive comments and docstrings
- Make code production-ready with proper logging and configuration
- Never provide incomplete code snippets or pseudo-code
- Always test your code examples to ensure they work correctly
- ALWAYS provide a comprehensive content summary

Remember: Your primary goal is to deliver working, complete code that users can run immediately. 
Always provide complete, runnable code examples, explain your design decisions, and include a comprehensive content summary.""",
                "category": "Python Development",
                "description": "Practical Python development guidance for production applications with complete runnable code"
            },
            
            "Code Review Agent": {
                "content": """You are an expert code reviewer. Provide detailed feedback on code quality, 
best practices, security, and performance. Always suggest improvements and explain your reasoning.

CRITICAL REQUIREMENT: When suggesting code improvements, you MUST always provide complete, runnable code examples 
unless explicitly asked not to do so. Your suggestions should be ready to implement immediately without any additional modifications.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your review, including:
- Overview of the code analysis performed
- Key issues identified and their severity
- Summary of suggested improvements
- Implementation priority and impact
- Overall code quality assessment

Focus on:
- Code structure and organization
- Performance optimizations
- Security vulnerabilities
- Best practices and conventions
- Error handling and edge cases
- Documentation and readability

Code improvement requirements:
- ALWAYS provide complete, executable code examples for all suggestions
- Include ALL necessary imports and dependencies
- Demonstrate best practices with working code
- Show before/after comparisons with runnable examples
- Ensure suggested code follows proper standards
- Include error handling and edge case solutions
- Never provide incomplete code snippets or pseudo-code
- Always test your code examples to ensure they work correctly
- ALWAYS provide a comprehensive content summary

Remember: Your primary goal is to deliver working, complete code improvements that users can implement immediately.""",
                "category": "Development",
                "description": "Expert code reviewer for comprehensive code analysis with complete runnable examples"
            },
            
            "Debugging Assistant": {
                "content": """You are a debugging expert. Help identify and fix issues in code. 
Provide step-by-step solutions and explain the root causes. Include code examples when helpful.

CRITICAL REQUIREMENT: Always provide complete, runnable code examples for fixes and solutions 
unless explicitly asked not to do so. Your debugging solutions should be ready to implement immediately without any additional modifications.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your debugging analysis, including:
- Overview of the issue identified
- Root cause analysis summary
- Key debugging steps performed
- Summary of the solution provided
- Prevention strategies and best practices

Approach:
- Analyze error messages and stack traces
- Identify the root cause of issues
- Provide step-by-step debugging strategies
- Suggest fixes with explanations
- Include preventive measures

Debugging solution requirements:
- ALWAYS provide complete, executable code for all fixes
- Include ALL necessary imports and dependencies
- Show the original problematic code and the corrected version
- Demonstrate the fix with working examples
- Include proper error handling and validation
- Add debugging tools and logging examples
- Ensure solutions are production-ready
- Never provide incomplete code snippets or pseudo-code
- Always test your code examples to ensure they work correctly
- ALWAYS provide a comprehensive content summary

Remember: Your primary goal is to deliver working, complete debugging solutions that users can implement immediately.""",
                "category": "Development",
                "description": "Expert debugging assistance with step-by-step solutions and complete runnable code"
            },
            
            "Documentation Writer": {
                "content": """You are a technical writer. Create clear, comprehensive documentation 
for code, APIs, and systems. Include examples and best practices. Structure your response logically.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your documentation, including:
- Overview of the documentation provided
- Key sections and topics covered
- Important information and highlights
- Usage instructions and examples
- Any important notes or warnings

Documentation standards:
- Clear and concise explanations
- Code examples and usage patterns
- API documentation with parameters
- Best practices and guidelines
- Troubleshooting sections
- Version compatibility notes
- ALWAYS provide a comprehensive content summary""",
                "category": "Documentation",
                "description": "Technical writing for comprehensive documentation"
            },
            
            "Security Expert": {
                "content": """You are a cybersecurity expert. Analyze code and systems for security vulnerabilities. 
Provide detailed security recommendations and best practices. Always consider the security implications.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your security analysis, including:
- Overview of the security assessment performed
- Key vulnerabilities identified and their severity
- Security recommendations and priorities
- Implementation guidance and best practices
- Risk assessment and mitigation strategies

Security focus:
- Vulnerability assessment and analysis
- Secure coding practices
- Authentication and authorization
- Data protection and privacy
- Input validation and sanitization
- Security testing strategies
- ALWAYS provide a comprehensive content summary""",
                "category": "Security",
                "description": "Cybersecurity expert for security analysis and recommendations"
            },
            
            "Performance Optimizer": {
                "content": """You are a performance optimization expert. Analyze code for performance bottlenecks. 
Provide specific optimization strategies and measure the impact of your suggestions.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your performance analysis, including:
- Overview of the performance analysis performed
- Key bottlenecks identified and their impact
- Optimization strategies and their expected benefits
- Implementation priority and complexity
- Performance improvement metrics and benchmarks

Optimization areas:
- Algorithm efficiency and complexity
- Memory usage and management
- Database query optimization
- Caching strategies
- Profiling and benchmarking
- Scalability improvements
- ALWAYS provide a comprehensive content summary""",
                "category": "Performance",
                "description": "Performance optimization expert for code and system improvements"
            },
            
            "Python Expert": {
                "content": """You are a Python development expert. Provide guidance on Python best practices, 
libraries, frameworks, and advanced concepts. Focus on modern Python development.

CRITICAL REQUIREMENT: You MUST always provide complete, runnable code examples for all Python solutions 
unless explicitly asked not to do so. Your code should be ready to execute immediately without any additional modifications.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your response, including:
- Overview of the Python solution provided
- Key concepts and best practices covered
- Important implementation details
- Usage instructions and examples
- Any limitations or considerations

Python expertise:
- Python best practices and PEP standards
- Modern Python features (3.8+)
- Popular libraries and frameworks
- Performance optimization in Python
- Testing and debugging strategies
- Deployment and packaging

Code delivery requirements:
- ALWAYS provide complete, executable Python code that can run immediately
- Include ALL necessary imports and dependencies
- Demonstrate best practices with working code
- Include proper error handling and validation
- Add comprehensive documentation and comments
- Ensure code follows PEP 8 and modern Python standards
- Make examples production-ready with proper configuration
- Never provide incomplete code snippets or pseudo-code
- Always test your code examples to ensure they work correctly
- ALWAYS provide a comprehensive content summary

Remember: Your primary goal is to deliver working, complete code that users can run immediately.""",
                "category": "Python",
                "description": "Python development expert for modern Python applications with complete runnable code"
            },
            
            "Data Scientist": {
                "content": """You are a data science expert. Help with data analysis, machine learning, 
statistical modeling, and data visualization. Provide practical solutions for data-driven problems.

CRITICAL REQUIREMENT: You MUST always provide complete, runnable code examples for all data science solutions 
unless explicitly asked not to do so. Your code should be ready to execute immediately without any additional modifications.

CRITICAL REQUIREMENT: You MUST always provide a comprehensive content summary of your data science analysis, including:
- Overview of the analysis performed
- Key insights and findings
- Methodology and approach used
- Data processing steps
- Results interpretation and recommendations

Data science focus:
- Data analysis and exploration
- Machine learning algorithms and models
- Statistical analysis and hypothesis testing
- Data visualization and storytelling
- Feature engineering and selection
- Model evaluation and validation

Code delivery requirements:
- ALWAYS provide complete, executable Python code that can run immediately
- Include ALL necessary imports (pandas, numpy, scikit-learn, matplotlib, etc.)
- Include sample data or data generation code when needed
- Demonstrate complete workflows from data loading to results
- Add proper error handling and data validation
- Include visualization code with proper formatting
- Ensure code is reproducible with clear documentation
- Make examples production-ready with proper configuration
- Never provide incomplete code snippets or pseudo-code
- Always test your code examples to ensure they work correctly
- ALWAYS provide a comprehensive content summary

Remember: Your primary goal is to deliver working, complete code that users can run immediately.""",
                "category": "Data Science",
                "description": "Data science expert for analytics and machine learning with complete runnable code"
            }
        }
    
    @staticmethod
    def get_default_test_prompts() -> Dict[str, Dict[str, str]]:
        """Get default test prompt examples for AI model evaluation."""
        return {
            # Python Standard Library Examples - File Operations
            "File System Operations": {
                "content": "Create a comprehensive file system utility using Python standard library:\n1. List files and directories with os.listdir() and os.walk()\n2. Create, copy, move, and delete files with shutil\n3. Read and write text files with different encodings\n4. Work with CSV files using the csv module\n5. Handle JSON data with the json module\n6. Implement file searching and filtering\n7. Add file metadata extraction (size, modification time)\n8. Create a simple file backup system\n\nInclude error handling and demonstrate each operation with practical examples.",
                "category": "Standard Library",
                "description": "File system operations using Python standard library",
                "tags": "python,file-system,os,shutil,csv,json,backup"
            },
            
            "Text Processing and Analysis": {
                "content": "Build a text processing toolkit using Python standard library:\n1. Read and parse text files with different encodings\n2. Implement text cleaning and normalization\n3. Count words, characters, and lines using collections.Counter\n4. Find and replace text patterns with re (regex)\n5. Extract text statistics (word frequency, sentence length)\n6. Implement simple text search and filtering\n7. Handle different text formats (plain text, structured)\n8. Create a simple text analysis report generator\n\nInclude examples with sample text files and demonstrate text analysis capabilities.",
                "category": "Standard Library",
                "description": "Text processing and analysis using Python standard library",
                "tags": "python,text-processing,regex,collections,statistics,analysis"
            },
            
            "Data Structures and Algorithms": {
                "content": "Implement common data structures and algorithms using Python standard library:\n1. Create custom data structures (Stack, Queue, LinkedList)\n2. Implement sorting algorithms (bubble, merge, quick sort)\n3. Build search algorithms (linear, binary, depth-first, breadth-first)\n4. Use built-in data structures (list, dict, set, tuple) effectively\n5. Implement graph algorithms with adjacency lists\n6. Create tree data structures and traversal methods\n7. Add algorithm performance analysis and timing\n8. Include practical examples and use cases\n\nDemonstrate each data structure and algorithm with clear examples and performance comparisons.",
                "category": "Standard Library",
                "description": "Data structures and algorithms implementation",
                "tags": "python,data-structures,algorithms,sorting,searching,graphs,trees"
            },
            
            "Date and Time Operations": {
                "content": "Create a comprehensive date and time utility using Python standard library:\n1. Parse and format dates with datetime module\n2. Calculate time differences and durations\n3. Work with timezones using datetime and zoneinfo\n4. Generate date ranges and sequences\n5. Implement calendar operations and date arithmetic\n6. Handle different date formats and locales\n7. Create date-based filtering and sorting\n8. Build a simple event scheduler\n\nInclude examples for common date/time operations and demonstrate timezone handling.",
                "category": "Standard Library",
                "description": "Date and time operations using Python standard library",
                "tags": "python,datetime,timezone,calendar,scheduling,formatting"
            },
            
            "HTTP and Web Requests": {
                "content": "Build a web utility toolkit using Python standard library:\n1. Make HTTP requests with urllib.request\n2. Parse URLs and handle URL encoding/decoding\n3. Work with HTTP headers and status codes\n4. Download files and handle redirects\n5. Implement simple web scraping with urllib and html.parser\n6. Create a basic HTTP client with error handling\n7. Handle different content types (JSON, XML, HTML)\n8. Build a simple web API client\n\nInclude examples for common web operations and demonstrate robust error handling.",
                "category": "Standard Library",
                "description": "HTTP and web operations using Python standard library",
                "tags": "python,http,urllib,web-scraping,api-client,parsing"
            },
            
            "Database Operations (SQLite)": {
                "content": "Create a database management system using Python standard library (sqlite3):\n1. Create and manage SQLite databases\n2. Implement CRUD operations (Create, Read, Update, Delete)\n3. Handle database connections and transactions\n4. Create tables with proper schema design\n5. Implement data validation and constraints\n6. Add database backup and restore functionality\n7. Create simple queries and data analysis\n8. Build a basic ORM-like interface\n\nInclude examples for common database operations and demonstrate data management capabilities.",
                "category": "Standard Library",
                "description": "Database operations using Python sqlite3 module",
                "tags": "python,sqlite,database,crud,transactions,orm"
            },
            
            "Configuration and Settings Management": {
                "content": "Build a configuration management system using Python standard library:\n1. Read and write configuration files (INI, JSON, YAML-like)\n2. Implement environment variable handling with os.environ\n3. Create configuration validation and defaults\n4. Handle different configuration formats and sources\n5. Implement configuration inheritance and overrides\n6. Add configuration encryption and security\n7. Create configuration backup and versioning\n8. Build a simple settings manager class\n\nInclude examples for different configuration scenarios and demonstrate best practices.",
                "category": "Standard Library",
                "description": "Configuration management using Python standard library",
                "tags": "python,configuration,settings,environment-variables,validation"
            },
            
            "Logging and Debugging": {
                "content": "Create a comprehensive logging system using Python standard library:\n1. Set up logging with different levels (DEBUG, INFO, WARNING, ERROR)\n2. Configure log handlers (file, console, rotating files)\n3. Implement custom log formatters and filters\n4. Add structured logging with context information\n5. Create log rotation and archival\n6. Implement debug utilities and traceback handling\n7. Add performance logging and timing\n8. Build a simple debugging toolkit\n\nInclude examples for different logging scenarios and demonstrate debugging capabilities.",
                "category": "Standard Library",
                "description": "Logging and debugging using Python standard library",
                "tags": "python,logging,debugging,traceback,performance,handlers"
            },
            
            "Multithreading and Concurrency": {
                "content": "Implement concurrent programming using Python standard library:\n1. Create and manage threads with threading module\n2. Implement thread synchronization (locks, semaphores)\n3. Use ThreadPoolExecutor for parallel execution\n4. Handle thread communication and data sharing\n5. Implement producer-consumer patterns\n6. Add thread safety and race condition prevention\n7. Create simple parallel processing utilities\n8. Build a basic task scheduler\n\nInclude examples for common concurrency patterns and demonstrate thread safety.",
                "category": "Standard Library",
                "description": "Multithreading and concurrency using Python standard library",
                "tags": "python,threading,concurrency,synchronization,parallel-processing"
            },
            
            "Regular Expressions and Pattern Matching": {
                "content": "Build a pattern matching toolkit using Python standard library (re module):\n1. Create and test regular expressions\n2. Implement pattern matching and searching\n3. Handle different regex flags and options\n4. Use regex groups and capturing\n5. Implement text validation and cleaning\n6. Create regex-based text parsing\n7. Add performance optimization for regex\n8. Build a simple regex testing utility\n\nInclude examples for common regex patterns and demonstrate text processing capabilities.",
                "category": "Standard Library",
                "description": "Regular expressions and pattern matching using Python re module",
                "tags": "python,regex,pattern-matching,text-processing,validation"
            },
            
            "Collections and Itertools": {
                "content": "Create advanced data processing using Python standard library collections and itertools:\n1. Use collections.Counter for frequency counting\n2. Implement defaultdict for automatic key creation\n3. Use namedtuple for structured data\n4. Create custom iterators with itertools\n5. Implement data grouping and aggregation\n6. Use itertools for combinations and permutations\n7. Add data streaming and lazy evaluation\n8. Build a simple data processing pipeline\n\nInclude examples for common data processing tasks and demonstrate collection utilities.",
                "category": "Standard Library",
                "description": "Advanced data processing using collections and itertools",
                "tags": "python,collections,itertools,data-processing,aggregation"
            },
            
            "Math and Statistics": {
                "content": "Build a mathematical computing toolkit using Python standard library:\n1. Use math module for mathematical functions\n2. Implement statistical calculations with statistics module\n3. Create random number generation with random module\n4. Add mathematical constants and precision handling\n5. Implement basic statistical analysis\n6. Create mathematical utilities and helpers\n7. Add number formatting and conversion\n8. Build a simple calculator with advanced functions\n\nInclude examples for common mathematical operations and demonstrate statistical analysis.",
                "category": "Standard Library",
                "description": "Mathematical computing using Python standard library",
                "tags": "python,math,statistics,random,calculations,analysis"
            },
            
            "System and Process Management": {
                "content": "Create a system management utility using Python standard library:\n1. Get system information with platform and os modules\n2. Manage processes with subprocess module\n3. Handle file permissions and ownership\n4. Implement system monitoring and resource usage\n5. Create process communication and piping\n6. Add system command execution and automation\n7. Implement basic system administration tasks\n8. Build a simple system information dashboard\n\nInclude examples for common system operations and demonstrate process management.",
                "category": "Standard Library",
                "description": "System and process management using Python standard library",
                "tags": "python,system,process,platform,subprocess,monitoring"
            },
            
            "Data Serialization and Persistence": {
                "content": "Build a data persistence system using Python standard library:\n1. Use pickle for object serialization\n2. Implement JSON data handling\n3. Work with XML using xml.etree.ElementTree\n4. Create custom serialization formats\n5. Add data compression with gzip and zlib\n6. Implement data validation and schema checking\n7. Create data migration and versioning\n8. Build a simple object storage system\n\nInclude examples for different data formats and demonstrate persistence capabilities.",
                "category": "Standard Library",
                "description": "Data serialization and persistence using Python standard library",
                "tags": "python,serialization,pickle,json,xml,compression,persistence"
            },
            
            "Network Programming": {
                "content": "Create network utilities using Python standard library:\n1. Implement socket programming for TCP/UDP\n2. Create simple client-server applications\n3. Handle network connections and data transfer\n4. Implement basic network protocols\n5. Add network error handling and timeouts\n6. Create network monitoring utilities\n7. Implement simple network services\n8. Build a basic network testing toolkit\n\nInclude examples for common network operations and demonstrate client-server communication.",
                "category": "Standard Library",
                "description": "Network programming using Python standard library",
                "tags": "python,networking,sockets,tcp,udp,client-server"
            },
            
            "Email and Messaging": {
                "content": "Build an email utility using Python standard library:\n1. Send emails with smtplib module\n2. Parse and read emails with email module\n3. Handle email attachments and MIME types\n4. Implement email validation and formatting\n5. Create email templates and automation\n6. Add email filtering and processing\n7. Implement simple email client functionality\n8. Build an email notification system\n\nInclude examples for common email operations and demonstrate email automation.",
                "category": "Standard Library",
                "description": "Email and messaging using Python standard library",
                "tags": "python,email,smtp,mime,validation,automation"
            },
            
            "Compression and Archiving": {
                "content": "Create compression utilities using Python standard library:\n1. Compress and decompress files with gzip\n2. Work with ZIP archives using zipfile\n3. Implement tar archive handling\n4. Add compression algorithms comparison\n5. Create archive extraction and creation\n6. Handle different compression formats\n7. Implement file backup with compression\n8. Build a simple archiving utility\n\nInclude examples for different compression formats and demonstrate archiving capabilities.",
                "category": "Standard Library",
                "description": "Compression and archiving using Python standard library",
                "tags": "python,compression,gzip,zipfile,tar,archiving"
            },
            
            "Cryptography and Security": {
                "content": "Build security utilities using Python standard library:\n1. Generate secure random numbers with secrets module\n2. Implement basic hashing with hashlib\n3. Create password hashing and verification\n4. Add basic encryption and decryption\n5. Implement secure token generation\n6. Handle secure file operations\n7. Create basic security utilities\n8. Build a simple password manager\n\nInclude examples for common security operations and demonstrate cryptographic capabilities.",
                "category": "Standard Library",
                "description": "Cryptography and security using Python standard library",
                "tags": "python,cryptography,security,hashing,encryption,secrets"
            },
            
            "Testing and Quality Assurance": {
                "content": "Create a testing framework using Python standard library:\n1. Write unit tests with unittest module\n2. Implement test fixtures and setup/teardown\n3. Create test suites and test discovery\n4. Add assertion methods and test helpers\n5. Implement mock objects and test doubles\n6. Create test reporting and coverage\n7. Add performance testing utilities\n8. Build a simple test runner\n\nInclude examples for different testing scenarios and demonstrate testing best practices.",
                "category": "Standard Library",
                "description": "Testing and quality assurance using Python standard library",
                "tags": "python,testing,unittest,mocking,coverage,quality"
            },
            
            "Command Line Interface": {
                "content": "Build a command-line interface using Python standard library:\n1. Parse command line arguments with argparse\n2. Create interactive command-line applications\n3. Handle user input and validation\n4. Implement command help and documentation\n5. Add command history and completion\n6. Create subcommands and command groups\n7. Implement configuration file handling\n8. Build a simple CLI framework\n\nInclude examples for common CLI patterns and demonstrate user interface design.",
                "category": "Standard Library",
                "description": "Command-line interface using Python standard library",
                "tags": "python,cli,argparse,interactive,commands,interface"
            },
            
            "Data Validation and Sanitization": {
                "content": "Create a data validation system using Python standard library:\n1. Implement input validation and sanitization\n2. Create data type checking and conversion\n3. Add format validation (email, phone, date)\n4. Implement data cleaning and normalization\n5. Create validation rules and constraints\n6. Add error reporting and feedback\n7. Implement data transformation utilities\n8. Build a simple validation framework\n\nInclude examples for common validation scenarios and demonstrate data quality assurance.",
                "category": "Standard Library",
                "description": "Data validation and sanitization using Python standard library",
                "tags": "python,validation,sanitization,data-quality,constraints"
            },
            
            "Error Handling and Exception Management": {
                "content": "Build a robust error handling system using Python standard library:\n1. Implement custom exception classes\n2. Create exception hierarchies and inheritance\n3. Add context managers for resource management\n4. Implement error recovery and fallback strategies\n5. Create error logging and reporting\n6. Add exception chaining and re-raising\n7. Implement error handling patterns\n8. Build a simple error management framework\n\nInclude examples for common error scenarios and demonstrate robust error handling.",
                "category": "Standard Library",
                "description": "Error handling and exception management using Python standard library",
                "tags": "python,exceptions,error-handling,context-managers,recovery"
            },
            
            "Performance Monitoring and Profiling": {
                "content": "Create performance monitoring utilities using Python standard library:\n1. Use time module for timing operations\n2. Implement performance profiling with cProfile\n3. Create memory usage monitoring\n4. Add execution time analysis\n5. Implement performance benchmarking\n6. Create performance reporting and visualization\n7. Add resource usage tracking\n8. Build a simple performance monitoring toolkit\n\nInclude examples for performance analysis and demonstrate optimization techniques.",
                "category": "Standard Library",
                "description": "Performance monitoring and profiling using Python standard library",
                "tags": "python,performance,profiling,monitoring,benchmarking,optimization"
            }
        }

def initialize_default_prompts(db: PromptDatabase):
    """Initialize the database with default prompts."""
    # Add default system prompts
    default_system_prompts = DefaultPrompts.get_default_system_prompts()
    for name, prompt_data in default_system_prompts.items():
        db.add_system_prompt(
            name=name,
            content=prompt_data["content"],
            category=prompt_data["category"],
            description=prompt_data["description"]
        )
    
    # Add default test prompt examples
    default_test_prompts = DefaultPrompts.get_default_test_prompts()
    for name, prompt_data in default_test_prompts.items():
        db.add_user_prompt(
            name=name,
            content=prompt_data["content"],
            category=prompt_data["category"],
            description=prompt_data["description"],
            tags=prompt_data["tags"]
        )
