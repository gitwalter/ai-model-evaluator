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
            # LangChain Library Examples
            "LangChain Agent with Tools": {
                "content": "Create a LangChain agent that uses multiple tools with the free Gemini API:\n1. Use DuckDuckGo search tool for web queries\n2. Implement a calculator tool for mathematical operations\n3. Add a weather API tool for location-based data\n4. Include conversation memory with ConversationBufferMemory\n5. Handle tool selection and execution errors\n6. Provide structured output with tool usage history\n\nUse the free Gemini API (gemini-1.5-flash-latest) as the LLM. Include complete agent configuration, tool definitions, and example interactions that can be run directly in the AI Model Evaluator app.",
                "category": "LangChain",
                "description": "LangChain agent with multiple tools and memory using Gemini API",
                "tags": "langchain,agent,tools,memory,duckduckgo,calculator,gemini"
            },
            
            "LangChain RAG with ChromaDB": {
                "content": "Build a Retrieval-Augmented Generation system using LangChain and ChromaDB with the free Gemini API:\n1. Load documents using LangChain document loaders (PDF, TXT, DOCX)\n2. Implement text chunking with RecursiveCharacterTextSplitter\n3. Create embeddings using free embedding models (sentence-transformers)\n4. Store vectors in ChromaDB with metadata\n5. Implement similarity search with MMR diversity\n6. Generate responses using the free Gemini API (gemini-1.5-flash-latest) with retrieved context\n7. Add conversation memory for follow-up questions\n8. Include evaluation metrics and performance monitoring\n\nProvide complete implementation with example documents and queries that can be run directly in the AI Model Evaluator app.",
                "category": "LangChain",
                "description": "RAG system with LangChain and ChromaDB using Gemini API",
                "tags": "langchain,rag,chromadb,embeddings,retrieval,memory,gemini"
            },
            
            "LangChain Chain Composition": {
                "content": "Create a complex LangChain chain that combines multiple components using the free Gemini API:\n1. LLMChain for basic text generation using Gemini\n2. SequentialChain for multi-step processing\n3. RouterChain for dynamic tool selection\n4. TransformChain for data preprocessing\n5. Custom chain for business logic\n6. Include error handling and fallback strategies\n7. Add monitoring and logging\n8. Implement chain validation and testing\n\nUse the free Gemini API (gemini-1.5-flash-latest) as the LLM. Demonstrate with a practical use case like content analysis and recommendation that can be run directly in the AI Model Evaluator app.",
                "category": "LangChain",
                "description": "Complex LangChain chain composition using Gemini API",
                "tags": "langchain,chain,composition,router,transform,monitoring,gemini"
            },
            
            # Pandas Library Examples
            "Pandas Data Analysis Pipeline": {
                "content": "Create a comprehensive data analysis pipeline using pandas:\n1. Load data from multiple sources (CSV, Excel, JSON, SQL)\n2. Perform exploratory data analysis (EDA) with pandas profiling\n3. Handle missing values using various strategies (drop, fill, interpolate)\n4. Implement data cleaning (duplicates, outliers, data types)\n5. Create feature engineering functions (date features, aggregations)\n6. Perform statistical analysis and hypothesis testing\n7. Generate visualizations with pandas plotting\n8. Export results to multiple formats\n\nInclude sample datasets and demonstrate each step with real examples.",
                "category": "Pandas",
                "description": "Complete pandas data analysis and processing pipeline",
                "tags": "pandas,data-analysis,eda,cleaning,feature-engineering,visualization"
            },
            
            "Pandas Performance Optimization": {
                "content": "Optimize pandas operations for large datasets:\n1. Use vectorized operations instead of loops\n2. Implement efficient data types (categorical, datetime)\n3. Apply chunking for memory management\n4. Use pandas query() for complex filtering\n5. Implement parallel processing with multiprocessing\n6. Optimize groupby operations with agg() and transform()\n7. Use pandas eval() for complex expressions\n8. Profile performance with memory_usage() and timeit\n\nDemonstrate performance improvements with benchmarks and memory profiling.",
                "category": "Pandas",
                "description": "Pandas performance optimization for large datasets",
                "tags": "pandas,optimization,vectorization,memory,performance,benchmarking"
            },
            
            "Pandas Time Series Analysis": {
                "content": "Build a time series analysis system using pandas:\n1. Load and preprocess time series data with proper datetime indexing\n2. Implement resampling operations (daily, weekly, monthly aggregations)\n3. Calculate rolling statistics (mean, std, min, max)\n4. Perform seasonal decomposition and trend analysis\n5. Implement lag features and time-based feature engineering\n6. Handle timezone conversions and daylight saving time\n7. Create time-based visualizations and plots\n8. Export time series data with proper formatting\n\nInclude examples with financial, weather, or IoT sensor data.",
                "category": "Pandas",
                "description": "Time series analysis and processing with pandas",
                "tags": "pandas,time-series,resampling,rolling,seasonal,visualization"
            },
            
            # NumPy Library Examples
            "NumPy Linear Algebra Operations": {
                "content": "Implement comprehensive linear algebra operations using NumPy:\n1. Matrix operations (multiplication, inverse, determinant, eigenvalues)\n2. Solve linear systems using numpy.linalg.solve()\n3. Perform Singular Value Decomposition (SVD)\n4. Implement Principal Component Analysis (PCA)\n5. Calculate correlation matrices and covariance\n6. Perform matrix factorizations (LU, QR, Cholesky)\n7. Implement custom linear algebra functions\n8. Optimize operations for large matrices\n\nInclude mathematical explanations and practical applications.",
                "category": "NumPy",
                "description": "Advanced linear algebra operations with NumPy",
                "tags": "numpy,linear-algebra,matrix,svd,pca,factorization,optimization"
            },
            
            "NumPy Signal Processing": {
                "content": "Create a signal processing toolkit using NumPy:\n1. Generate synthetic signals (sine, square, sawtooth waves)\n2. Implement Fast Fourier Transform (FFT) and inverse FFT\n3. Apply digital filters (low-pass, high-pass, band-pass)\n4. Perform convolution and correlation operations\n5. Implement window functions (Hamming, Hanning, Blackman)\n6. Add noise and perform noise reduction\n7. Analyze signal frequency content\n8. Create signal visualization and analysis tools\n\nInclude examples with audio, image, and sensor signal processing.",
                "category": "NumPy",
                "description": "Signal processing and analysis with NumPy",
                "tags": "numpy,signal-processing,fft,filters,convolution,noise,visualization"
            },
            
            "NumPy Statistical Computing": {
                "content": "Build a statistical computing framework using NumPy:\n1. Generate random numbers from various distributions\n2. Calculate descriptive statistics (mean, median, std, percentiles)\n3. Implement hypothesis testing (t-test, chi-square, ANOVA)\n4. Perform Monte Carlo simulations\n5. Calculate confidence intervals and p-values\n6. Implement regression analysis and correlation\n7. Create statistical visualization functions\n8. Add statistical validation and testing\n\nInclude examples with real-world datasets and statistical analysis.",
                "category": "NumPy",
                "description": "Statistical computing and analysis with NumPy",
                "tags": "numpy,statistics,distributions,hypothesis-testing,monte-carlo,regression"
            },
            
            # BeautifulSoup (bs4) Library Examples
            "BeautifulSoup Web Scraping": {
                "content": "Create a comprehensive web scraping system using BeautifulSoup:\n1. Parse HTML and XML documents with different parsers\n2. Navigate DOM tree using tags, classes, and IDs\n3. Extract text, links, images, and structured data\n4. Handle dynamic content and JavaScript-rendered pages\n5. Implement robust error handling and retry logic\n6. Add rate limiting and respect robots.txt\n7. Store scraped data in structured formats (CSV, JSON, SQL)\n8. Create reusable scraping functions and classes\n\nInclude examples for e-commerce, news, and social media scraping.",
                "category": "BeautifulSoup",
                "description": "Web scraping and HTML parsing with BeautifulSoup",
                "tags": "beautifulsoup,web-scraping,html-parsing,dom-navigation,data-extraction"
            },
            
            "BeautifulSoup Data Extraction": {
                "content": "Build a data extraction pipeline using BeautifulSoup:\n1. Extract tables and convert to pandas DataFrames\n2. Parse forms and extract form data\n3. Extract metadata (title, description, keywords)\n4. Handle nested structures and complex selectors\n5. Implement data cleaning and validation\n6. Add support for different character encodings\n7. Create data transformation functions\n8. Export data to multiple formats\n\nDemonstrate with real websites and complex data structures.",
                "category": "BeautifulSoup",
                "description": "Structured data extraction and parsing with BeautifulSoup",
                "tags": "beautifulsoup,data-extraction,tables,forms,metadata,cleaning"
            },
            
            # Scikit-learn Library Examples
            "Scikit-learn Classification Pipeline": {
                "content": "Build a complete classification pipeline using scikit-learn:\n1. Load and preprocess data with sklearn.preprocessing\n2. Implement feature selection and dimensionality reduction\n3. Train multiple classifiers (Random Forest, SVM, Logistic Regression)\n4. Perform cross-validation and hyperparameter tuning\n5. Evaluate models with multiple metrics (accuracy, precision, recall, F1)\n6. Create confusion matrices and classification reports\n7. Implement ensemble methods (Voting, Bagging, Boosting)\n8. Add model persistence and deployment preparation\n\nInclude comprehensive evaluation and comparison of different algorithms.",
                "category": "Scikit-learn",
                "description": "Complete classification pipeline with scikit-learn",
                "tags": "scikit-learn,classification,preprocessing,feature-selection,ensemble,evaluation"
            },
            
            "Scikit-learn Clustering Analysis": {
                "content": "Implement clustering analysis using scikit-learn:\n1. Perform K-means clustering with optimal k selection\n2. Implement hierarchical clustering and dendrograms\n3. Apply DBSCAN for density-based clustering\n4. Use Gaussian Mixture Models (GMM)\n5. Evaluate clustering quality with silhouette scores\n6. Visualize clusters in 2D and 3D space\n7. Handle high-dimensional data with dimensionality reduction\n8. Create clustering validation and comparison tools\n\nInclude examples with customer segmentation and data exploration.",
                "category": "Scikit-learn",
                "description": "Clustering analysis and visualization with scikit-learn",
                "tags": "scikit-learn,clustering,kmeans,hierarchical,dbscan,gmm,visualization"
            },
            
            # TensorFlow/Keras Library Examples
            "TensorFlow Neural Network": {
                "content": "Build a custom neural network using TensorFlow/Keras:\n1. Create custom layers and models using Keras API\n2. Implement different architectures (MLP, CNN, RNN, LSTM)\n3. Add regularization techniques (Dropout, BatchNorm, L1/L2)\n4. Implement custom loss functions and metrics\n5. Use callbacks for model monitoring and early stopping\n6. Add TensorBoard integration for experiment tracking\n7. Implement model checkpointing and restoration\n8. Create data generators and augmentation pipelines\n\nInclude examples for image classification, text processing, and time series.",
                "category": "TensorFlow",
                "description": "Custom neural network implementation with TensorFlow/Keras",
                "tags": "tensorflow,keras,neural-network,custom-layers,regularization,callbacks"
            },
            
            "TensorFlow Transfer Learning": {
                "content": "Implement transfer learning using TensorFlow/Keras:\n1. Load pre-trained models (ResNet, VGG, BERT, GPT)\n2. Fine-tune models for specific tasks\n3. Implement feature extraction and classification heads\n4. Add data augmentation and preprocessing\n5. Use learning rate scheduling and optimization\n6. Implement model evaluation and testing\n7. Add model interpretation and visualization\n8. Create deployment-ready models (SavedModel, TFLite)\n\nInclude examples for computer vision and NLP transfer learning.",
                "category": "TensorFlow",
                "description": "Transfer learning and model adaptation with TensorFlow",
                "tags": "tensorflow,transfer-learning,pretrained-models,fine-tuning,deployment"
            },
            
            # Selenium Library Examples
            "Selenium Web Automation": {
                "content": "Create a web automation framework using Selenium:\n1. Set up WebDriver for different browsers (Chrome, Firefox, Edge)\n2. Implement page object model (POM) design pattern\n3. Handle dynamic elements and wait strategies\n4. Perform form filling and submission\n5. Implement screenshot and video recording\n6. Add parallel test execution\n7. Handle authentication and session management\n8. Create reusable automation functions and classes\n\nInclude examples for e-commerce testing and web scraping automation.",
                "category": "Selenium",
                "description": "Web automation and testing with Selenium",
                "tags": "selenium,web-automation,webdriver,pom,wait-strategies,parallel-testing"
            },
            
            # FastAPI Library Examples
            "FastAPI REST API": {
                "content": "Build a production-ready REST API using FastAPI:\n1. Create API endpoints with proper HTTP methods\n2. Implement request/response models with Pydantic\n3. Add authentication and authorization (JWT, OAuth)\n4. Implement database integration (SQLAlchemy, async)\n5. Add input validation and error handling\n6. Implement rate limiting and caching\n7. Add API documentation with automatic OpenAPI generation\n8. Create testing framework with pytest\n\nInclude examples for user management, data processing, and external integrations.",
                "category": "FastAPI",
                "description": "Production REST API with FastAPI and Pydantic",
                "tags": "fastapi,rest-api,pydantic,authentication,database,testing"
            },
            
            # OpenCV Library Examples
            "OpenCV Image Processing": {
                "content": "Create an image processing pipeline using OpenCV:\n1. Load and save images in various formats\n2. Implement basic operations (resize, crop, rotate, flip)\n3. Apply filters and transformations (blur, sharpen, edge detection)\n4. Perform color space conversions and histogram analysis\n5. Implement object detection and recognition\n6. Add face detection and recognition capabilities\n7. Create video processing and analysis\n8. Implement real-time image processing\n\nInclude examples for computer vision applications and image analysis.",
                "category": "OpenCV",
                "description": "Image and video processing with OpenCV",
                "tags": "opencv,image-processing,computer-vision,object-detection,face-recognition"
            },
            
            # LangGraph Library Examples
            "LangGraph Simple Agent": {
                "content": "Build a simple agent using LangGraph with the free Gemini API:\n1. Define agent state and configuration\n2. Create nodes for different agent actions (thinking, tool use, response)\n3. Implement conditional edges for agent decision making\n4. Add memory and conversation history\n5. Handle tool selection and execution\n6. Implement error handling and fallback strategies\n7. Add monitoring and logging capabilities\n8. Create a simple chat interface\n\nUse the free Gemini API (gemini-1.5-flash-latest) as the LLM. Demonstrate with a practical use case like a task planning agent or a simple assistant that can be run directly in the AI Model Evaluator app.",
                "category": "LangGraph",
                "description": "Simple agent implementation with LangGraph using Gemini API",
                "tags": "langgraph,agent,state-management,nodes,edges,memory,gemini"
            },
            
            "LangGraph Multi-Agent System": {
                "content": "Create a multi-agent system using LangGraph with the free Gemini API:\n1. Design agent roles and responsibilities\n2. Implement inter-agent communication protocols\n3. Create shared state management and coordination\n4. Add agent discovery and registration mechanisms\n5. Implement task delegation and routing\n6. Handle agent conflicts and consensus building\n7. Add monitoring and performance tracking\n8. Create a unified interface for multi-agent interactions\n\nUse the free Gemini API (gemini-1.5-flash-latest) as the LLM for all agents. Demonstrate with a practical use case like a customer service system with specialized agents (billing, technical support, sales) that can be run directly in the AI Model Evaluator app.",
                "category": "LangGraph",
                "description": "Multi-agent system orchestration with LangGraph using Gemini API",
                "tags": "langgraph,multi-agent,communication,coordination,delegation,consensus,gemini"
            },
            
            # RAG Libraries Examples
            "LlamaIndex RAG with ChromaDB": {
                "content": "Build a RAG system using LlamaIndex and ChromaDB with the free Gemini API:\n1. Load documents using LlamaIndex document loaders\n2. Implement document chunking and preprocessing\n3. Create embeddings using free embedding models (sentence-transformers)\n4. Store vectors in ChromaDB with metadata\n5. Implement similarity search and retrieval\n6. Add query processing and response generation using the free Gemini API (gemini-1.5-flash-latest)\n7. Implement conversation memory and context management\n8. Add evaluation metrics and performance monitoring\n\nInclude examples with different document types (PDF, web pages, structured data) and demonstrate query answering capabilities that can be run directly in the AI Model Evaluator app.",
                "category": "RAG",
                "description": "RAG system with LlamaIndex and ChromaDB using Gemini API",
                "tags": "llamaindex,chromadb,rag,embeddings,retrieval,conversation,gemini"
            },
            
            "LlamaIndex RAG with Qdrant": {
                "content": "Build a RAG system using LlamaIndex and Qdrant with the free Gemini API:\n1. Set up Qdrant vector database with proper configuration\n2. Load and process documents with LlamaIndex\n3. Create embeddings using free embedding models (sentence-transformers) and store in Qdrant collections\n4. Implement advanced search strategies (filtering, scoring)\n5. Add hybrid search combining dense and sparse retrievers\n6. Implement query expansion and refinement\n7. Add result ranking and re-ranking\n8. Create a production-ready API interface using the free Gemini API (gemini-1.5-flash-latest) for response generation\n\nDemonstrate with large-scale document collections and complex query scenarios that can be run directly in the AI Model Evaluator app.",
                "category": "RAG",
                "description": "RAG system with LlamaIndex and Qdrant using Gemini API",
                "tags": "llamaindex,qdrant,rag,hybrid-search,ranking,api,gemini"
            },
            
            # MCP (Model Context Protocol) Examples
            "MCP Server for File System": {
                "content": "Build an MCP (Model Context Protocol) server for file system access that integrates with the free Gemini API:\n1. Implement MCP server protocol and message handling\n2. Create file system operations (read, write, list, search)\n3. Add file metadata and content analysis capabilities using the free Gemini API (gemini-1.5-flash-latest)\n4. Implement file watching and change notifications\n5. Add security and access control mechanisms\n6. Handle large files and streaming operations\n7. Implement caching and performance optimization\n8. Create comprehensive error handling and logging\n\nInclude examples for common file operations and demonstrate integration with the free Gemini API that can be run directly in the AI Model Evaluator app.",
                "category": "MCP",
                "description": "MCP server for file system operations with Gemini API",
                "tags": "mcp,file-system,protocol,operations,security,caching,gemini"
            },
            
            "MCP Server for Google Functions": {
                "content": "Build an MCP server for Google Cloud Functions that integrates with the free Gemini API:\n1. Implement MCP server for Google Cloud integration\n2. Create function deployment and management operations\n3. Add function invocation and monitoring capabilities\n4. Implement environment variable and configuration management\n5. Add logging and error tracking integration\n6. Handle authentication and authorization\n7. Implement cost monitoring and optimization using the free Gemini API (gemini-1.5-flash-latest) for analysis\n8. Create deployment pipelines and CI/CD integration\n\nDemonstrate with practical use cases like automated function deployment and monitoring that can be run directly in the AI Model Evaluator app.",
                "category": "MCP",
                "description": "MCP server for Google Cloud Functions with Gemini API",
                "tags": "mcp,google-cloud,functions,deployment,monitoring,ci-cd,gemini"
            },
            
            "Agent with MCP File System": {
                "content": "Build an AI agent that uses an MCP server for file system access with the free Gemini API:\n1. Create an agent that connects to the file system MCP server\n2. Implement file reading, writing, and analysis capabilities using the free Gemini API (gemini-1.5-flash-latest)\n3. Add document processing and content extraction\n4. Implement file organization and management tasks\n5. Add search and filtering capabilities\n6. Handle file format conversion and processing\n7. Implement backup and synchronization features\n8. Create a user-friendly interface for file operations\n\nDemonstrate with practical use cases like document analysis, file organization, and automated file processing that can be run directly in the AI Model Evaluator app.",
                "category": "MCP",
                "description": "AI agent using MCP server for file operations with Gemini API",
                "tags": "mcp,agent,file-operations,document-processing,automation,gemini"
            },
            
            "Agent with MCP Google Functions": {
                "content": "Build an AI agent that uses an MCP server for Google Cloud Functions with the free Gemini API:\n1. Create an agent that connects to the Google Functions MCP server\n2. Implement function deployment and management automation using the free Gemini API (gemini-1.5-flash-latest)\n3. Add monitoring and alerting capabilities\n4. Implement cost optimization and resource management\n5. Add function testing and validation\n6. Handle environment configuration and secrets management\n7. Implement automated scaling and performance tuning\n8. Create deployment workflows and rollback mechanisms\n\nDemonstrate with practical use cases like automated function deployment, monitoring, and optimization that can be run directly in the AI Model Evaluator app.",
                "category": "MCP",
                "description": "AI agent using MCP server for Google Functions with Gemini API",
                "tags": "mcp,agent,google-functions,automation,monitoring,optimization,gemini"
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
