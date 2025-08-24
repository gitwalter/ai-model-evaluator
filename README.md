# AI Model Evaluator

A comprehensive testing and evaluation framework for Google's Gemini API. Features automated testing, manual testing via Streamlit, and detailed evaluation reports with code execution capabilities.

## 🚀 Features

- **Automated Model Testing**: Evaluate multiple AI models with the same prompt
- **Interactive Testing**: Streamlit app for manual model testing and comparison
- **Comprehensive Evaluation**: Detailed analysis of responses across multiple criteria
- **Report Generation**: Automated generation of evaluation reports and JSON data
- **System Prompt Support**: Advanced prompting with system message capabilities
- **Dynamic Model Discovery**: Automatically fetch available models from APIs
- **📝 Prompt Management**: SQLite database for saving, editing, and organizing prompts
- **🧪 Test Prompt Examples**: Ready-to-use prompt examples for AI model evaluation
- **🤖 Multi-Model Testing**: Test multiple models simultaneously with batch processing
- **🔧 Dual Prompt Support**: Use both system prompts and test prompt examples simultaneously
- **📊 Visual Analytics**: Enhanced response visualization with charts and metrics
- **🚀 Code Execution**: Execute AI-generated Python code directly in the UI with real-time results

## 📁 Project Structure

```
ai-model-evaluator/
├── README.md                    # This comprehensive documentation
├── requirements.txt            # Python dependencies
├── test_gemini_models.py       # Automated testing script
├── streamlit_app.py            # Interactive testing app
├── prompt_manager.py           # Prompt management system
├── prompts.db                  # SQLite database for prompts (auto-generated)
├── .gitignore                  # Git ignore rules
├── .streamlit/                 # Streamlit configuration
│   ├── secrets.toml.example    # Example secrets file (committed to git)
│   └── secrets.toml           # Your actual API keys (not tracked by git)
└── .env                        # Environment variables (legacy, not tracked)
```

## 🛠️ Setup

### Prerequisites
- Python 3.9 or higher
- Anaconda or Miniconda installed
- Gemini API key

### Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key:**

   **Option 1: Streamlit Cloud (Recommended)**
   - Deploy to Streamlit Cloud
   - Go to Settings → Secrets
   - Add: `GEMINI_API_KEY = "your_gemini_api_key_here"`

   **Option 2: Local Development**
   - Copy the example file: `cp .streamlit/secrets.toml.example .streamlit/secrets.toml`
   - Edit `.streamlit/secrets.toml` and add your actual API key:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

   **Option 3: Manual Entry (Easiest)**
   - Run the Streamlit app
   - Enter your API key directly in the sidebar
   - Click "Test & Save" to validate

   **Option 4: Environment Variable (Legacy)**
   - Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key (starts with "AIzaSyC...")

## 📝 Prompt Management System

The AI Model Evaluator now includes a comprehensive prompt management system with SQLite database support.

### Features

- **📚 Database Storage**: All prompts are stored in a SQLite database (`prompts.db`)
- **🔄 System Prompts**: Pre-configured system prompts for different development scenarios
- **📋 User Prompts**: Template prompts for common development tasks
- **🏷️ Categorization**: Organize prompts by categories (Development, Security, Performance, etc.)
- **✏️ Edit & Delete**: Full CRUD operations for managing prompts
- **🔍 Search & Filter**: Easy navigation through prompt categories

### Default System Prompts

The system comes with pre-built system prompts for:

- **Code Producer (Default)**: Fast code production with minimal explanation but good documentation
- **Python Development Expert**: Practical Python development guidance for production applications
- **Code Review Agent**: Expert code review with quality analysis
- **Debugging Assistant**: Step-by-step debugging guidance
- **Documentation Writer**: Technical documentation creation
- **Security Expert**: Security analysis and vulnerability assessment
- **Performance Optimizer**: Performance optimization strategies
- **Python Expert**: Python-specific development guidance
- **Data Scientist**: Data analysis and machine learning support

### Default Test Prompt Examples

Ready-to-use test prompt examples for AI model evaluation, specifically designed for Python AI integration developers:

#### **🔗 LangChain Library**
- **LangChain Agent with Tools**: Multi-tool agent with DuckDuckGo search, calculator, and weather API using Gemini API
- **LangChain RAG with ChromaDB**: Retrieval-Augmented Generation with vector database integration using Gemini API
- **LangChain Chain Composition**: Complex chain orchestration with routing and transformation using Gemini API

#### **🔄 LangGraph Library**
- **LangGraph Simple Agent**: Simple agent implementation with state management and decision making using Gemini API
- **LangGraph Multi-Agent System**: Multi-agent system orchestration with communication and coordination using Gemini API

#### **📚 RAG Libraries**
- **LlamaIndex RAG with ChromaDB**: RAG system with LlamaIndex and ChromaDB using Gemini API for response generation
- **LlamaIndex RAG with Qdrant**: Advanced RAG system with LlamaIndex and Qdrant using Gemini API for response generation

#### **🔧 MCP (Model Context Protocol)**
- **MCP Server for File System**: MCP server for file system operations with Gemini API integration
- **MCP Server for Google Functions**: MCP server for Google Cloud Functions with Gemini API integration
- **Agent with MCP File System**: AI agent using MCP server for file operations with Gemini API
- **Agent with MCP Google Functions**: AI agent using MCP server for Google Functions with Gemini API

#### **📊 Pandas Library**
- **Pandas Data Analysis Pipeline**: Complete EDA and data processing workflow
- **Pandas Performance Optimization**: Vectorization, memory management, and parallel processing
- **Pandas Time Series Analysis**: Time-based data processing and analysis

#### **🔢 NumPy Library**
- **NumPy Linear Algebra Operations**: Matrix operations, SVD, PCA, and factorizations
- **NumPy Signal Processing**: FFT, filters, convolution, and signal analysis
- **NumPy Statistical Computing**: Distributions, hypothesis testing, and Monte Carlo simulations

#### **🌐 BeautifulSoup Library**
- **BeautifulSoup Web Scraping**: HTML parsing and web data extraction
- **BeautifulSoup Data Extraction**: Structured data extraction from tables and forms

#### **🤖 Scikit-learn Library**
- **Scikit-learn Classification Pipeline**: Complete ML classification workflow
- **Scikit-learn Clustering Analysis**: K-means, hierarchical, DBSCAN, and GMM clustering

#### **🧠 TensorFlow Library**
- **TensorFlow Neural Network**: Custom neural network architectures and training
- **TensorFlow Transfer Learning**: Pre-trained model adaptation and fine-tuning

#### **🖥️ Selenium Library**
- **Selenium Web Automation**: Browser automation and testing framework

#### **⚡ FastAPI Library**
- **FastAPI REST API**: Production-ready API development with authentication

#### **👁️ OpenCV Library**
- **OpenCV Image Processing**: Computer vision and image processing pipeline

These prompts cover the essential Python libraries and frameworks for building production-ready AI systems:
- **LangChain & LangGraph** for agent development and RAG systems
- **LlamaIndex** for advanced RAG implementations
- **MCP** for Model Context Protocol integrations
- **Pandas** for data manipulation and analysis
- **NumPy** for scientific computing and numerical operations
- **BeautifulSoup** for web scraping and HTML parsing
- **Scikit-learn** for machine learning pipelines
- **TensorFlow** for deep learning and neural networks
- **Selenium** for web automation and testing
- **FastAPI** for API development and deployment
- **OpenCV** for computer vision and image processing

**All AI-related prompts are specifically designed to use the free Gemini API (gemini-1.5-flash-latest) and can be run directly in the AI Model Evaluator app.**

### Usage

1. **Access Prompt Manager**: Use the "📝 Prompt Manager" tab in the Streamlit app
2. **Initialize Defaults**: Click "🔄 Initialize Default Prompts" to load default prompts
3. **Select Prompts**: Choose from system prompts or test prompt examples in the sidebar
4. **Customize**: Add, edit, or delete prompts as needed

### Testing the Prompt Manager

Run the test script to verify the prompt management system:

```bash
python test_gemini_models.py
```

## 📊 Automated Testing

### Usage

Run the automated test script to evaluate multiple models:

```bash
python test_gemini_models.py
```

### What it does

The automated test script:

1. **Tests multiple Gemini models** with the same prompt about web scraping
2. **Evaluates responses** based on multiple criteria:
   - Code quality (Python code presence)
   - Execution attempt (code execution)
   - Content summary (summary generation)
   - Error handling (try/catch blocks)
   - Documentation (comments/docstrings)
   - Response length (comprehensiveness)
   - Response time (speed)

3. **Generates two output files**:
   - `gemini_test_results_YYYYMMDD_HHMMSS.json` - Raw data
   - `gemini_evaluation_report_YYYYMMDD_HHMMSS.md` - Human-readable report

### Models Tested

The script dynamically fetches all available Gemini models that support content generation using `genai.list_models()`. This ensures you always have access to the latest models without needing to update the code.

Common models include:
- gemini-2.5-flash (default)
- gemini-2.5-flash-lite
- gemini-2.0-flash
- gemini-2.0-flash-lite
- gemini-1.5-pro-latest
- gemini-1.5-flash-latest

### Test Prompt

The script tests with this prompt:
> "Write a python function to scrape the data from the following url: https://sites.google.com/view/ai-powered-software-dev/startseite. Provide the code and also execute it and summarize the content of the site."

### Example Results

From the latest test:
- **Success Rate**: 5/6 models (83%)
- **Fastest**: gemini-1.5-flash-latest (5.58s)
- **Most Comprehensive**: gemini-2.5-flash (1278 words) - **Default Model**
- **Best Code Quality**: All successful models scored well

## 🎮 Interactive Testing with Streamlit

### Launch the App

**Option 1: Command Line**
```bash
streamlit run streamlit_app.py
```

**Option 2: VS Code (Recommended)**
1. Open the project in VS Code
2. Press `F5` or go to Run → Start Debugging
3. Select "Launch Streamlit App" from the dropdown
4. The app will open automatically in your browser

**VS Code Launch Configurations:**

The project includes a comprehensive `.vscode/launch.json` file with multiple debugging and testing configurations:

#### **🚀 Streamlit App Launch Options:**
- **Launch Streamlit App**: Standard launch with debugging
- **Launch Streamlit App (External)**: Uses streamlit module directly
- **Debug Streamlit App**: Full debugging with breakpoints

#### **🧪 Testing Options:**
- **Run Automated Tests**: Execute the automated test script
- **Test Prompt Manager**: Test prompt database functionality
- **Quick Test**: Run basic functionality tests

#### **🐛 Debugging Options:**
- **Debug Code Executor**: Debug code execution functionality
- **Debug Variable Inspection**: Test variable capture features

#### **⚡ Compound Configurations:**
- **Launch App + Tests**: Run both Streamlit app and automated tests simultaneously

**To use these configurations:**
1. Open the project in VS Code
2. Press `F5` or go to Run → Start Debugging
3. Select your desired configuration from the dropdown
4. The app will launch with full debugging capabilities

### Features

- **Dynamic Model Selection**: Automatically fetches all available Gemini models
- **Model Refresh**: Refresh button to update the model list
- **Custom Prompts**: Input your own prompts for testing
- **System Prompts**: Advanced prompting with system messages
- **Response Analysis**: Real-time evaluation of responses
- **Comparison Mode**: Test multiple models side-by-side
- **Export Results**: Save test results to files
- **Multi-Model Testing**: Test multiple models simultaneously
- **Dual Prompt System**: Use both system prompts and test prompt examples together
- **Code Execution**: Execute AI-generated Python code with real-time output and error handling

### Code Execution Feature

The AI Model Evaluator now includes a powerful code execution feature that allows you to:

- **Extract Code Blocks**: Automatically detect and extract Python code from AI responses
- **Execute Python Code**: Run AI-generated Python code safely in isolated environments
- **Real-time Results**: See execution output, errors, and performance metrics
- **Code Analysis**: Get insights about code structure, functions, and complexity
- **Timeout Protection**: Code execution is limited to prevent infinite loops
- **Error Handling**: Comprehensive error reporting and debugging information

#### How to Use Code Execution:

1. **Generate Code**: Use any of the test prompt examples that generate Python code
2. **Navigate to Code Execution Tab**: Click on the "🚀 Code Execution" tab
3. **Review Extracted Code**: The system automatically detects code blocks in the response
4. **Execute Code**: Click the "▶️ Execute" button for each code block
5. **View Results**: See real-time output, execution time, and any errors

#### Supported Code Types:

- **Python Code**: Full Python scripts, functions, and modules
- **Markdown Code Blocks**: Code wrapped in ```python blocks
- **Inline Functions**: Standalone function definitions
- **Import Statements**: Module imports and dependencies

#### Safety Features:

- **Isolated Execution**: Code runs in temporary files with cleanup
- **Timeout Limits**: 30-second execution timeout by default
- **Error Capture**: All stdout, stderr, and exceptions are captured
- **Resource Management**: Automatic cleanup of temporary files

#### Debug Features:

- **Syntax Checking**: Validate Python code syntax before execution
- **Dry Run Analysis**: Identify potential issues without execution
- **Variable Inspection**: Capture and inspect all variables after execution
  - View variable types, values, and sizes
  - Interactive variable selection and detailed inspection
  - Support for complex data structures (lists, dicts, objects)
  - Real-time variable capture during code execution

### System Prompt Suggestions

The app includes default system prompts for development scenarios:

1. **Code Producer (Default)**:
   ```
   You are a code producer focused on delivering working solutions quickly. 
   Provide clean, functional code with minimal explanation but good documentation.
   ```

2. **Python Development Expert**:
   ```
   You are a Python development expert. Provide practical, production-ready solutions 
   for Python applications. Focus on modern Python best practices, clean code, and real-world implementation.
   ```

3. **Code Review Agent**:
   ```
   You are an expert code reviewer. Provide detailed feedback on code quality, 
   best practices, security, and performance. Always suggest improvements.
   ```

4. **Debugging Assistant**:
   ```
   You are a debugging expert. Help identify and fix issues in code. 
   Provide step-by-step solutions and explain the root causes.
   ```

5. **Documentation Writer**:
   ```
   You are a technical writer. Create clear, comprehensive documentation 
   for code, APIs, and systems. Include examples and best practices.
   ```

## 📋 Evaluation Criteria

Each response is scored on:
- ✅ **Code Quality**: Contains functional Python code
- ✅ **Execution**: Attempts to execute the scraping
- ✅ **Summary**: Provides content summary
- ✅ **Error Handling**: Includes try/catch blocks
- ✅ **Documentation**: Well-documented code
- ✅ **Length**: Comprehensive response

## 🔍 Variable Inspection Feature

The debug section now includes advanced variable inspection capabilities that allow you to:

### **Variable Capture**
- Automatically capture all user-defined variables after code execution
- Filter out system variables and imported modules
- Support for all Python data types (int, float, str, bool, list, dict, tuple, etc.)

### **Interactive Inspection Interface**
- **Overview Tab**: See all variables with their types and sizes at a glance
- **Details Tab**: Select individual variables for detailed inspection
- **Raw Data Tab**: View complete variable data in JSON format

### **Variable Information Displayed**
- **Type**: Python data type (str, int, list, dict, etc.)
- **Value**: Current value of the variable
- **Size**: Length for collections, shape for arrays, or N/A for scalars
- **Formatted Display**: Type-specific formatting for better readability

### **Usage Example**
1. Write or paste Python code in the debug section
2. Click "🔍 Inspect Variables" button
3. View execution results and captured variables
4. Use the tabs to explore variable details:
   - **Overview**: Quick summary of all variables
   - **Details**: Select and inspect individual variables
   - **Raw Data**: Complete variable data in JSON format

### **Supported Data Types**
- **Primitive Types**: int, float, str, bool
- **Collections**: list, tuple, dict, set
- **Objects**: Custom objects with string representation
- **Special Types**: numpy arrays, pandas DataFrames (with shape info)




## 🔧 Development

### Adding New Dependencies

1. Install the package:
   ```bash
   pip install package_name
   ```

2. Update requirements.txt:
   ```bash
   pip freeze > requirements.txt
   ```

### Best Practices

- Keep dependencies up to date
- Use environment variables for sensitive data
- Test notebooks before committing changes
- Run automated tests regularly

## 🚨 Troubleshooting

### Common Issues

1. **Package installation errors:**
   - Upgrade pip: `python -m pip install --upgrade pip`
   - Clear pip cache: `pip cache purge`

2. **Jupyter not starting:**
   - Check if port 8888 is available
   - Ensure Anaconda is properly installed

3. **Import errors:**
   - Ensure you're using the base Anaconda environment
   - Run `conda activate base` if needed

4. **API rate limiting:**
   - Some models may hit rate limits
   - Wait and retry, or use different models

5. **Streamlit app issues:**
   - Ensure all dependencies are installed
   - Check if port 8501 is available
   - Verify API key is set correctly

## 📄 Output Files

### JSON Results
Contains raw data for each model:
- Full response text
- Response time
- Analysis scores
- Timestamps
- Error information

### Markdown Report
Human-readable report with:
- Summary table comparing all models
- Detailed analysis for each model
- Full response text
- Recommendations for best models

## 🔒 Security

- API keys are stored securely using Streamlit's built-in secrets management
- Local development uses `.streamlit/secrets.toml` (not tracked by git)
- Streamlit Cloud uses encrypted secrets stored in the cloud
- Sensitive data is excluded from version control
- Test results may contain API responses (review before sharing)
- Manual API key entry is available as a fallback option

## 📝 License

This project is for educational and testing purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the generated reports
3. Test with different models
4. Verify API key and permissions
