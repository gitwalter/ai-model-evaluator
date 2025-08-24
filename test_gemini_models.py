#!/usr/bin/env python3
"""
Test script to evaluate multiple Gemini models with the same prompt.
Generates responses and evaluation for each model.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import textwrap

from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=api_key)

# Test prompt
TEST_PROMPT = """Write a python function to scrape the data from the following url: https://sites.google.com/view/ai-powered-software-dev/startseite. Provide the code and also execute it and summarize the content of the site."""

def get_available_models():
    """Get list of available Gemini models that support content generation."""
    try:
        models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models.append(model.name)
        return sorted(models)
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        # Fallback to common models if API call fails
        return [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest"
        ]

# Get models to test dynamically
MODELS_TO_TEST = get_available_models()

class ModelEvaluator:
    """Evaluates Gemini models and generates reports."""
    
    def __init__(self):
        self.results = {}
        self.evaluation_criteria = {
            "code_quality": "Does the response include functional Python code?",
            "execution_attempt": "Does the code attempt to execute the scraping?",
            "content_summary": "Does it provide a summary of the scraped content?",
            "error_handling": "Does the code include error handling?",
            "documentation": "Is the code well-documented?",
            "response_length": "Is the response comprehensive?",
            "response_time": "How fast was the response?"
        }
    
    def test_model(self, model_name: str) -> Dict[str, Any]:
        """Test a single model and return results."""
        print(f"Testing model: {model_name}")
        
        try:
            # Create model instance
            model = genai.GenerativeModel(model_name)
            
            # Measure response time
            start_time = time.time()
            response = model.generate_content(TEST_PROMPT)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Extract response text
            response_text = response.text if response.text else "No response text"
            
            # Analyze response
            analysis = self._analyze_response(response_text)
            
            result = {
                "model_name": model_name,
                "prompt": TEST_PROMPT,
                "response": response_text,
                "response_time": response_time,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            print(f"✅ {model_name}: {response_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ {model_name}: Error - {str(e)}")
            return {
                "model_name": model_name,
                "prompt": TEST_PROMPT,
                "response": f"Error: {str(e)}",
                "response_time": 0,
                "analysis": {},
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_response(self, response_text: str) -> Dict[str, Any]:
        """Analyze the response based on evaluation criteria."""
        analysis = {}
        
        # Check for Python code
        analysis["code_quality"] = {
            "score": 0,
            "details": "No Python code found"
        }
        if "def " in response_text or "import " in response_text:
            analysis["code_quality"] = {
                "score": 1,
                "details": "Python code detected"
            }
        
        # Check for execution attempt
        analysis["execution_attempt"] = {
            "score": 0,
            "details": "No execution attempt found"
        }
        if any(keyword in response_text.lower() for keyword in ["execute", "run", "result", "output"]):
            analysis["execution_attempt"] = {
                "score": 1,
                "details": "Execution attempt detected"
            }
        
        # Check for content summary
        analysis["content_summary"] = {
            "score": 0,
            "details": "No content summary found"
        }
        if any(keyword in response_text.lower() for keyword in ["summary", "content", "site contains", "information"]):
            analysis["content_summary"] = {
                "score": 1,
                "details": "Content summary detected"
            }
        
        # Check for error handling
        analysis["error_handling"] = {
            "score": 0,
            "details": "No error handling found"
        }
        if any(keyword in response_text.lower() for keyword in ["try:", "except", "error", "exception"]):
            analysis["error_handling"] = {
                "score": 1,
                "details": "Error handling detected"
            }
        
        # Check for documentation
        analysis["documentation"] = {
            "score": 0,
            "details": "No documentation found"
        }
        if any(keyword in response_text.lower() for keyword in ["#", "comment", "docstring", "function"]):
            analysis["documentation"] = {
                "score": 1,
                "details": "Documentation detected"
            }
        
        # Check response length
        word_count = len(response_text.split())
        analysis["response_length"] = {
            "score": min(word_count / 100, 1),  # Normalize to 0-1
            "details": f"{word_count} words"
        }
        
        return analysis
    
    def run_tests(self) -> Dict[str, Any]:
        """Run tests on all models."""
        print("🚀 Starting Gemini model evaluation...")
        print(f"Testing {len(MODELS_TO_TEST)} models with prompt:")
        print(f"'{TEST_PROMPT[:100]}...'")
        print(f"Available models: {', '.join(MODELS_TO_TEST)}")
        print("-" * 80)
        
        for model_name in MODELS_TO_TEST:
            result = self.test_model(model_name)
            self.results[model_name] = result
            time.sleep(1)  # Rate limiting
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return "No results to report"
        
        report = []
        report.append("# Gemini Model Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("")
        report.append("| Model | Status | Response Time | Code Quality | Execution | Summary | Error Handling | Documentation | Length |")
        report.append("|-------|--------|---------------|--------------|-----------|---------|----------------|---------------|--------|")
        
        for model_name, result in self.results.items():
            analysis = result.get("analysis", {})
            status = result.get("status", "unknown")
            response_time = result.get("response_time", 0)
            
            row = [
                model_name,
                status,
                f"{response_time:.2f}s",
                "✅" if analysis.get("code_quality", {}).get("score", 0) > 0 else "❌",
                "✅" if analysis.get("execution_attempt", {}).get("score", 0) > 0 else "❌",
                "✅" if analysis.get("content_summary", {}).get("score", 0) > 0 else "❌",
                "✅" if analysis.get("error_handling", {}).get("score", 0) > 0 else "❌",
                "✅" if analysis.get("documentation", {}).get("score", 0) > 0 else "❌",
                analysis.get("response_length", {}).get("details", "0 words")
            ]
            report.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        report.append("")
        
        # Detailed analysis
        report.append("## Detailed Analysis")
        report.append("")
        
        for model_name, result in self.results.items():
            report.append(f"### {model_name}")
            report.append("")
            
            if result.get("status") == "error":
                report.append(f"**Error:** {result.get('error', 'Unknown error')}")
                report.append("")
                continue
            
            analysis = result.get("analysis", {})
            response_time = result.get("response_time", 0)
            
            report.append(f"**Response Time:** {response_time:.2f} seconds")
            report.append("")
            
            for criterion, data in analysis.items():
                score = data.get("score", 0)
                details = data.get("details", "")
                status = "✅" if score > 0 else "❌"
                report.append(f"- **{criterion.replace('_', ' ').title()}:** {status} {details}")
            
            report.append("")
            report.append("**Full Response:**")
            report.append("```")
            report.append(result.get("response", "No response"))
            report.append("```")
            report.append("")
            report.append("---")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        # Find best performing models
        successful_results = {k: v for k, v in self.results.items() if v.get("status") == "success"}
        
        if successful_results:
            # Best response time
            fastest = min(successful_results.items(), key=lambda x: x[1].get("response_time", float('inf')))
            report.append(f"- **Fastest Model:** {fastest[0]} ({fastest[1].get('response_time', 0):.2f}s)")
            
            # Most comprehensive
            most_comprehensive = max(successful_results.items(), 
                                   key=lambda x: x[1].get("analysis", {}).get("response_length", {}).get("score", 0))
            report.append(f"- **Most Comprehensive:** {most_comprehensive[0]} ({most_comprehensive[1].get('analysis', {}).get('response_length', {}).get('details', '0 words')})")
            
            # Best code quality
            best_code = max(successful_results.items(),
                          key=lambda x: sum(x[1].get("analysis", {}).get(criterion, {}).get("score", 0) 
                                          for criterion in ["code_quality", "execution_attempt", "error_handling", "documentation"]))
            report.append(f"- **Best Code Quality:** {best_code[0]}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Results saved to: {filename}")
        return filename
    
    def save_report(self, filename: str = None):
        """Save evaluation report to markdown file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_evaluation_report_{timestamp}.md"
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {filename}")
        return filename

def main():
    """Main function to run the evaluation."""
    print("🔬 AI Model Evaluation Tool")
    print("=" * 50)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Run tests
    results = evaluator.run_tests()
    
    # Generate and save reports
    json_file = evaluator.save_results()
    report_file = evaluator.save_report()
    
    print("\n" + "=" * 50)
    print("✅ Evaluation complete!")
    print(f"📁 Results: {json_file}")
    print(f"📄 Report: {report_file}")
    
    # Print quick summary
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"📊 Success rate: {successful}/{len(results)} models")

if __name__ == "__main__":
    main()
