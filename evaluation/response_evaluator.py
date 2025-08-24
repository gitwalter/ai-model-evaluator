#!/usr/bin/env python3
"""
Response Evaluator for AI Model Evaluator.
Handles response analysis and evaluation based on multiple criteria.
"""

from typing import Dict, Any, List
import re
import ast
import json


class ResponseEvaluator:
    """Evaluates AI model responses based on multiple criteria."""
    
    def __init__(self):
        """Initialize the ResponseEvaluator with evaluation criteria and weights."""
        self.evaluation_criteria = {
            "code_quality": "Quality and functionality of provided code",
            "execution_attempt": "Attempts to execute or demonstrate code",
            "content_summary": "Provides meaningful summary or explanation",
            "error_handling": "Includes proper error handling mechanisms",
            "documentation": "Code documentation and comments quality",
            "response_length": "Comprehensive and detailed response",
            "code_complexity": "Sophistication of code implementation",
            "relevance": "Relevance to the given prompt",
            "clarity": "Clarity and readability of response"
        }
        
        # Define weights for each criterion (sum should be 1.0)
        self.criterion_weights = {
            "code_quality": 0.20,
            "execution_attempt": 0.15,
            "content_summary": 0.12,
            "error_handling": 0.10,
            "documentation": 0.08,
            "response_length": 0.10,
            "code_complexity": 0.12,
            "relevance": 0.08,
            "clarity": 0.05
        }
    
    def analyze_response(self, response_text: str) -> Dict[str, Any]:
        """Analyze the response based on evaluation criteria."""
        analysis = {}
        
        # Analyze each criterion with more sophisticated methods
        analysis["code_quality"] = self._analyze_code_quality(response_text)
        analysis["execution_attempt"] = self._analyze_execution_attempt(response_text)
        analysis["content_summary"] = self._analyze_content_summary(response_text)
        analysis["error_handling"] = self._analyze_error_handling(response_text)
        analysis["documentation"] = self._analyze_documentation(response_text)
        analysis["response_length"] = self._analyze_response_length(response_text)
        analysis["code_complexity"] = self._analyze_code_complexity(response_text)
        analysis["relevance"] = self._analyze_relevance(response_text)
        analysis["clarity"] = self._analyze_clarity(response_text)
        
        return analysis
    
    def _analyze_code_quality(self, response_text: str) -> Dict[str, Any]:
        """Analyze code quality with more nuanced scoring."""
        score = 0.0
        details = []
        
        # Check for code presence
        code_blocks = re.findall(r'```(?:(\w+)\n)?(.*?)```', response_text, re.DOTALL)
        if code_blocks:
            score += 0.3
            details.append("Code blocks found")
        
        # Check for Python-specific code
        python_indicators = ["def ", "import ", "class ", "if __name__", "return ", "print("]
        python_count = sum(1 for indicator in python_indicators if indicator in response_text)
        if python_count >= 3:
            score += 0.4
            details.append(f"Strong Python code indicators ({python_count})")
        elif python_count >= 1:
            score += 0.2
            details.append(f"Basic Python code indicators ({python_count})")
        
        # Check for syntax validity (basic check)
        try:
            # Extract Python code and check syntax
            python_code = self._extract_python_code(response_text)
            if python_code:
                ast.parse(python_code)
                score += 0.3
                details.append("Valid Python syntax")
        except:
            pass
        
        return {
            "score": min(score, 1.0),
            "details": "; ".join(details) if details else "No significant code found"
        }
    
    def _analyze_execution_attempt(self, response_text: str) -> Dict[str, Any]:
        """Analyze execution attempt with more detailed scoring."""
        score = 0.0
        details = []
        
        # Check for execution keywords
        execution_keywords = ["execute", "run", "result", "output", "print(", "console.log"]
        execution_count = sum(1 for keyword in execution_keywords if keyword in response_text.lower())
        
        if execution_count >= 3:
            score += 0.5
            details.append(f"Multiple execution indicators ({execution_count})")
        elif execution_count >= 1:
            score += 0.3
            details.append(f"Basic execution indicators ({execution_count})")
        
        # Check for actual execution examples
        if "example output" in response_text.lower() or "sample result" in response_text.lower():
            score += 0.3
            details.append("Execution examples provided")
        
        # Check for interactive elements
        if "input(" in response_text or "raw_input(" in response_text:
            score += 0.2
            details.append("Interactive execution elements")
        
        return {
            "score": min(score, 1.0),
            "details": "; ".join(details) if details else "No execution attempt found"
        }
    
    def _analyze_content_summary(self, response_text: str) -> Dict[str, Any]:
        """Analyze content summary with more nuanced scoring."""
        score = 0.0
        details = []
        
        # Check for summary keywords
        summary_keywords = ["summary", "overview", "description", "explanation", "analysis"]
        summary_count = sum(1 for keyword in summary_keywords if keyword in response_text.lower())
        
        if summary_count >= 2:
            score += 0.4
            details.append(f"Multiple summary indicators ({summary_count})")
        elif summary_count >= 1:
            score += 0.2
            details.append(f"Basic summary indicators ({summary_count})")
        
        # Check for structured content
        if any(marker in response_text for marker in ["##", "###", "**", "•", "- "]):
            score += 0.3
            details.append("Structured content format")
        
        # Check for detailed explanations
        sentences = response_text.split('.')
        long_sentences = [s for s in sentences if len(s.split()) > 10]
        if len(long_sentences) >= 3:
            score += 0.3
            details.append("Detailed explanations provided")
        
        return {
            "score": min(score, 1.0),
            "details": "; ".join(details) if details else "No content summary found"
        }
    
    def _analyze_error_handling(self, response_text: str) -> Dict[str, Any]:
        """Analyze error handling with more sophisticated scoring."""
        score = 0.0
        details = []
        
        # Check for error handling keywords
        error_keywords = ["try:", "except", "error", "exception", "catch", "finally", "raise"]
        error_count = sum(1 for keyword in error_keywords if keyword in response_text.lower())
        
        if error_count >= 3:
            score += 0.5
            details.append(f"Comprehensive error handling ({error_count} indicators)")
        elif error_count >= 1:
            score += 0.3
            details.append(f"Basic error handling ({error_count} indicators)")
        
        # Check for specific error types
        specific_errors = ["ValueError", "TypeError", "FileNotFoundError", "ImportError"]
        specific_count = sum(1 for error in specific_errors if error in response_text)
        if specific_count >= 1:
            score += 0.3
            details.append(f"Specific error types handled ({specific_count})")
        
        # Check for error messages
        if "error message" in response_text.lower() or "exception handling" in response_text.lower():
            score += 0.2
            details.append("Error message handling")
        
        return {
            "score": min(score, 1.0),
            "details": "; ".join(details) if details else "No error handling found"
        }
    
    def _analyze_documentation(self, response_text: str) -> Dict[str, Any]:
        """Analyze documentation with more detailed scoring."""
        score = 0.0
        details = []
        
        # Check for documentation keywords
        doc_keywords = ["#", "comment", "docstring", "function", "//", "/*", "*/", "'''", '"""']
        doc_count = sum(1 for keyword in doc_keywords if keyword in response_text)
        
        if doc_count >= 4:
            score += 0.4
            details.append(f"Extensive documentation ({doc_count} indicators)")
        elif doc_count >= 2:
            score += 0.3
            details.append(f"Good documentation ({doc_count} indicators)")
        elif doc_count >= 1:
            score += 0.1
            details.append(f"Basic documentation ({doc_count} indicators)")
        
        # Check for function documentation
        if '"""' in response_text or "'''" in response_text:
            score += 0.3
            details.append("Function docstrings present")
        
        # Check for inline comments
        comment_lines = [line for line in response_text.split('\n') if line.strip().startswith('#')]
        if len(comment_lines) >= 2:
            score += 0.3
            details.append(f"Multiple inline comments ({len(comment_lines)})")
        
        return {
            "score": min(score, 1.0),
            "details": "; ".join(details) if details else "No documentation found"
        }
    
    def _analyze_response_length(self, response_text: str) -> Dict[str, Any]:
        """Analyze response length with more nuanced scoring."""
        word_count = len(response_text.split())
        char_count = len(response_text)
        
        # Score based on word count with diminishing returns
        if word_count >= 500:
            score = 1.0
            details = f"Very comprehensive ({word_count} words)"
        elif word_count >= 300:
            score = 0.8
            details = f"Comprehensive ({word_count} words)"
        elif word_count >= 200:
            score = 0.6
            details = f"Detailed ({word_count} words)"
        elif word_count >= 100:
            score = 0.4
            details = f"Moderate ({word_count} words)"
        elif word_count >= 50:
            score = 0.2
            details = f"Basic ({word_count} words)"
        else:
            score = 0.0
            details = f"Too brief ({word_count} words)"
        
        return {
            "score": score,
            "details": details
        }
    
    def _analyze_code_complexity(self, response_text: str) -> Dict[str, Any]:
        """Analyze code complexity and sophistication."""
        score = 0.0
        details = []
        
        # Extract Python code
        python_code = self._extract_python_code(response_text)
        if not python_code:
            return {
                "score": 0.0,
                "details": "No Python code found"
            }
        
        # Check for advanced Python features
        advanced_features = {
            "classes": ["class ", "def __init__", "self."],
            "decorators": ["@", "decorator"],
            "context managers": ["with ", "contextmanager"],
            "generators": ["yield ", "generator"],
            "lambda functions": ["lambda "],
            "list comprehensions": ["for ", "in ", "["],
            "exception handling": ["try:", "except", "finally"],
            "imports": ["import ", "from ", "as "]
        }
        
        feature_count = 0
        for feature_name, indicators in advanced_features.items():
            if any(indicator in python_code for indicator in indicators):
                feature_count += 1
                details.append(feature_name)
        
        if feature_count >= 6:
            score = 1.0
            details = f"Very complex code ({feature_count} advanced features)"
        elif feature_count >= 4:
            score = 0.8
            details = f"Complex code ({feature_count} advanced features)"
        elif feature_count >= 2:
            score = 0.5
            details = f"Moderate complexity ({feature_count} advanced features)"
        elif feature_count >= 1:
            score = 0.2
            details = f"Basic complexity ({feature_count} advanced features)"
        else:
            score = 0.0
            details = "Simple code"
        
        return {
            "score": score,
            "details": details
        }
    
    def _analyze_relevance(self, response_text: str) -> Dict[str, Any]:
        """Analyze relevance to the prompt."""
        # This is a simplified version - in practice, you'd compare against the original prompt
        score = 0.0
        details = []
        
        # Check for technical terms that suggest relevance
        technical_terms = ["python", "code", "function", "script", "program", "algorithm", "data", "api"]
        term_count = sum(1 for term in technical_terms if term in response_text.lower())
        
        if term_count >= 4:
            score = 0.8
            details.append(f"Highly relevant ({term_count} technical terms)")
        elif term_count >= 2:
            score = 0.5
            details.append(f"Moderately relevant ({term_count} technical terms)")
        elif term_count >= 1:
            score = 0.2
            details.append(f"Somewhat relevant ({term_count} technical terms)")
        else:
            score = 0.0
            details.append("Low relevance")
        
        return {
            "score": score,
            "details": "; ".join(details)
        }
    
    def _analyze_clarity(self, response_text: str) -> Dict[str, Any]:
        """Analyze clarity and readability of the response."""
        score = 0.0
        details = []
        
        # Check for clear structure
        if any(marker in response_text for marker in ["##", "###", "**", "•", "- ", "1.", "2."]):
            score += 0.3
            details.append("Clear structure with formatting")
        
        # Check for code blocks (improves clarity)
        code_blocks = re.findall(r'```.*?```', response_text, re.DOTALL)
        if len(code_blocks) >= 2:
            score += 0.3
            details.append(f"Multiple code blocks ({len(code_blocks)})")
        elif len(code_blocks) >= 1:
            score += 0.2
            details.append("Code blocks present")
        
        # Check for explanations
        explanation_indicators = ["explains", "shows", "demonstrates", "illustrates", "example"]
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in response_text.lower())
        if explanation_count >= 2:
            score += 0.4
            details.append(f"Clear explanations ({explanation_count} indicators)")
        
        return {
            "score": min(score, 1.0),
            "details": "; ".join(details) if details else "Basic clarity"
        }
    
    def _extract_python_code(self, response_text: str) -> str:
        """Extract Python code from response text."""
        python_code = ""
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:python\n)?(.*?)```', response_text, re.DOTALL)
        python_code += "\n".join(code_blocks)
        
        # Extract inline code
        inline_code = re.findall(r'`([^`]+)`', response_text)
        python_code += "\n".join(inline_code)
        
        return python_code
    
    def calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate weighted overall score from analysis results."""
        weighted_score = 0.0
        
        for criterion, weight in self.criterion_weights.items():
            if criterion in analysis:
                score = analysis[criterion].get("score", 0)
                weighted_score += score * weight
        
        return min(weighted_score, 1.0)
    
    def get_score_breakdown(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed score breakdown for display."""
        breakdown = []
        for criterion, data in analysis.items():
            if criterion in self.criterion_weights:
                weight = self.criterion_weights[criterion]
                score = data.get("score", 0)
                weighted_score = score * weight
                
                breakdown.append({
                    "criterion": criterion.replace('_', ' ').title(),
                    "score": score,
                    "weighted_score": weighted_score,
                    "weight": weight,
                    "details": data.get("details", ""),
                    "status": "✅" if score >= 0.7 else "🟡" if score >= 0.4 else "❌"
                })
        
        # Sort by weighted score (most important first)
        breakdown.sort(key=lambda x: x["weighted_score"], reverse=True)
        return breakdown
    
    def get_quality_indicator(self, overall_score: float) -> Dict[str, str]:
        """Get quality indicator based on overall score."""
        if overall_score >= 0.85:
            return {"color": "🟢", "text": "Excellent"}
        elif overall_score >= 0.70:
            return {"color": "🟢", "text": "Very Good"}
        elif overall_score >= 0.55:
            return {"color": "🟡", "text": "Good"}
        elif overall_score >= 0.40:
            return {"color": "🟠", "text": "Fair"}
        elif overall_score >= 0.25:
            return {"color": "🔴", "text": "Poor"}
        else:
            return {"color": "🔴", "text": "Very Poor"}
