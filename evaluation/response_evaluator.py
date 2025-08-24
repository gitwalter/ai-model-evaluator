#!/usr/bin/env python3
"""
Response Evaluator for AI Model Evaluator.
Handles response analysis and evaluation based on multiple criteria.
"""

from typing import Dict, Any, List
import re


class ResponseEvaluator:
    """Evaluates AI model responses based on multiple criteria."""
    
    def __init__(self):
        """Initialize the ResponseEvaluator with evaluation criteria."""
        self.evaluation_criteria = {
            "code_quality": "Does the response include functional code?",
            "execution_attempt": "Does the code attempt to execute something?",
            "content_summary": "Does it provide a summary or explanation?",
            "error_handling": "Does the code include error handling?",
            "documentation": "Is the code well-documented?",
            "response_length": "Is the response comprehensive?",
            "response_time": "How fast was the response?"
        }
    
    def analyze_response(self, response_text: str) -> Dict[str, Any]:
        """Analyze the response based on evaluation criteria."""
        analysis = {}
        
        # Check for code (Python, JavaScript, etc.)
        analysis["code_quality"] = self._analyze_code_quality(response_text)
        
        # Check for execution attempt
        analysis["execution_attempt"] = self._analyze_execution_attempt(response_text)
        
        # Check for content summary
        analysis["content_summary"] = self._analyze_content_summary(response_text)
        
        # Check for error handling
        analysis["error_handling"] = self._analyze_error_handling(response_text)
        
        # Check for documentation
        analysis["documentation"] = self._analyze_documentation(response_text)
        
        # Check response length
        analysis["response_length"] = self._analyze_response_length(response_text)
        
        return analysis
    
    def _analyze_code_quality(self, response_text: str) -> Dict[str, Any]:
        """Analyze code quality in the response."""
        code_indicators = ["def ", "import ", "function ", "class ", "const ", "let ", "var "]
        if any(indicator in response_text for indicator in code_indicators):
            return {
                "score": 1,
                "details": "Code detected"
            }
        return {
            "score": 0,
            "details": "No code found"
        }
    
    def _analyze_execution_attempt(self, response_text: str) -> Dict[str, Any]:
        """Analyze execution attempt in the response."""
        execution_keywords = ["execute", "run", "result", "output", "print", "console.log"]
        if any(keyword in response_text.lower() for keyword in execution_keywords):
            return {
                "score": 1,
                "details": "Execution attempt detected"
            }
        return {
            "score": 0,
            "details": "No execution attempt found"
        }
    
    def _analyze_content_summary(self, response_text: str) -> Dict[str, Any]:
        """Analyze content summary in the response."""
        summary_keywords = ["summary", "content", "contains", "information", "overview", "description"]
        if any(keyword in response_text.lower() for keyword in summary_keywords):
            return {
                "score": 1,
                "details": "Content summary detected"
            }
        return {
            "score": 0,
            "details": "No content summary found"
        }
    
    def _analyze_error_handling(self, response_text: str) -> Dict[str, Any]:
        """Analyze error handling in the response."""
        error_keywords = ["try:", "except", "error", "exception", "catch", "finally"]
        if any(keyword in response_text.lower() for keyword in error_keywords):
            return {
                "score": 1,
                "details": "Error handling detected"
            }
        return {
            "score": 0,
            "details": "No error handling found"
        }
    
    def _analyze_documentation(self, response_text: str) -> Dict[str, Any]:
        """Analyze documentation in the response."""
        doc_keywords = ["#", "comment", "docstring", "function", "//", "/*", "*/"]
        if any(keyword in response_text.lower() for keyword in doc_keywords):
            return {
                "score": 1,
                "details": "Documentation detected"
            }
        return {
            "score": 0,
            "details": "No documentation found"
        }
    
    def _analyze_response_length(self, response_text: str) -> Dict[str, Any]:
        """Analyze response length."""
        word_count = len(response_text.split())
        return {
            "score": min(word_count / 50, 1),  # Normalize to 0-1
            "details": f"{word_count} words"
        }
    
    def calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall score from analysis results."""
        scores = [
            analysis["code_quality"]["score"],
            analysis["execution_attempt"]["score"],
            analysis["content_summary"]["score"],
            analysis["error_handling"]["score"],
            analysis["documentation"]["score"],
            analysis["response_length"]["score"]
        ]
        return sum(scores) / len(scores)
    
    def get_score_breakdown(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed score breakdown for display."""
        breakdown = []
        for criterion, data in analysis.items():
            if criterion != "response_time":  # Skip response_time as it's not a score
                breakdown.append({
                    "criterion": criterion.replace('_', ' ').title(),
                    "score": data.get("score", 0),
                    "details": data.get("details", ""),
                    "status": "✅" if data.get("score", 0) > 0 else "❌"
                })
        return breakdown
    
    def get_quality_indicator(self, overall_score: float) -> Dict[str, str]:
        """Get quality indicator based on overall score."""
        if overall_score >= 0.8:
            return {"color": "🟢", "text": "Excellent"}
        elif overall_score >= 0.6:
            return {"color": "🟡", "text": "Good"}
        elif overall_score >= 0.4:
            return {"color": "🟠", "text": "Fair"}
        else:
            return {"color": "��", "text": "Poor"}
