
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import subprocess
import tempfile
import os
from .ch02 import generate_text_basic_cache
from .qwen3 import Qwen3Tokenizer
from .domain_experts import DomainExpertSystem


class PromptManager:
    """Manages different prompt templates for various reasoning tasks."""
    
    def __init__(self):
        self.templates = {
            "cot": "Let's think step by step.\n\nQuestion: {question}\n\nAnswer:",
            "pal": "Generate Python code to solve this problem step by step.\n\nQuestion: {question}\n\nPython code:\n```python\n",
            "tot": "Consider multiple approaches to solve this problem.\n\nQuestion: {question}\n\nApproach {approach_num}:",
            "verify": "Rate the correctness of this reasoning on a scale of 1-10.\n\nQuestion: {question}\nReasoning: {reasoning}\n\nConfidence score:"
        }
    
    def get_prompt(self, template_type: str, **kwargs) -> str:
        return self.templates[template_type].format(**kwargs)


class PALExecutor:
    """Program-Aided Language executor for running generated Python code safely."""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def run_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment."""
        try:
            # Clean the code
            code = self._clean_code(code)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                fname = f.name
            
            try:
                result = subprocess.check_output(
                    ["python3", fname], 
                    timeout=self.timeout,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                return {"success": True, "result": result.strip(), "error": None}
            except subprocess.TimeoutExpired:
                return {"success": False, "result": None, "error": "Execution timeout"}
            except subprocess.CalledProcessError as e:
                return {"success": False, "result": None, "error": str(e.output)}
            finally:
                os.unlink(fname)
                
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}
    
    def _clean_code(self, code: str) -> str:
        """Clean and prepare code for execution."""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*$', '', code)
        
        # Add print statement for the final result if not present
        lines = code.strip().split('\n')
        if lines and not any('print(' in line for line in lines[-3:]):
            # Find the last assignment or expression
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if '=' in line and not line.startswith('#'):
                    var_name = line.split('=')[0].strip()
                    lines.append(f'print({var_name})')
                    break
        
        return '\n'.join(lines)


class SelfConsistencyModule:
    """Implements self-consistency by sampling multiple reasoning chains."""
    
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
    
    def generate_multiple_chains(self, model, tokenizer, prompt: str, max_new_tokens: int = 256) -> List[str]:
        """Generate multiple reasoning chains with different sampling."""
        chains = []
        
        for i in range(self.num_samples):
            # Add slight variations to encourage diversity
            varied_prompt = f"{prompt}\n\nReasoning path {i+1}:"
            
            # Tokenize
            token_ids = tokenizer.encode(varied_prompt).unsqueeze(0)
            
            # Generate with sampling (add temperature if model supports it)
            generated_ids = generate_text_basic_cache(
                model, token_ids, max_new_tokens, eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(generated_ids[0])
            chains.append(generated_text)
        
        return chains
    
    def majority_vote(self, chains: List[str]) -> str:
        """Select the most common final answer from multiple chains."""
        # Extract final answers (simple heuristic: last number or short phrase)
        answers = []
        for chain in chains:
            answer = self._extract_final_answer(chain)
            if answer:
                answers.append(answer)
        
        if not answers:
            return chains[0] if chains else ""
        
        # Count occurrences
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0][0]
        
        # Return the full chain that contains the most common answer
        for chain in chains:
            if most_common in chain:
                return chain
        
        return chains[0]
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from reasoning text."""
        # Look for common answer patterns
        patterns = [
            r"(?:answer is|answer:|final answer:)\s*([^\n.]+)",
            r"(?:therefore|thus|so),?\s*([^\n.]+)",
            r"(\d+(?:\.\d+)?)",  # Numbers
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[-1].strip()
        
        return None


class TreeOfThoughts:
    """Implements Tree-of-Thoughts reasoning with breadth-first search."""
    
    def __init__(self, max_depth: int = 3, beam_width: int = 3):
        self.max_depth = max_depth
        self.beam_width = beam_width
    
    def search(self, model, tokenizer, verifier, question: str) -> Dict[str, Any]:
        """Perform ToT search to find the best reasoning path."""
        prompt_manager = PromptManager()
        
        # Initialize with multiple starting approaches
        current_nodes = []
        for i in range(self.beam_width):
            initial_prompt = prompt_manager.get_prompt("tot", question=question, approach_num=i+1)
            token_ids = tokenizer.encode(initial_prompt).unsqueeze(0)
            
            generated_ids = generate_text_basic_cache(
                model, token_ids, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id
            )
            
            reasoning = tokenizer.decode(generated_ids[0])
            score = verifier.score_reasoning(question, reasoning)
            
            current_nodes.append({
                "reasoning": reasoning,
                "score": score,
                "depth": 0,
                "parent": None
            })
        
        best_path = None
        best_score = -1
        
        # Iterative deepening
        for depth in range(self.max_depth):
            next_nodes = []
            
            # Expand top nodes
            top_nodes = sorted(current_nodes, key=lambda x: x["score"], reverse=True)[:self.beam_width]
            
            for node in top_nodes:
                if node["score"] > best_score:
                    best_score = node["score"]
                    best_path = node
                
                # Generate continuations
                continuations = self._generate_continuations(
                    model, tokenizer, node["reasoning"], question
                )
                
                for continuation in continuations:
                    score = verifier.score_reasoning(question, continuation)
                    next_nodes.append({
                        "reasoning": continuation,
                        "score": score,
                        "depth": depth + 1,
                        "parent": node
                    })
            
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        return {
            "best_reasoning": best_path["reasoning"] if best_path else "",
            "best_score": best_score,
            "search_tree": self._build_tree_trace(best_path)
        }
    
    def _generate_continuations(self, model, tokenizer, current_reasoning: str, question: str) -> List[str]:
        """Generate possible continuations of current reasoning."""
        continuations = []
        
        continuation_prompts = [
            f"{current_reasoning}\n\nNext step:",
            f"{current_reasoning}\n\nAlternatively:",
            f"{current_reasoning}\n\nLet me verify this:"
        ]
        
        for prompt in continuation_prompts:
            token_ids = tokenizer.encode(prompt).unsqueeze(0)
            generated_ids = generate_text_basic_cache(
                model, token_ids, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id
            )
            continuation = tokenizer.decode(generated_ids[0])
            continuations.append(continuation)
        
        return continuations
    
    def _build_tree_trace(self, node) -> List[str]:
        """Build the trace of the best path through the tree."""
        trace = []
        current = node
        while current:
            trace.append(current["reasoning"])
            current = current.get("parent")
        return list(reversed(trace))


class ReasoningVerifier:
    """Lightweight verifier to score reasoning quality."""
    
    def __init__(self):
        # For now, use heuristic scoring
        # In a full implementation, this would be a fine-tuned model
        pass
    
    def score_reasoning(self, question: str, reasoning: str) -> float:
        """Score reasoning quality on a scale of 0-1."""
        score = 0.5  # Base score
        
        # Heuristic checks
        if len(reasoning.strip()) > 50:  # Non-trivial length
            score += 0.1
        
        if "step" in reasoning.lower():  # Step-by-step reasoning
            score += 0.1
        
        if any(word in reasoning.lower() for word in ["because", "therefore", "since", "thus"]):
            score += 0.1  # Causal reasoning
        
        if reasoning.count('\n') >= 2:  # Multi-line reasoning
            score += 0.1
        
        # Check for mathematical expressions if it's a math problem
        if re.search(r'\d+', question) and re.search(r'[\d+\-*/=()]', reasoning):
            score += 0.1
        
        return min(score, 1.0)


class ReasoningEngine:
    """Main reasoning engine coordinating all components."""
    
    def __init__(self, model, tokenizer, memory_layer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_layer = memory_layer
        self.prompt_manager = PromptManager()
        self.pal_executor = PALExecutor()
        self.self_consistency = SelfConsistencyModule()
        self.tree_of_thoughts = TreeOfThoughts()
        self.verifier = ReasoningVerifier()
        self.domain_experts = DomainExpertSystem()
        
        # Performance tracking
        self.method_usage_stats = {
            "cot": 0, "pal": 0, "self_consistency": 0, "tot": 0
        }
    
    def solve_query(self, question: str, method: str = "auto") -> Dict[str, Any]:
        """Solve a query using the specified reasoning method."""
        if method == "auto":
            method = self._detect_best_method(question)
        
        # Track method usage
        if method in self.method_usage_stats:
            self.method_usage_stats[method] += 1
        
        # Check for domain expertise
        domain_prompt, domain_name = self.domain_experts.get_domain_reasoning_prompt(question)
        
        # Solve using the selected method with domain enhancement
        if method == "pal":
            result = self._solve_with_pal(question, domain_prompt)
        elif method == "tot":
            result = self._solve_with_tot(question, domain_prompt)
        elif method == "self_consistency":
            result = self._solve_with_self_consistency(question, domain_prompt)
        else:  # Default to chain-of-thought
            result = self._solve_with_cot(question, domain_prompt)
        
        # Add domain information to result
        if domain_name:
            result["domain"] = domain_name
            result["domain_enhanced"] = True
            
            # Validate reasoning using domain expertise
            validation = self.domain_experts.validate_domain_reasoning(
                question, result["reasoning"], result["answer"]
            )
            result["domain_validation"] = validation
            result["compliance_score"] = validation.get("compliance_score", 0.7)
        else:
            result["domain"] = "general"
            result["domain_enhanced"] = False
            result["compliance_score"] = 0.7
        
        # Track performance if memory layer is available
        if self.memory_layer:
            try:
                # Consider high confidence (>0.7) as success
                success = result["confidence"] > 0.7
                self.memory_layer.track_performance(
                    method, question, success, result["confidence"]
                )
            except:
                pass  # Don't fail if tracking fails
        
        return result
    
    def _detect_best_method(self, question: str) -> str:
        """Automatically detect the best reasoning method for the question."""
        # If we have a memory layer with performance tracking, use it
        if hasattr(self, 'memory_layer') and self.memory_layer:
            try:
                adaptive_method = self.memory_layer.get_best_method_for_question(question)
                if adaptive_method != "auto":
                    return adaptive_method
            except:
                pass  # Fall back to heuristic method
        
        question_lower = question.lower()
        
        # Use PAL for mathematical/computational problems
        if any(word in question_lower for word in ["calculate", "compute", "math", "equation", "solve"]):
            if re.search(r'\d+', question):
                return "pal"
        
        # Use ToT for complex planning problems
        if any(word in question_lower for word in ["plan", "strategy", "design", "approach"]):
            return "tot"
        
        # Use self-consistency for factual questions
        if any(word in question_lower for word in ["what", "who", "when", "where", "which"]):
            return "self_consistency"
        
        return "cot"  # Default
    
    def _solve_with_cot(self, question: str, domain_prompt: str = "") -> Dict[str, Any]:
        """Solve using chain-of-thought reasoning."""
        if domain_prompt:
            prompt = domain_prompt
        else:
            prompt = self.prompt_manager.get_prompt("cot", question=question)
        
        token_ids = self.tokenizer.encode(prompt).unsqueeze(0)
        
        generated_ids = generate_text_basic_cache(
            self.model, token_ids, max_new_tokens=256, eos_token_id=self.tokenizer.eos_token_id
        )
        
        reasoning = self.tokenizer.decode(generated_ids[0])
        confidence = self.verifier.score_reasoning(question, reasoning)
        
        return {
            "answer": self._extract_final_answer(reasoning),
            "reasoning": reasoning,
            "method": "chain_of_thought",
            "confidence": confidence,
            "trace": [reasoning]
        }
    
    def _solve_with_pal(self, question: str, domain_prompt: str = "") -> Dict[str, Any]:
        """Solve using Program-Aided Language reasoning."""
        if domain_prompt and "code" in domain_prompt.lower():
            prompt = domain_prompt
        else:
            prompt = self.prompt_manager.get_prompt("pal", question=question)
        
        token_ids = self.tokenizer.encode(prompt).unsqueeze(0)
        
        generated_ids = generate_text_basic_cache(
            self.model, token_ids, max_new_tokens=256, eos_token_id=self.tokenizer.eos_token_id
        )
        
        code = self.tokenizer.decode(generated_ids[0])
        execution_result = self.pal_executor.run_python(code)
        
        reasoning = f"Generated code:\n{code}\n\nExecution result:\n{execution_result}"
        confidence = 0.9 if execution_result["success"] else 0.3
        
        answer = execution_result["result"] if execution_result["success"] else "Error in execution"
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "method": "program_aided_language",
            "confidence": confidence,
            "trace": [code, str(execution_result)]
        }
    
    def _solve_with_self_consistency(self, question: str, domain_prompt: str = "") -> Dict[str, Any]:
        """Solve using self-consistency with multiple reasoning chains."""
        if domain_prompt:
            prompt = domain_prompt
        else:
            prompt = self.prompt_manager.get_prompt("cot", question=question)
        
        chains = self.self_consistency.generate_multiple_chains(
            self.model, self.tokenizer, prompt
        )
        
        best_reasoning = self.self_consistency.majority_vote(chains)
        confidence = self.verifier.score_reasoning(question, best_reasoning)
        
        return {
            "answer": self._extract_final_answer(best_reasoning),
            "reasoning": best_reasoning,
            "method": "self_consistency",
            "confidence": confidence,
            "trace": chains
        }
    
    def _solve_with_tot(self, question: str, domain_prompt: str = "") -> Dict[str, Any]:
        """Solve using Tree-of-Thoughts reasoning."""
        # For ToT, we'll enhance the question with domain context if available
        enhanced_question = question
        if domain_prompt:
            enhanced_question = f"{domain_prompt}\n\nOriginal question: {question}"
        
        result = self.tree_of_thoughts.search(
            self.model, self.tokenizer, self.verifier, enhanced_question
        )
        
        return {
            "answer": self._extract_final_answer(result["best_reasoning"]),
            "reasoning": result["best_reasoning"],
            "method": "tree_of_thoughts",
            "confidence": result["best_score"],
            "trace": result["search_tree"]
        }
    
    def _extract_final_answer(self, reasoning: str) -> str:
        """Extract the final answer from reasoning text."""
        # Simple extraction logic
        lines = reasoning.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return reasoning.strip()
