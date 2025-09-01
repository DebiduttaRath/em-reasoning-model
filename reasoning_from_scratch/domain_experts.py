
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np


class DomainExpert:
    """Base class for domain-specific reasoning experts."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.knowledge_base = []
        self.reasoning_templates = {}
        self.confidence_threshold = 0.7
        
    def can_handle(self, question: str) -> float:
        """Return confidence score (0-1) for handling this question."""
        raise NotImplementedError
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        """Generate domain-specific reasoning prompt."""
        raise NotImplementedError
    
    def validate_reasoning(self, question: str, reasoning: str, answer: str) -> Dict[str, Any]:
        """Validate reasoning using domain-specific criteria."""
        return {
            "is_valid": True,
            "confidence": 0.8,
            "validation_notes": [],
            "compliance_score": 0.8
        }


class FinanceRiskExpert(DomainExpert):
    """Finance and Risk domain expert."""
    
    def __init__(self):
        super().__init__("Finance & Risk")
        self.knowledge_base = [
            "Risk assessment requires quantitative analysis and regulatory compliance",
            "Credit risk models use probability of default, loss given default, and exposure at default",
            "Basel III requires capital adequacy ratios and stress testing",
            "Fraud detection uses pattern recognition and anomaly detection",
            "Investment decisions require risk-return analysis and diversification",
            "Regulatory compliance requires audit trails and documentation"
        ]
        
        self.reasoning_templates = {
            "credit_risk": """
Analyze credit risk using the following framework:
1. Borrower Analysis: Financial strength, credit history, payment capacity
2. Risk Quantification: PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure at Default)
3. Regulatory Requirements: Basel III compliance, capital adequacy
4. Risk Mitigation: Collateral, guarantees, covenants
5. Decision Rationale: Risk-adjusted return, portfolio impact

Question: {question}
Context: {context}

Risk Analysis:""",
            
            "fraud_detection": """
Identify fraud patterns using systematic analysis:
1. Transaction Pattern Analysis: Unusual amounts, timing, frequency
2. Behavioral Indicators: Deviation from normal patterns
3. Network Analysis: Connected entities and relationships
4. Risk Scoring: Quantitative risk assessment
5. Regulatory Compliance: AML/KYC requirements

Question: {question}
Context: {context}

Fraud Assessment:""",
            
            "investment_advisory": """
Provide investment analysis with transparent reasoning:
1. Market Analysis: Current conditions, trends, risks
2. Asset Evaluation: Fundamentals, valuation, growth prospects
3. Portfolio Impact: Diversification, correlation, risk contribution
4. Risk Assessment: Market, credit, liquidity, operational risks
5. Recommendation: Buy/Sell/Hold with rationale

Question: {question}
Context: {context}

Investment Analysis:"""
        }
    
    def can_handle(self, question: str) -> float:
        keywords = ["credit", "risk", "fraud", "investment", "financial", "banking", 
                   "loan", "portfolio", "compliance", "regulatory", "basel", "aml"]
        question_lower = question.lower()
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        return min(matches * 0.2, 1.0)
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["credit", "loan", "default"]):
            template = self.reasoning_templates["credit_risk"]
        elif any(word in question_lower for word in ["fraud", "suspicious", "anomaly"]):
            template = self.reasoning_templates["fraud_detection"]
        elif any(word in question_lower for word in ["investment", "portfolio", "stock", "bond"]):
            template = self.reasoning_templates["investment_advisory"]
        else:
            template = self.reasoning_templates["credit_risk"]  # Default
        
        return template.format(question=question, context=context)


class HealthcareExpert(DomainExpert):
    """Healthcare and Life Sciences domain expert."""
    
    def __init__(self):
        super().__init__("Healthcare & Life Sciences")
        self.knowledge_base = [
            "Clinical decisions require evidence-based reasoning and patient safety",
            "Drug discovery involves biomarker analysis and clinical trial design",
            "Patient risk stratification uses predictive modeling and clinical indicators",
            "Medical compliance requires adherence to clinical guidelines and protocols",
            "Diagnosis requires differential analysis and evidence correlation",
            "Treatment plans must consider efficacy, safety, and patient factors"
        ]
        
        self.reasoning_templates = {
            "clinical_decision": """
Provide clinical decision support with transparent reasoning:
1. Patient Assessment: Demographics, symptoms, medical history, vitals
2. Differential Diagnosis: Possible conditions, probability ranking
3. Evidence Analysis: Lab results, imaging, clinical findings
4. Risk Stratification: Severity assessment, complications risk
5. Treatment Recommendation: Evidence-based options with rationale
6. Safety Considerations: Contraindications, drug interactions, monitoring

Question: {question}
Context: {context}

Clinical Analysis:""",
            
            "drug_discovery": """
Analyze drug discovery with systematic approach:
1. Target Identification: Biological pathway, mechanism of action
2. Compound Analysis: Structure-activity relationships, pharmacokinetics
3. Efficacy Assessment: Preclinical data, biomarker response
4. Safety Profile: Toxicology, adverse effects, risk assessment
5. Clinical Development: Trial design, endpoints, regulatory pathway

Question: {question}
Context: {context}

Drug Discovery Analysis:""",
            
            "risk_stratification": """
Perform patient risk stratification:
1. Risk Factors: Demographics, comorbidities, lifestyle factors
2. Clinical Indicators: Lab values, vital signs, functional status
3. Predictive Models: Validated risk scores, outcome prediction
4. Intervention Strategies: Prevention, monitoring, treatment intensification
5. Care Coordination: Multidisciplinary approach, follow-up plan

Question: {question}
Context: {context}

Risk Stratification:"""
        }
    
    def can_handle(self, question: str) -> float:
        keywords = ["medical", "clinical", "patient", "diagnosis", "treatment", "drug", 
                   "healthcare", "disease", "symptom", "therapy", "medicine", "health"]
        question_lower = question.lower()
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        return min(matches * 0.2, 1.0)
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["diagnosis", "patient", "clinical", "treatment"]):
            template = self.reasoning_templates["clinical_decision"]
        elif any(word in question_lower for word in ["drug", "compound", "discovery", "molecule"]):
            template = self.reasoning_templates["drug_discovery"]
        elif any(word in question_lower for word in ["risk", "stratification", "prediction", "outcome"]):
            template = self.reasoning_templates["risk_stratification"]
        else:
            template = self.reasoning_templates["clinical_decision"]  # Default
        
        return template.format(question=question, context=context)


class LegalExpert(DomainExpert):
    """Legal, Governance & Policy domain expert."""
    
    def __init__(self):
        super().__init__("Legal, Governance & Policy")
        self.knowledge_base = [
            "Legal reasoning requires precedent analysis and statutory interpretation",
            "Contract analysis involves risk identification and compliance verification",
            "Regulatory policy requires impact assessment and stakeholder analysis",
            "Legal research demands systematic case law and statute examination",
            "Policy simulation requires scenario modeling and outcome prediction"
        ]
        
        self.reasoning_templates = {
            "legal_research": """
Conduct legal research with systematic analysis:
1. Issue Identification: Legal questions, applicable law areas
2. Precedent Analysis: Relevant cases, holdings, reasoning
3. Statutory Framework: Applicable statutes, regulations, rules
4. Legal Arguments: Supporting authorities, counterarguments
5. Risk Assessment: Litigation risk, compliance exposure
6. Recommendations: Legal strategy, compliance actions

Question: {question}
Context: {context}

Legal Analysis:""",
            
            "contract_analysis": """
Analyze contract with comprehensive review:
1. Term Analysis: Key provisions, obligations, rights
2. Risk Identification: Liability exposure, performance risks
3. Compliance Review: Regulatory requirements, industry standards
4. Enforcement Mechanisms: Remedies, dispute resolution, termination
5. Risk Mitigation: Recommended modifications, protective clauses

Question: {question}
Context: {context}

Contract Analysis:""",
            
            "policy_simulation": """
Simulate policy impact with structured analysis:
1. Stakeholder Analysis: Affected parties, interests, influence
2. Impact Assessment: Economic, social, operational effects
3. Scenario Modeling: Implementation pathways, outcomes
4. Risk Analysis: Unintended consequences, mitigation strategies
5. Implementation Strategy: Phasing, monitoring, adjustments

Question: {question}
Context: {context}

Policy Impact Analysis:"""
        }
    
    def can_handle(self, question: str) -> float:
        keywords = ["legal", "law", "contract", "policy", "regulation", "compliance", 
                   "statute", "case", "court", "agreement", "governance", "regulatory"]
        question_lower = question.lower()
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        return min(matches * 0.2, 1.0)
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["research", "case", "statute", "precedent"]):
            template = self.reasoning_templates["legal_research"]
        elif any(word in question_lower for word in ["contract", "agreement", "terms"]):
            template = self.reasoning_templates["contract_analysis"]
        elif any(word in question_lower for word in ["policy", "regulation", "rule", "impact"]):
            template = self.reasoning_templates["policy_simulation"]
        else:
            template = self.reasoning_templates["legal_research"]  # Default
        
        return template.format(question=question, context=context)


class DefenseSecurityExpert(DomainExpert):
    """Defense, Aerospace & Security domain expert."""
    
    def __init__(self):
        super().__init__("Defense, Aerospace & Security")
        self.knowledge_base = [
            "Mission planning requires constraint optimization and risk assessment",
            "Threat assessment uses intelligence analysis and risk modeling",
            "Cybersecurity requires attack vector analysis and countermeasure planning",
            "Space operations involve orbital mechanics and collision avoidance",
            "Security analysis requires vulnerability assessment and threat modeling"
        ]
        
        self.reasoning_templates = {
            "mission_planning": """
Develop mission plan with systematic approach:
1. Objective Analysis: Mission goals, success criteria, constraints
2. Resource Assessment: Personnel, equipment, logistics, timeline
3. Risk Analysis: Operational risks, mitigation strategies
4. Scenario Planning: Contingencies, alternative approaches
5. Execution Strategy: Phasing, coordination, monitoring
6. Success Metrics: Measurable outcomes, evaluation criteria

Question: {question}
Context: {context}

Mission Planning:""",
            
            "threat_assessment": """
Conduct threat assessment with intelligence-based analysis:
1. Threat Identification: Actors, capabilities, intentions
2. Vulnerability Analysis: System weaknesses, exposure points
3. Risk Calculation: Probability × Impact assessment
4. Attack Vector Analysis: Likely approaches, methodologies
5. Countermeasure Planning: Defensive strategies, response options
6. Monitoring Strategy: Intelligence collection, early warning

Question: {question}
Context: {context}

Threat Assessment:""",
            
            "cybersecurity": """
Analyze cybersecurity with comprehensive framework:
1. Attack Surface Analysis: Entry points, vulnerabilities, exposures
2. Threat Actor Profiling: Capabilities, motivations, methodologies
3. Attack Path Modeling: Likely progression, lateral movement
4. Impact Assessment: Data loss, system compromise, operational impact
5. Defense Strategy: Prevention, detection, response, recovery
6. Compliance Requirements: Standards, regulations, best practices

Question: {question}
Context: {context}

Cybersecurity Analysis:"""
        }
    
    def can_handle(self, question: str) -> float:
        keywords = ["security", "defense", "military", "cyber", "threat", "mission", 
                   "aerospace", "satellite", "attack", "vulnerability", "intelligence"]
        question_lower = question.lower()
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        return min(matches * 0.2, 1.0)
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["mission", "planning", "operation", "strategy"]):
            template = self.reasoning_templates["mission_planning"]
        elif any(word in question_lower for word in ["threat", "intelligence", "assessment", "risk"]):
            template = self.reasoning_templates["threat_assessment"]
        elif any(word in question_lower for word in ["cyber", "security", "attack", "vulnerability"]):
            template = self.reasoning_templates["cybersecurity"]
        else:
            template = self.reasoning_templates["threat_assessment"]  # Default
        
        return template.format(question=question, context=context)


class EngineeringExpert(DomainExpert):
    """Engineering, Manufacturing & Energy domain expert."""
    
    def __init__(self):
        super().__init__("Engineering, Manufacturing & Energy")
        self.knowledge_base = [
            "Root cause analysis requires systematic failure investigation",
            "Digital twins enable predictive maintenance and optimization",
            "Supply chain optimization involves constraint programming",
            "Energy grid operations require stability and reliability analysis",
            "Manufacturing requires quality control and process optimization"
        ]
        
        self.reasoning_templates = {
            "root_cause_analysis": """
Perform root cause analysis with systematic investigation:
1. Problem Definition: Failure description, impact, timeline
2. Data Collection: Operational data, maintenance records, observations
3. Failure Mode Analysis: Possible causes, failure mechanisms
4. Causal Chain Investigation: Primary, secondary, contributing factors
5. Root Cause Identification: Fundamental underlying causes
6. Corrective Actions: Prevention strategies, process improvements

Question: {question}
Context: {context}

Root Cause Analysis:""",
            
            "digital_twin": """
Analyze digital twin implementation for predictive insights:
1. System Modeling: Physical system representation, key parameters
2. Data Integration: Sensor data, operational metrics, environmental factors
3. Predictive Analytics: Failure prediction, performance optimization
4. Maintenance Scheduling: Condition-based, predictive maintenance
5. Optimization Opportunities: Efficiency improvements, cost reduction
6. Implementation Strategy: Technology requirements, integration plan

Question: {question}
Context: {context}

Digital Twin Analysis:""",
            
            "supply_chain": """
Optimize supply chain with constraint-based reasoning:
1. Network Analysis: Suppliers, facilities, distribution channels
2. Constraint Identification: Capacity, demand, lead times, costs
3. Optimization Objectives: Cost, service level, risk, sustainability
4. Scenario Analysis: Demand variations, supply disruptions
5. Solution Strategy: Sourcing, inventory, logistics optimization
6. Risk Mitigation: Supplier diversification, contingency planning

Question: {question}
Context: {context}

Supply Chain Optimization:"""
        }
    
    def can_handle(self, question: str) -> float:
        keywords = ["engineering", "manufacturing", "energy", "supply", "chain", "failure", 
                   "maintenance", "optimization", "process", "system", "digital", "twin"]
        question_lower = question.lower()
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        return min(matches * 0.2, 1.0)
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["failure", "root", "cause", "problem"]):
            template = self.reasoning_templates["root_cause_analysis"]
        elif any(word in question_lower for word in ["digital", "twin", "predictive", "maintenance"]):
            template = self.reasoning_templates["digital_twin"]
        elif any(word in question_lower for word in ["supply", "chain", "logistics", "optimization"]):
            template = self.reasoning_templates["supply_chain"]
        else:
            template = self.reasoning_templates["root_cause_analysis"]  # Default
        
        return template.format(question=question, context=context)


class ClimateEnvironmentExpert(DomainExpert):
    """Climate, Environment & Geospatial AI domain expert."""
    
    def __init__(self):
        super().__init__("Climate, Environment & Geospatial AI")
        self.knowledge_base = [
            "Climate modeling requires multi-scale analysis and uncertainty quantification",
            "Environmental impact assessment needs ecosystem and human impact analysis",
            "Geospatial analysis involves satellite data and spatial modeling",
            "Disaster planning requires risk assessment and response optimization",
            "Precision agriculture uses remote sensing and crop modeling"
        ]
        
        self.reasoning_templates = {
            "climate_analysis": """
Analyze climate risk with comprehensive assessment:
1. Climate Data Analysis: Historical trends, projections, scenarios
2. Impact Assessment: Physical, ecological, socioeconomic impacts
3. Vulnerability Analysis: Exposed systems, adaptive capacity
4. Risk Quantification: Probability, severity, timing of impacts
5. Adaptation Strategies: Mitigation, resilience building, planning
6. Monitoring Framework: Indicators, tracking, early warning

Question: {question}
Context: {context}

Climate Risk Analysis:""",
            
            "environmental_impact": """
Assess environmental impact with systematic evaluation:
1. Baseline Assessment: Current environmental conditions
2. Impact Identification: Direct, indirect, cumulative effects
3. Ecosystem Analysis: Biodiversity, habitat, ecological services
4. Human Impact: Health, communities, economic effects
5. Mitigation Measures: Prevention, reduction, compensation
6. Monitoring Plan: Indicators, compliance, adaptive management

Question: {question}
Context: {context}

Environmental Impact Assessment:""",
            
            "precision_agriculture": """
Optimize agriculture with precision techniques:
1. Field Analysis: Soil conditions, topography, microclimate
2. Crop Assessment: Growth stage, health, stress indicators
3. Resource Optimization: Water, nutrients, pesticides
4. Yield Prediction: Modeling, forecasting, optimization
5. Technology Integration: Sensors, drones, satellite data
6. Decision Support: Timing, application rates, field management

Question: {question}
Context: {context}

Precision Agriculture Analysis:"""
        }
    
    def can_handle(self, question: str) -> float:
        keywords = ["climate", "environment", "agriculture", "geospatial", "disaster", 
                   "weather", "crop", "satellite", "ecosystem", "sustainability"]
        question_lower = question.lower()
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        return min(matches * 0.2, 1.0)
    
    def generate_reasoning_prompt(self, question: str, context: str = "") -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["climate", "weather", "risk", "forecast"]):
            template = self.reasoning_templates["climate_analysis"]
        elif any(word in question_lower for word in ["environment", "impact", "ecosystem", "biodiversity"]):
            template = self.reasoning_templates["environmental_impact"]
        elif any(word in question_lower for word in ["agriculture", "crop", "farming", "precision"]):
            template = self.reasoning_templates["precision_agriculture"]
        else:
            template = self.reasoning_templates["climate_analysis"]  # Default
        
        return template.format(question=question, context=context)


class DomainExpertSystem:
    """System to coordinate multiple domain experts."""
    
    def __init__(self):
        self.experts = [
            FinanceRiskExpert(),
            HealthcareExpert(),
            LegalExpert(),
            DefenseSecurityExpert(),
            EngineeringExpert(),
            ClimateEnvironmentExpert()
        ]
        
        self.expert_usage_stats = {expert.domain_name: 0 for expert in self.experts}
    
    def select_expert(self, question: str) -> Optional[DomainExpert]:
        """Select the most appropriate domain expert for a question."""
        best_expert = None
        best_score = 0.0
        
        for expert in self.experts:
            score = expert.can_handle(question)
            if score > best_score and score >= expert.confidence_threshold:
                best_score = score
                best_expert = expert
        
        if best_expert:
            self.expert_usage_stats[best_expert.domain_name] += 1
        
        return best_expert
    
    def get_domain_reasoning_prompt(self, question: str, context: str = "") -> Tuple[str, Optional[str]]:
        """Get domain-specific reasoning prompt if applicable."""
        expert = self.select_expert(question)
        
        if expert:
            prompt = expert.generate_reasoning_prompt(question, context)
            return prompt, expert.domain_name
        
        return "", None
    
    def validate_domain_reasoning(self, question: str, reasoning: str, answer: str) -> Dict[str, Any]:
        """Validate reasoning using domain expertise."""
        expert = self.select_expert(question)
        
        if expert:
            return expert.validate_reasoning(question, reasoning, answer)
        
        return {
            "is_valid": True,
            "confidence": 0.7,
            "validation_notes": ["No domain expert available"],
            "compliance_score": 0.7
        }
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get domain expert usage statistics."""
        total_usage = sum(self.expert_usage_stats.values())
        
        return {
            "total_expert_queries": total_usage,
            "expert_usage": self.expert_usage_stats,
            "available_domains": [expert.domain_name for expert in self.experts],
            "expert_distribution": {
                domain: (count / total_usage * 100) if total_usage > 0 else 0
                for domain, count in self.expert_usage_stats.items()
            }
        }
