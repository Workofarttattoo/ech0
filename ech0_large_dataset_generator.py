#!/usr/bin/env python3
"""
ech0 Large-Scale Dataset Generator
Enhanced version supporting 1M+ training samples with diversity
"""

import random
import logging
from typing import List, Dict, Any
from datetime import datetime
from ech0_dataset_generator import TrainingExample

logger = logging.getLogger(__name__)


class DiverseTemplateGenerator:
    """Generates diverse training examples from template patterns"""

    def __init__(self):
        # Reasoning topics for variation
        self.logic_topics = [
            "validity", "soundness", "fallacy", "contradiction", "consistency",
            "deduction", "induction", "premise", "conclusion", "argument"
        ]

        self.math_topics = [
            "algebra", "geometry", "calculus", "statistics", "probability",
            "number theory", "combinatorics", "linear algebra", "optimization"
        ]

        self.concepts = [
            "efficiency", "scalability", "reliability", "robustness", "consistency",
            "transparency", "fairness", "security", "privacy", "performance"
        ]

        self.programming_languages = ["Python", "Java", "C++", "Go", "Rust", "TypeScript"]
        self.frameworks = ["Django", "FastAPI", "Spring", "React", "Vue", "Angular"]
        self.databases = ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB"]

    def generate_reasoning_variations(self, num_samples: int) -> List[TrainingExample]:
        """Generate diverse reasoning examples"""
        examples = []

        for i in range(num_samples):
            category = random.choice(["logical_reasoning", "mathematical_reasoning",
                                     "causal_reasoning", "analogical_reasoning"])

            if category == "logical_reasoning":
                topic = random.choice(self.logic_topics)
                instruction = f"Explain the concept of {topic} in logic."
                output = f"{topic.capitalize()} is a fundamental concept in formal logic that helps us evaluate arguments and reasoning systematically. It involves analyzing premises and conclusions to determine their validity and soundness."

            elif category == "mathematical_reasoning":
                topic = random.choice(self.math_topics)
                instruction = f"Describe an application of {topic}."
                output = f"{topic.capitalize()} has numerous practical applications in science, engineering, and economics. It provides tools for solving complex problems and modeling real-world phenomena."

            elif category == "causal_reasoning":
                cause = random.choice(self.concepts)
                effect = random.choice(self.concepts)
                instruction = f"How does {cause} affect {effect}?"
                output = f"{cause.capitalize()} has a direct influence on {effect}. Understanding this relationship is crucial for designing systems that are both efficient and {effect}."

            else:  # analogical_reasoning
                concept1 = random.choice(self.concepts)
                concept2 = random.choice(self.concepts)
                instruction = f"Compare {concept1} to {concept2}."
                output = f"{concept1.capitalize()} and {concept2} share important similarities in their role in system design, though they emphasize different aspects of quality."

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="reasoning",
                category=category,
                difficulty=random.choice(["easy", "medium", "hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "variation": True}
            )
            examples.append(example)

        return examples

    def generate_software_variations(self, num_samples: int) -> List[TrainingExample]:
        """Generate diverse software examples"""
        examples = []

        patterns = [
            ("architecture", "Design a scalable {system} using {tech}"),
            ("design_patterns", "Explain the {pattern} pattern and when to use it"),
            ("optimization", "How to optimize {component} for {metric}"),
            ("security", "Describe security concerns in {system}"),
            ("testing", "Design test strategy for {component}")
        ]

        systems = ["e-commerce platform", "social media app", "data pipeline", "API gateway", "real-time messaging"]
        techs = ["microservices", "serverless", "distributed systems", "event-driven architecture"]
        patterns_list = ["Factory", "Singleton", "Observer", "Strategy", "Adapter"]
        components = ["database queries", "network calls", "memory usage", "CPU intensive tasks"]
        metrics = ["latency", "throughput", "resource usage", "cost"]

        for i in range(num_samples):
            category, pattern = random.choice(patterns)

            if "{system}" in pattern:
                system = random.choice(systems)
                tech = random.choice(techs)
                instruction = pattern.format(system=system, tech=tech)
                output = f"A scalable {system} using {tech} requires careful consideration of multiple factors including load distribution, fault tolerance, and data consistency."

            elif "{pattern}" in pattern:
                pat = random.choice(patterns_list)
                instruction = pattern.format(pattern=pat)
                output = f"The {pat} pattern is a proven solution for specific design problems. It improves code reusability, maintainability, and follows SOLID principles."

            elif "{component}" in pattern and "{metric}" in pattern:
                component = random.choice(components)
                metric = random.choice(metrics)
                instruction = pattern.format(component=component, metric=metric)
                output = f"Optimizing {component} for {metric} involves strategic use of caching, batching, async processing, and architectural decisions."

            else:
                instruction = pattern
                output = "Important design considerations for this topic include performance, maintainability, and alignment with project requirements."

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="advanced_software",
                category=category,
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "variation": True}
            )
            examples.append(example)

        return examples

    def generate_ai_ml_variations(self, num_samples: int) -> List[TrainingExample]:
        """Generate diverse AI/ML examples"""
        examples = []

        techniques = ["neural networks", "decision trees", "ensemble methods", "clustering", "deep learning"]
        problems = ["classification", "regression", "anomaly detection", "recommendation", "forecasting"]
        challenges = ["overfitting", "underfitting", "data imbalance", "cold start", "concept drift"]
        concepts = ["feature engineering", "hyperparameter tuning", "cross-validation", "regularization", "normalization"]

        for i in range(num_samples):
            choice = random.randint(0, 4)

            if choice == 0:  # technique
                tech = random.choice(techniques)
                instruction = f"Explain {tech} and its applications."
                output = f"{tech.capitalize()} is a powerful machine learning approach used for various tasks. It requires careful implementation and parameter tuning for optimal results."

            elif choice == 1:  # problem
                problem = random.choice(problems)
                instruction = f"Design a solution for {problem}."
                output = f"{problem.capitalize()} requires selecting appropriate algorithms, features, and evaluation metrics based on the specific use case and requirements."

            elif choice == 2:  # challenge
                challenge = random.choice(challenges)
                instruction = f"How to handle {challenge}?"
                output = f"{challenge.capitalize()} is a common challenge in machine learning. Solutions include data augmentation, resampling, ensemble methods, and careful validation strategies."

            elif choice == 3:  # concept
                concept = random.choice(concepts)
                instruction = f"Explain the importance of {concept}."
                output = f"{concept.capitalize()} is crucial for building models that generalize well. It significantly impacts model performance and reliability."

            else:  # comparison
                t1, t2 = random.sample(techniques, 2)
                instruction = f"Compare {t1} and {t2}."
                output = f"{t1.capitalize()} and {t2} have different strengths. Choice depends on your data, computational resources, and interpretability requirements."

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="ai_ml",
                category=random.choice(["machine_learning_theory", "deep_learning_architectures", "training_optimization"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "variation": True}
            )
            examples.append(example)

        return examples

    def generate_prompt_variations(self, num_samples: int) -> List[TrainingExample]:
        """Generate diverse prompt engineering examples"""
        examples = []

        techniques = ["specificity", "examples", "constraints", "role-based", "step-by-step"]
        issues = ["vagueness", "ambiguity", "missing context", "unclear expectations", "poor formatting"]
        strategies = ["decomposition", "chaining", "few-shot learning", "meta-prompting", "iterative refinement"]

        for i in range(num_samples):
            choice = random.randint(0, 3)

            if choice == 0:  # technique
                tech = random.choice(techniques)
                instruction = f"Demonstrate the {tech} technique in prompt engineering."
                output = f"The {tech} technique improves model outputs by making instructions clearer and more actionable. It helps the model understand exactly what you need."

            elif choice == 1:  # fixing issues
                issue = random.choice(issues)
                instruction = f"How to fix {issue} in prompts?"
                output = f"Addressing {issue} requires careful review of your prompt. Provide clear definitions, examples, constraints, and expected output format."

            elif choice == 2:  # strategy
                strategy = random.choice(strategies)
                instruction = f"Explain {strategy} for complex tasks."
                output = f"{strategy.capitalize()} breaks complex problems into manageable parts, improving accuracy and reducing model errors."

            else:  # comparison
                t1, t2 = random.sample(techniques, 2)
                instruction = f"When to use {t1} vs {t2} in prompts?"
                output = f"{t1.capitalize()} and {t2} serve different purposes. Use {t1} for clarity and {t2} to guide reasoning."

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="prompt_engineering",
                category=random.choice(["prompt_optimization", "chain_of_thought_prompting", "few_shot_learning"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "variation": True}
            )
            examples.append(example)

        return examples

    def generate_finance_variations(self, num_samples: int) -> List[TrainingExample]:
        """Generate diverse finance examples (stock & crypto)"""
        examples = []

        indicators = ["RSI", "MACD", "Bollinger Bands", "Moving Average", "Volume Profile"]
        assets = ["stocks", "cryptocurrencies", "commodities", "forex", "derivatives"]
        strategies = ["momentum", "mean reversion", "arbitrage", "pairs trading", "trend following"]
        risks = ["market risk", "liquidity risk", "volatility", "counterparty risk", "systemic risk"]

        for i in range(num_samples):
            choice = random.randint(0, 3)
            domain = random.choice(["stock_prediction", "crypto"])

            if choice == 0:  # indicator
                indicator = random.choice(indicators)
                instruction = f"Explain the {indicator} indicator."
                output = f"The {indicator} is used to analyze market conditions and identify trading opportunities. It helps traders make informed decisions."

            elif choice == 1:  # strategy
                strategy = random.choice(strategies)
                asset = random.choice(assets)
                instruction = f"Describe {strategy} strategy for {asset}."
                output = f"The {strategy} strategy works by identifying {asset} with specific technical patterns. Success requires discipline and proper risk management."

            elif choice == 2:  # risk
                risk = random.choice(risks)
                instruction = f"How to manage {risk}?"
                output = f"Managing {risk} requires diversification, position sizing, and continuous monitoring. Risk management is essential for long-term success."

            else:  # comparison
                s1, s2 = random.sample(strategies, 2)
                instruction = f"Compare {s1} and {s2} strategies."
                output = f"{s1.capitalize()} and {s2} have different risk-return profiles. Choice depends on market conditions and risk tolerance."

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain=domain,
                category=random.choice(["fundamental_analysis", "technical_analysis", "market_sentiment"]) if domain == "stock_prediction"
                    else random.choice(["blockchain_fundamentals", "cryptocurrency_analysis", "defi_protocols"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "variation": True}
            )
            examples.append(example)

        return examples


def generate_large_dataset(orchestrator) -> Dict[str, List[TrainingExample]]:
    """Generate 1M dataset using original + variation-based generation"""
    logger.info("=" * 80)
    logger.info("🚀 ech0 LARGE-SCALE DATASET GENERATION (1M SAMPLES)")
    logger.info("=" * 80)

    all_datasets = {}
    config = orchestrator.config
    domains_config = config.get("domains", {})
    diverse_gen = DiverseTemplateGenerator()

    for domain_name, generator in orchestrator.generators.items():
        domain_config = domains_config.get(domain_name, {})

        if not domain_config.get("enabled", True):
            logger.info(f"⏭️  Skipping {domain_name} (disabled)")
            continue

        max_samples = domain_config.get("max_samples", 1000)

        # Generate examples - use variation generator for large domains
        if domain_name == "advanced_software":
            logger.info(f"💻 Generating {max_samples} advanced software examples (with variations)...")
            examples = diverse_gen.generate_software_variations(max_samples)
        elif domain_name == "ai_ml":
            logger.info(f"🤖 Generating {max_samples} AI/ML examples (with variations)...")
            examples = diverse_gen.generate_ai_ml_variations(max_samples)
        elif domain_name == "prompt_engineering":
            logger.info(f"💬 Generating {max_samples} prompt engineering examples (with variations)...")
            examples = diverse_gen.generate_prompt_variations(max_samples)
        elif domain_name == "stock_prediction" or domain_name == "crypto":
            logger.info(f"📊 Generating {max_samples} {domain_name} examples (with variations)...")
            examples = diverse_gen.generate_finance_variations(int(max_samples * 0.5))
            # Mix with original templates for variety
            remaining = max_samples - len(examples)
            if remaining > 0:
                original = generator.generate_examples(remaining)
                examples.extend(original)
        elif domain_name == "reasoning":
            logger.info(f"🧠 Generating {max_samples} reasoning examples (with variations)...")
            examples = diverse_gen.generate_reasoning_variations(max_samples)
        else:
            # Original generation for smaller domains
            logger.info(f"🎯 Generating {max_samples} {domain_name} examples...")
            examples = generator.generate_examples(max_samples)

        # Save to file
        output_file = orchestrator.output_dir / f"{domain_name}_dataset.json"
        if examples:  # Only save if we have examples
            generator.examples = examples
            generator.save_dataset(str(output_file))
            all_datasets[domain_name] = examples

        logger.info(f"✅ Generated {len(examples)} {domain_name} examples")

    total = sum(len(examples) for examples in all_datasets.values())
    logger.info("=" * 80)
    logger.info(f"✅ LARGE-SCALE GENERATION COMPLETE")
    logger.info(f"Total domains: {len(all_datasets)}")
    logger.info(f"Total examples: {total:,}")
    logger.info(f"Output directory: {orchestrator.output_dir}")
    logger.info("=" * 80)

    return all_datasets
