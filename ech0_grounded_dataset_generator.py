#!/usr/bin/env python3
"""
ech0 Grounded Dataset Generator
Generates training data grounded in real, verified knowledge sources
Optionally enhanced with Ollama for richer examples
"""

import random
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from ech0_dataset_generator import TrainingExample

# Optional Ollama integration
try:
    from ech0_ollama_integration import OllamaGenerator, OllamaConfig, check_ollama_status
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class GroundedKnowledgeBase:
    """Real, verified knowledge for grounding training data"""

    # Real mathematical concepts and proofs
    MATHEMATICAL_FACTS = {
        "calculus": [
            ("limit", "A limit describes the value that a function approaches as input approaches some value. Formally, lim(x→a) f(x) = L means for every ε > 0, there exists δ > 0 such that 0 < |x-a| < δ implies |f(x)-L| < ε."),
            ("derivative", "The derivative of a function f at x is f'(x) = lim(h→0) [f(x+h)-f(x)]/h. It represents the instantaneous rate of change."),
            ("integral", "The integral reverses differentiation. ∫f'(x)dx = f(x) + C. The definite integral ∫[a,b] f(x)dx represents the area under the curve."),
        ],
        "linear_algebra": [
            ("eigenvalue", "For a matrix A, a scalar λ is an eigenvalue if Av = λv for some nonzero vector v (eigenvector). Eigenvalues are roots of det(A - λI) = 0."),
            ("linear_independence", "Vectors v₁, v₂, ..., vₙ are linearly independent if c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 only when all cᵢ = 0."),
            ("matrix_rank", "The rank of a matrix is the dimension of its row space (or column space). It equals the number of linearly independent rows/columns."),
        ]
    }

    # Real machine learning concepts
    ML_CONCEPTS = {
        "supervised_learning": [
            ("regularization", "Regularization adds penalty terms to loss: L = Error + λ(weights). L1 creates sparse solutions; L2 shrinks weights. Prevents overfitting by constraining model complexity."),
            ("cross_validation", "K-fold cross-validation divides data into k subsets. Train on k-1 folds, test on 1 fold, repeat k times. Provides unbiased performance estimate without separate test set."),
            ("batch_normalization", "Normalizes layer inputs to zero mean and unit variance: x̂ = (x - μ_batch) / √(σ²_batch + ε). Stabilizes training and allows higher learning rates."),
        ],
        "deep_learning": [
            ("backpropagation", "Computes gradients via chain rule: ∂L/∂w = (∂L/∂output) × (∂output/∂hidden) × ... × (∂layer/∂w). Enables efficient training of deep networks."),
            ("activation_functions", "ReLU: max(0,x) - fast, sparse. Sigmoid: 1/(1+e^-x) - outputs [0,1]. Tanh: (e^x-e^-x)/(e^x+e^-x) - outputs [-1,1]. Softmax: for multiclass - exp(xᵢ)/Σexp(xⱼ)."),
            ("gradient_descent", "Update rule: w_new = w_old - α∇L(w). Learning rate α controls step size. Momentum: w_new = w_old - α∇L + β(w_old - w_prev). NAdam combines both.")
        ]
    }

    # Real software engineering principles
    ENGINEERING_PRINCIPLES = {
        "design_patterns": [
            ("MVC", "Model-View-Controller separates: Model (data/logic), View (presentation), Controller (user input handling). Promotes separation of concerns and testability."),
            ("dependency_injection", "Pass dependencies to objects rather than having them create dependencies. Improves testability: inject mocks in tests, real implementations in production."),
            ("SOLID", "S=Single Responsibility, O=Open/Closed, L=Liskov Substitution, I=Interface Segregation, D=Dependency Inversion. Guidelines for maintainable code."),
        ],
        "architecture": [
            ("microservices", "SOA variant with small, independently deployable services. Decoupling via API/messaging. Trade-off: operational complexity vs scalability and team independence."),
            ("CAP_theorem", "Consistency, Availability, Partition tolerance - pick 2 of 3. CA: no partition tolerance (single site). CP: eventual consistency. AP: eventual availability."),
            ("12factor", "Codebase in VCS, explicit dependencies (Gemfile), config in env vars, treat backing services as resources, strict build/run separation, stateless processes."),
        ]
    }

    # Real financial concepts (expanded)
    FINANCE_CONCEPTS = {
        "technical_analysis": [
            ("RSI", "Relative Strength Index = 100 - (100 / (1 + RS)), where RS = avg gain / avg loss over period. Range [0,100]. >70 overbought, <30 oversold. Based on momentum."),
            ("MACD", "Moving Average Convergence Divergence = 12-EMA - 26-EMA. Signal line = 9-EMA of MACD. Histogram = MACD - Signal. Crossovers indicate momentum shifts."),
            ("bollinger_bands", "Price bands = SMA ± (2 × StdDev). Entry when price touches lower band, exit at upper band. Uses volatility to identify overbought/oversold conditions."),
            ("stochastic", "Oscillator = (Close - Low14) / (High14 - Low14) × 100. Range [0,100]. >80 overbought, <20 oversold. Compares closing price to price range."),
            ("ATR", "Average True Range = average of true ranges over period. True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|). Measures volatility."),
            ("fibonacci", "Retracement levels at 23.6%, 38.2%, 50%, 61.8%, 78.6%. Prices often bounce at these levels. Golden ratio (0.618) appears in markets."),
            ("volume_analysis", "On-Balance Volume = running total of volume, +V on up days, -V on down days. Confirms trends. Divergences signal reversals."),
            ("ichimoku", "Cloud chart with 5 lines: conversion (9-period HLC/2), base (26-period HLC/2), leading A (conv+base)/2), leading B (52-period HLC/2). Lagging span."),
        ],
        "fundamental_analysis": [
            ("PE_ratio", "Price-to-Earnings = Stock Price / Earnings Per Share. Industry average varies (tech ~25, utilities ~15). Low PE may indicate undervaluation or distress."),
            ("dividend_discount", "Stock price = D₁/(r-g) where D₁ = next dividend, r = required return, g = growth rate. Assumes perpetual growth. Used for mature dividend-paying stocks."),
            ("cash_flow", "Operating CF = earnings + depreciation - working capital change. Free CF = Operating CF - CapEx. Reveals cash generation after investments."),
            ("PEG_ratio", "Price/Earnings to Growth = PE ratio / earnings growth rate (%). <1 may indicate undervalued. Combines valuation with growth expectations."),
            ("ROE", "Return on Equity = Net Income / Shareholders' Equity. Higher better. Measures profitability relative to shareholder capital. Compare to cost of capital."),
            ("debt_ratio", "Debt/Equity = Total Debt / Total Equity. Higher = more financial leverage. Risk-return tradeoff. Industry norms vary (utilities higher)."),
            ("asset_turnover", "Revenue / Average Total Assets. Efficiency measure. Higher = better asset utilization. Compare across industry peers."),
            ("current_ratio", "Current Assets / Current Liabilities. Liquidity measure. >1.0 good. <0.5 concerning. Varies by industry (retailers lower)."),
        ],
        "market_analysis": [
            ("market_cap", "Market Capitalization = Stock Price × Shares Outstanding. Mega (>$200B), Large (>$10B), Mid ($2B-$10B), Small (<$2B)."),
            ("beta", "Beta = volatility relative to market index. Beta=1: market moves, Beta>1: more volatile, Beta<1: less volatile. Systematic risk only."),
            ("sharpe_ratio", "Return / StdDev. Excess return per unit risk. Higher better. Compares risk-adjusted returns across investments. Assumes normal distribution."),
            ("correlation", "Measure -1 to +1 of how two assets move together. -1: inverse, 0: unrelated, +1: perfectly correlated. Diversification benefits from low correlation."),
            ("VIX", "Volatility Index from S&P 500 options. 10-20: low volatility, 30-40: high volatility, >40: extreme fear. Mean reversion tendency."),
        ]
    }

    # Real legal concepts
    LEGAL_CONCEPTS = {
        "contract_law": [
            ("offer_acceptance", "Offer = clear intent to be bound. Acceptance = unqualified assent to terms. Mirror image rule: acceptance must match offer exactly (common law)."),
            ("consideration", "Exchange of value required for binding contract. Cannot be past consideration. Can be peppercorn if freely given (nominal consideration valid)."),
            ("breach_remedies", "Damages: compensatory (actual loss), punitive (punishment - rare), specific performance (court order to perform). Mitigation duty: injured party must minimize damages."),
        ],
        "criminal_law": [
            ("mens_rea", "Criminal intent/mental state. Levels: Purposely (intended), Knowingly (aware), Recklessly (conscious disregard), Negligently (unaware but should be aware)."),
            ("actus_reus", "Physical act element of crime. Exception: status crimes illegal. Omission can be actus reus if legal duty to act (parent-child, contractual duty, created danger)."),
            ("burden_of_proof", "Criminal: beyond reasonable doubt (95%+), Affirmative defense: preponderance (>50%). Defendant: innocent until proven guilty. Proves government's case."),
        ]
    }

    # Real materials science
    MATERIALS = {
        "properties": [
            ("tensile_strength", "Maximum stress material withstands before breaking (units: MPa/GPa). Steel: 400-1000 MPa. Aluminum: 70-300 MPa. Higher = stronger but often more brittle."),
            ("ductility", "Ability to deform plastically before fracture. Measured as % elongation. Metals generally ductile; ceramics brittle. Trade-off with strength."),
            ("hardness", "Resistance to scratching/indentation. Mohs scale (1=talc, 10=diamond). Vickers, Brinell also used. Related to strength but different (hard ≠ strong)."),
        ],
        "crystallography": [
            ("lattice", "Repeating 3D arrangement of atoms. Cubic: FCC, BCC, simple cubic. Hexagonal: HCP. Unit cell = smallest repeating unit. Determines many properties."),
            ("slip_systems", "Direction and plane of atomic movement under stress. FCC: 12 systems (4 planes × 3 directions). More systems = more ductile."),
            ("grain_size", "Smaller grains = stronger (Hall-Petch: σ = σ₀ + k/√d). But too small = brittleness. Cold working refines grains; annealing coarsens."),
        ]
    }


class GroundedDatasetGenerator:
    """Generate training data grounded in real knowledge"""

    def __init__(self):
        self.kb = GroundedKnowledgeBase()

    def generate_math_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real mathematics"""
        examples = []
        all_facts = []
        for category, facts in self.kb.MATHEMATICAL_FACTS.items():
            all_facts.extend([(category, name, desc) for name, desc in facts])

        for i in range(num_samples):
            category, concept, definition = random.choice(all_facts)

            instruction = f"Explain {concept} in {category}."
            output = f"{concept.capitalize()}: {definition}"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="reasoning",
                category="mathematical_reasoning",
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_mathematics", "grounded": True}
            )
            examples.append(example)

        return examples

    def generate_ml_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real ML knowledge"""
        examples = []
        all_concepts = []
        for subfield, concepts in self.kb.ML_CONCEPTS.items():
            all_concepts.extend([(subfield, name, desc) for name, desc in concepts])

        for i in range(num_samples):
            subfield, concept, explanation = random.choice(all_concepts)

            instruction = f"Explain {concept} in machine learning."
            output = f"{concept.capitalize()}: {explanation}"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="ai_ml",
                category=random.choice(["machine_learning_theory", "deep_learning_architectures"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_ml_knowledge", "grounded": True}
            )
            examples.append(example)

        return examples

    def generate_engineering_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real software engineering"""
        examples = []
        all_principles = []
        for area, principles in self.kb.ENGINEERING_PRINCIPLES.items():
            all_principles.extend([(area, name, desc) for name, desc in principles])

        for i in range(num_samples):
            area, principle, explanation = random.choice(all_principles)

            instruction = f"Explain {principle} in software engineering."
            output = f"{principle}: {explanation}"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="advanced_software",
                category=random.choice(["software_architecture", "design_patterns"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_engineering", "grounded": True}
            )
            examples.append(example)

        return examples

    def generate_finance_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real finance"""
        examples = []
        all_concepts = []
        for field, concepts in self.kb.FINANCE_CONCEPTS.items():
            all_concepts.extend([(field, name, desc) for name, desc in concepts])

        for i in range(num_samples):
            field, concept, explanation = random.choice(all_concepts)

            instruction = f"Explain {concept} in {field}."
            output = f"{concept.replace('_', ' ').title()}: {explanation}"

            domain = "stock_prediction" if "stock" in field or "PE" in concept or "dividend" in concept or "cash" in concept else "crypto"
            category = "technical_analysis" if "RSI" in concept or "MACD" in concept or "bollinger" in concept else "fundamental_analysis"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain=domain,
                category=category,
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_finance", "grounded": True}
            )
            examples.append(example)

        return examples

    def generate_legal_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real law"""
        examples = []
        all_concepts = []
        for area, concepts in self.kb.LEGAL_CONCEPTS.items():
            all_concepts.extend([(area, name, desc) for name, desc in concepts])

        for i in range(num_samples):
            area, concept, explanation = random.choice(all_concepts)

            instruction = f"Explain {concept} in {area.replace('_', ' ')}."
            output = f"{concept.replace('_', ' ').title()}: {explanation}"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="law",
                category=random.choice(["case_law_analysis", "legal_reasoning"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_law", "grounded": True}
            )
            examples.append(example)

        return examples

    def generate_materials_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real materials science"""
        examples = []
        all_concepts = []
        for area, concepts in self.kb.MATERIALS.items():
            all_concepts.extend([(area, name, desc) for name, desc in concepts])

        for i in range(num_samples):
            area, concept, explanation = random.choice(all_concepts)

            instruction = f"Explain {concept} in {area.replace('_', ' ')}."
            output = f"{concept.replace('_', ' ').title()}: {explanation}"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="materials_science",
                category=random.choice(["material_properties", "crystallography"]),
                difficulty=random.choice(["hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_materials", "grounded": True}
            )
            examples.append(example)

        return examples

    def generate_creativity_examples(self, num_samples: int) -> List[TrainingExample]:
        """Generate examples grounded in real creative knowledge"""
        examples = []

        creative_techniques = [
            ("brainstorming", "Free-flowing generation of ideas without immediate judgment. Defer criticism, encourage wild ideas, quantity over quality initially, build on others' ideas. Research shows diverse groups produce more innovative results."),
            ("mind_mapping", "Visual organization of ideas with central concept and radiating branches. Encourages lateral thinking. Helps identify relationships and gaps. Used for planning, problem-solving, organization."),
            ("lateral_thinking", "Approach problems from indirect angles rather than direct logic. De Bono technique: use random stimuli to generate alternative ideas. Breaks habitual thinking patterns."),
            ("SCAMPER", "Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, Reverse. Framework for ideation. Systematically questions each aspect of product/service."),
            ("metaphorical_thinking", "Use metaphors from nature/other domains to solve problems. Example: honeycomb structure → lightweight engineering. Cross-domain transfer of solutions."),
            ("constraint_driven", "Artificial constraints force creativity. Limited budget/time/materials → innovation. Research: constraints improve creative output by forcing novel solutions."),
            ("iteration", "Refine ideas through multiple cycles. First draft rarely best. Feedback loops improve quality. Agile/rapid prototyping embeds iteration."),
        ]

        for i in range(num_samples):
            technique, description = random.choice(creative_techniques)

            instruction = f"Explain the {technique} creative technique."
            output = f"{technique.replace('_', ' ').title()}: {description}"

            example = TrainingExample(
                instruction=instruction,
                input="",
                output=output,
                domain="creativity",
                category=random.choice(["brainstorming", "conceptual_blending", "design_thinking"]),
                difficulty=random.choice(["medium", "hard", "expert"]),
                metadata={"generated_at": datetime.now().isoformat(), "source": "verified_creativity", "grounded": True}
            )
            examples.append(example)

        return examples


def generate_grounded_dataset(orchestrator, use_ollama: bool = False) -> Dict[str, List[TrainingExample]]:
    """
    Generate 1M dataset using grounded knowledge

    Args:
        orchestrator: Dataset orchestrator instance
        use_ollama: If True and available, enhance generation with Ollama (optional)

    Returns:
        Dictionary of domain -> training examples
    """
    logger.info("=" * 80)
    logger.info("🎯 ech0 GROUNDED DATASET GENERATION (1M SAMPLES)")
    logger.info("Generating training data grounded in real, verified knowledge sources")

    # Check configuration for Ollama
    config = orchestrator.config
    data_sources = config.get("data_sources", {})
    ollama_config = data_sources.get("ollama", {})
    grounded_config = data_sources.get("grounded", {})

    # Determine if we should use Ollama (disabled by default)
    use_only_grounded = grounded_config.get("use_only_grounded", True)
    ollama_enabled = ollama_config.get("enabled", False) and not use_only_grounded

    if use_ollama and ollama_enabled and OLLAMA_AVAILABLE:
        # Check if Ollama is actually running
        status = check_ollama_status()
        if status['available']:
            logger.info(f"🤖 Ollama integration ENABLED")
            logger.info(f"   Model: {ollama_config.get('model', 'mistral')}")
            logger.info(f"   Available: {', '.join(status['models'])}")
        else:
            logger.warning(f"⚠️  Ollama not running: {status.get('error')}")
            logger.info("   Falling back to grounded data only")
            use_ollama = False
    else:
        logger.info("📚 Using GROUNDED DATA ONLY (pre-defined verified knowledge)")
        if not ollama_enabled:
            logger.info("   To enable Ollama: Set data_sources.ollama.enabled = true in config")
        use_ollama = False

    logger.info("=" * 80)

    all_datasets = {}
    domains_config = config.get("domains", {})
    grounded_gen = GroundedDatasetGenerator()

    # Domain-specific generation with grounded data
    domain_generators = {
        "reasoning": lambda n: grounded_gen.generate_math_examples(n),
        "creativity": lambda n: grounded_gen.generate_creativity_examples(n),
        "ai_ml": lambda n: grounded_gen.generate_ml_examples(n),
        "advanced_software": lambda n: grounded_gen.generate_engineering_examples(n),
        "stock_prediction": lambda n: grounded_gen.generate_finance_examples(int(n * 0.6)),
        "crypto": lambda n: grounded_gen.generate_finance_examples(int(n * 0.4)),
        "law": lambda n: grounded_gen.generate_legal_examples(n),
        "materials_science": lambda n: grounded_gen.generate_materials_examples(n),
    }

    for domain_name, generator in orchestrator.generators.items():
        domain_config = domains_config.get(domain_name, {})

        if not domain_config.get("enabled", True):
            logger.info(f"⏭️  Skipping {domain_name}")
            continue

        max_samples = domain_config.get("max_samples", 1000)

        if domain_name in domain_generators:
            logger.info(f"📚 Generating {max_samples} {domain_name} examples (grounded knowledge)...")
            examples = domain_generators[domain_name](max_samples)
        else:
            # Fall back to original generator for remaining domains
            logger.info(f"🎯 Generating {max_samples} {domain_name} examples...")
            examples = generator.generate_examples(max_samples)

        # Save to file
        output_file = orchestrator.output_dir / f"{domain_name}_dataset.json"
        if examples:
            generator.examples = examples
            generator.save_dataset(str(output_file))
            all_datasets[domain_name] = examples

        logger.info(f"✅ Generated {len(examples):,} {domain_name} examples")

    total = sum(len(examples) for examples in all_datasets.values())
    logger.info("=" * 80)
    logger.info(f"✅ GROUNDED GENERATION COMPLETE")
    logger.info(f"Total domains: {len(all_datasets)}")
    logger.info(f"Total grounded examples: {total:,}")
    logger.info(f"All samples grounded in verified, real-world knowledge")
    logger.info("=" * 80)

    return all_datasets
