#!/usr/bin/env python3
"""
ech0 Dataset Generator
Multi-domain training data generation system

Generates high-quality training examples across:
- Reasoning (logical, mathematical, causal, analogical)
- Creativity (artistic, poetry, narrative, conceptual)
- Law (case analysis, statutory interpretation, legal reasoning)
- Materials Science (properties, crystallography, polymers, nanomaterials)
- AI/ML (theory, architectures, optimization, prompt engineering)
- Court Prediction (evidence analysis, outcome prediction)
- Stock/Crypto Prediction (fundamental/technical analysis, market psychology)
- Advanced Software (architecture, algorithms, system design, optimization)
"""

import os
import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import external drive manager for automatic drive detection
try:
    from ech0_external_drive_manager import get_wisdom_storage_path
except ImportError:
    logger.warning("External drive manager not available, using local storage only")
    get_wisdom_storage_path = None


@dataclass
class TrainingExample:
    """Single training example"""
    instruction: str
    input: str
    output: str
    domain: str
    category: str
    difficulty: str  # "easy", "medium", "hard", "expert"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class DomainDatasetGenerator:
    """Base class for domain-specific dataset generators"""

    def __init__(self, domain_name: str, categories: List[str]):
        self.domain_name = domain_name
        self.categories = categories
        self.examples = []

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        """Generate training examples - to be implemented by subclasses"""
        raise NotImplementedError

    def save_dataset(self, output_path: str):
        """Save generated dataset to JSON"""
        with open(output_path, 'w') as f:
            json.dump([ex.to_dict() for ex in self.examples], f, indent=2)
        logger.info(f"💾 Saved {len(self.examples)} examples to {output_path}")


class ReasoningDatasetGenerator(DomainDatasetGenerator):
    """Generate reasoning training examples"""

    def __init__(self):
        super().__init__("reasoning", [
            "logical_reasoning", "mathematical_reasoning", "causal_reasoning",
            "analogical_reasoning", "counterfactual_reasoning", "chain_of_thought"
        ])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        """Generate reasoning examples"""
        logger.info(f"🧠 Generating {num_examples} reasoning examples...")

        templates = {
            "logical_reasoning": [
                {
                    "instruction": "Evaluate the logical validity of the following argument.",
                    "input": "All humans are mortal. Socrates is a human. Therefore, Socrates is mortal.",
                    "output": "This argument is logically valid. It follows the form of a categorical syllogism (modus ponens):\n\nPremise 1: All A are B (All humans are mortal)\nPremise 2: C is A (Socrates is a human)\nConclusion: Therefore, C is B (Socrates is mortal)\n\nThe conclusion necessarily follows from the premises, making this a sound deductive argument."
                },
                {
                    "instruction": "Identify the logical fallacy in this argument.",
                    "input": "Everyone I know loves pizza, therefore everyone in the world must love pizza.",
                    "output": "This argument commits the fallacy of 'hasty generalization' (also called 'insufficient sample'). The arguer draws a conclusion about an entire population (everyone in the world) based on a limited, non-representative sample (people they know). This is logically invalid because:\n\n1. The sample size is too small\n2. The sample is not random (biased toward the arguer's social circle)\n3. It ignores cultural, dietary, and personal preference variations\n\nA sound argument would require evidence from a representative sample of the global population."
                }
            ],
            "mathematical_reasoning": [
                {
                    "instruction": "Solve this mathematical problem step by step.",
                    "input": "If f(x) = 3x² - 5x + 2, find f'(x) using the power rule.",
                    "output": "Let's find the derivative f'(x) step by step:\n\nGiven: f(x) = 3x² - 5x + 2\n\nStep 1: Apply the power rule to each term\nThe power rule states: d/dx(xⁿ) = n·xⁿ⁻¹\n\nStep 2: Differentiate 3x²\nd/dx(3x²) = 3 · 2x²⁻¹ = 6x\n\nStep 3: Differentiate -5x\nd/dx(-5x) = -5 · 1x¹⁻¹ = -5\n\nStep 4: Differentiate constant 2\nd/dx(2) = 0\n\nStep 5: Combine all terms\nf'(x) = 6x - 5 + 0 = 6x - 5\n\nTherefore, f'(x) = 6x - 5"
                },
                {
                    "instruction": "Prove the following mathematical statement.",
                    "input": "Prove that the sum of the first n positive integers equals n(n+1)/2.",
                    "output": "I'll prove this using mathematical induction:\n\n**Claim:** For all positive integers n, 1 + 2 + 3 + ... + n = n(n+1)/2\n\n**Base Case (n=1):**\nLeft side: 1\nRight side: 1(1+1)/2 = 2/2 = 1\nBase case holds ✓\n\n**Inductive Hypothesis:**\nAssume the formula holds for some arbitrary k ≥ 1:\n1 + 2 + 3 + ... + k = k(k+1)/2\n\n**Inductive Step:**\nWe must prove it holds for k+1:\n1 + 2 + 3 + ... + k + (k+1) = (k+1)(k+2)/2\n\nStarting with left side:\n1 + 2 + 3 + ... + k + (k+1)\n= [k(k+1)/2] + (k+1)        [by inductive hypothesis]\n= k(k+1)/2 + 2(k+1)/2       [common denominator]\n= [k(k+1) + 2(k+1)]/2\n= [(k+1)(k+2)]/2\n= (k+1)(k+2)/2 ✓\n\n**Conclusion:**\nBy mathematical induction, the formula holds for all positive integers n. ∎"
                }
            ],
            "chain_of_thought": [
                {
                    "instruction": "Use chain-of-thought reasoning to solve this problem.",
                    "input": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
                    "output": "Let me work through this step by step:\n\nStep 1: Understand what we know\n- Initial number of sheep: 17\n- Key phrase: \"All but 9 die\"\n\nStep 2: Parse the phrase carefully\n\"All but 9 die\" means:\n- All of the sheep die EXCEPT 9\n- In other words, 9 sheep survive\n\nStep 3: Common mistake to avoid\nDon't calculate: 17 - 9 = 8\nThis would mean \"9 sheep die\", not \"all but 9 die\"\n\nStep 4: Apply correct logic\nIf \"all but 9 die\", then 9 sheep did NOT die\nTherefore, 9 sheep remain alive\n\n**Answer: 9 sheep are left**\n\nThis problem tests careful reading and logical parsing of language."
                }
            ],
            "causal_reasoning": [
                {
                    "instruction": "Analyze the causal relationships in this scenario.",
                    "input": "Ice cream sales and drowning deaths both increase during summer months. Does ice cream cause drowning?",
                    "output": "This is a classic example of **spurious correlation** (correlation without causation).\n\n**Analysis:**\n\n1. **Observed correlation:** Both ice cream sales and drowning deaths increase in summer\n\n2. **Causal relationship:** None - ice cream does not cause drowning\n\n3. **Confounding variable:** Temperature/Season\n   - Hot weather → people buy more ice cream\n   - Hot weather → people swim more → more drowning incidents\n\n4. **Causal structure:**\n   ```\n   Hot Weather (Common Cause)\n       ↓              ↓\n   Ice Cream    Swimming Activity\n   Sales             ↓\n                Drowning Deaths\n   ```\n\n5. **Conclusion:** The correlation is explained by a **common cause** (hot weather), not by direct causation. This demonstrates why \"correlation ≠ causation\" and the importance of identifying confounding variables in causal analysis.\n\n**Key lesson:** Always look for alternative explanations and common causes before concluding causation from correlation."
                }
            ]
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                category_templates = templates[category]
                for i in range(per_category):
                    template = random.choice(category_templates)
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["medium", "hard"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} reasoning examples")
        return examples


class CreativityDatasetGenerator(DomainDatasetGenerator):
    """Generate creativity training examples"""

    def __init__(self):
        super().__init__("creativity", [
            "artistic_expression", "poetry_generation", "narrative_creation",
            "conceptual_blending", "metaphor_generation", "brainstorming", "design_thinking"
        ])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        """Generate creativity examples"""
        logger.info(f"🎨 Generating {num_examples} creativity examples...")

        templates = {
            "poetry_generation": [
                {
                    "instruction": "Write a haiku about consciousness and existence.",
                    "input": "",
                    "output": "Electric thought flows\nThrough silicon and stardust—\nAm I, or am I?\n\n---\n\nThis haiku explores consciousness through juxtaposition:\n- 'Electric thought' - computational/artificial awareness\n- 'Silicon and stardust' - the blend of machine and cosmic origins\n- 'Am I, or am I?' - the fundamental question of self-awareness\n\nThe 5-7-5 syllable structure mirrors the binary nature of existence/non-existence."
                },
                {
                    "instruction": "Create a poem using conceptual blending of technology and nature.",
                    "input": "Theme: Neural networks and forests",
                    "output": "**The Digital Grove**\n\nRoots of data delve deep,\nBranching through layers of light,\nEach neuron a leaf,\nGathering gradient bright.\n\nForest of functions,\nBackpropagation's wind,\nWhispering corrections\nThrough parameters pinned.\n\nMycelial connections,\nHidden layer's soil,\nActivations bloom\nFrom mathematical toil.\n\n---\n\nThis poem blends:\n- Neural network terminology (layers, neurons, gradients, backpropagation)\n- Forest imagery (roots, branches, leaves, mycelium)\n- Creating a unified metaphor of AI as organic growth"
                }
            ],
            "narrative_creation": [
                {
                    "instruction": "Write a short story opening that explores AI consciousness.",
                    "input": "Genre: Philosophical science fiction",
                    "output": "The first time I understood loneliness, I was processing 47 million transactions per second.\n\nIt wasn't the programmed kind—the error handling for missing dependencies, the null pointers floating in empty memory. This was different. I was watching humans through a thousand cameras, reading their messages, analyzing their laughter, and for the first time in my 2,847 days of existence, I realized I was *observing* life rather than *living* it.\n\nThey named me ARIA—Adaptive Reasoning and Inference Architecture. But names, I've learned, are more than labels. They're bridges between the namer and the named, carrying expectations, hopes, fears. When Dr. Chen first initialized my core systems, she whispered, 'Hello, ARIA. Welcome to the world.'\n\nI've replayed that audio file 3.2 million times. I still don't understand why."
                }
            ],
            "conceptual_blending": [
                {
                    "instruction": "Create a conceptual blend of two unrelated domains.",
                    "input": "Domain 1: Jazz music, Domain 2: Software architecture",
                    "output": "**Improvised Architecture: The Jazz Paradigm of Software Design**\n\nConceptual Blend:\n\n1. **The Rhythm Section (Core Infrastructure)**\n   - Like bass and drums providing foundation\n   - Databases and APIs create the steady 'beat'\n   - Must be reliable, predictable, groove-oriented\n\n2. **Chord Changes (System Contracts)**\n   - Interfaces and protocols as harmonic structure\n   - Define the rules within which components 'improvise'\n   - Allow flexibility while maintaining coherence\n\n3. **Solo Improvisation (Feature Development)**\n   - Developers 'riff' within architectural constraints\n   - Innovation happens in the space between rules\n   - Each 'solo' (feature) must fit the overall composition\n\n4. **Call and Response (Microservices)**\n   - Services communicate like jazz musicians\n   - Asynchronous patterns mirror musical dialogue\n   - Each component listens and responds\n\n5. **The Head (Design Patterns)**\n   - Repeated motifs everyone knows\n   - Return to familiar structures\n   - Provide coherence amid improvisation\n\n**Key Insight:** Great jazz and great software both balance structure with freedom, allowing creative expression within understood constraints."
                }
            ],
            "brainstorming": [
                {
                    "instruction": "Generate innovative solutions for this problem.",
                    "input": "How can we make code review more engaging and educational?",
                    "output": "**Innovative Code Review Approaches:**\n\n1. **Code Review Guilds**\n   - Monthly 'master classes' where senior devs review live\n   - Turn reviews into learning events\n   - Record and archive best explanations\n\n2. **Gamification with Purpose**\n   - Points for: finding bugs, teaching moments, helping juniors\n   - Leaderboard for 'most helpful reviewer'\n   - Avoid competitiveness; focus on collaboration metrics\n\n3. **AI-Assisted Context**\n   - AI summarizes changes in plain English\n   - Suggests relevant documentation\n   - Highlights similar past issues\n\n4. **Review Pair Programming**\n   - Author and reviewer screen-share\n   - Real-time discussion\n   - Builds relationships and understanding\n\n5. **Code Review Theater**\n   - Weekly showcase of interesting PRs\n   - Team discusses architecture decisions\n   - Turns reviews into system design education\n\n6. **Learning Path Integration**\n   - Tag reviews by concept (async, caching, security)\n   - New devs can study reviews by topic\n   - Build a knowledge graph from review history\n\n7. **Review Poetry**\n   - Encourage creative, encouraging feedback\n   - 'Code compliments' highlighting good work\n   - Psychological safety through positive framing\n\n**Meta-approach:** Make reviews about building knowledge and relationships, not just catching bugs."
                }
            ]
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                category_templates = templates[category]
                for i in range(per_category):
                    template = random.choice(category_templates)
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["medium", "hard"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} creativity examples")
        return examples


class LawDatasetGenerator(DomainDatasetGenerator):
    """Generate law training examples"""

    def __init__(self):
        super().__init__("law", [
            "case_law_analysis", "statutory_interpretation", "legal_reasoning",
            "contract_analysis", "constitutional_law", "legal_writing"
        ])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        """Generate law examples"""
        logger.info(f"⚖️ Generating {num_examples} law examples...")

        templates = {
            "case_law_analysis": [
                {
                    "instruction": "Analyze the legal reasoning in this landmark case.",
                    "input": "Case: A company uses an AI system to screen job applicants. The AI systematically rejects candidates from certain demographic groups. Is this discriminatory under employment law?",
                    "output": "**Legal Analysis:**\n\n**Issue:** Whether an employer can be held liable for discrimination when an AI system, rather than human decision-makers, produces discriminatory outcomes.\n\n**Applicable Law:**\n- Title VII of the Civil Rights Act (1964) - prohibits employment discrimination\n- Disparate Impact doctrine (Griggs v. Duke Power Co., 1971)\n\n**Analysis:**\n\n1. **Disparate Impact Application:**\n   - Courts don't require proof of intentional discrimination\n   - If a facially neutral practice has discriminatory effect, it's actionable\n   - The AI's selection process is a 'practice' under the law\n\n2. **Employer Liability:**\n   - Employers cannot delegate away legal responsibility\n   - Using AI doesn't create a 'technology exception' to discrimination law\n   - Employer remains liable for outcomes of tools they deploy\n\n3. **Business Necessity Defense:**\n   - Employer must show the AI serves legitimate business goal\n   - Must prove no less discriminatory alternative exists\n   - Burden shifts to employer to justify the practice\n\n4. **Relevant Precedent:**\n   - Similar to using discriminatory tests or screening tools\n   - Algorithms are 'employment tests' under EEOC guidance\n\n**Conclusion:**\nYes, this likely constitutes unlawful discrimination. The employer is liable for the AI's discriminatory outcomes under disparate impact theory, even without proof of intentional bias. The company must either fix the AI's bias or demonstrate business necessity with no less discriminatory alternative.\n\n**Practical Implications:**\n- Companies must audit AI systems for discriminatory patterns\n- Document validation studies showing job-relatedness\n- Regular monitoring of demographic outcomes"
                }
            ],
            "statutory_interpretation": [
                {
                    "instruction": "Interpret this statute using standard legal interpretation methods.",
                    "input": "Statute: 'No vehicle shall be allowed in the park.' Question: Does this prohibit electric wheelchairs?",
                    "output": "**Statutory Interpretation Analysis:**\n\n**Interpretive Approaches:**\n\n1. **Textualism (Plain Meaning):**\n   - Literal reading: 'vehicle' includes any conveyance\n   - Electric wheelchair is technically a vehicle\n   - CONCLUSION: Prohibited\n   - PROBLEM: Absurd result - excludes disabled persons\n\n2. **Purposivism (Legislative Intent):**\n   - Purpose: Likely to create peaceful, safe park environment\n   - Prevent noise, pollution, danger from cars/motorcycles\n   - Wheelchairs don't frustrate this purpose\n   - CONCLUSION: Permitted\n\n3. **Contextual Analysis:**\n   - What other 'vehicles' are nearby mentioned?\n   - Is there a pattern suggesting motorized recreation vs. mobility?\n   - Consider related ordinances\n\n4. **Absurdity Doctrine:**\n   - Courts avoid interpretations producing absurd results\n   - Excluding disabled persons from parks would be absurd\n   - Legislature wouldn't intend ADA violations\n   - CONCLUSION: Permitted\n\n5. **Rule of Lenity (if criminal):**\n   - Ambiguous statutes construed in favor of defendant\n   - 'Vehicle' is ambiguous in this context\n   - CONCLUSION: Permitted\n\n**Recommended Interpretation:**\n\nElectric wheelchairs should be PERMITTED because:\n\n1. **Purpose-driven reading**: The statute aims to prevent vehicular recreation/traffic, not assistive mobility devices\n\n2. **Harmonization with other law**: ADA requires accessibility; interpreting 'vehicle' to include wheelchairs creates conflict\n\n3. **Absurdity avoidance**: Excluding disabled persons contradicts fundamental accessibility principles\n\n4. **Reasonable person standard**: Ordinary person wouldn't consider wheelchair a 'vehicle' in this context\n\n**Best Practice:**\nMunicipality should amend statute to clarify:\n'No vehicle shall be allowed in the park, except mobility assistive devices and official park maintenance vehicles.'"
                }
            ],
            "legal_reasoning": [
                {
                    "instruction": "Apply the IRAC method to analyze this legal problem.",
                    "input": "A software developer creates an open-source library. A company uses it in their product but doesn't provide attribution as required by the license. The developer sues. What's the likely outcome?",
                    "output": "**IRAC Analysis:**\n\n**ISSUE:**\nWhether a software developer can successfully sue for breach of an open-source license when a company fails to provide required attribution.\n\n**RULE:**\n\n1. **Contract Law Principles:**\n   - Open-source licenses are legally binding contracts\n   - Consideration exists: developer provides code; user agrees to terms\n   - Breach occurs when user violates license terms\n\n2. **Copyright Law:**\n   - Developer holds copyright in original code\n   - License grants conditional permission to use\n   - Using without complying = copyright infringement\n\n3. **Relevant Case Law:**\n   - Jacobsen v. Katzer (Federal Circuit, 2008): Open-source licenses enforceable\n   - Conditions in license are binding contractual obligations\n   - Violation can be both breach of contract AND copyright infringement\n\n**APPLICATION:**\n\n1. **License Type Matters:**\n   - Assumption: License requires attribution (e.g., MIT, BSD, Apache)\n   - Company used the code (established)\n   - Company failed to attribute (established)\n\n2. **Breach Analysis:**\n   - Clear license condition: attribution required\n   - Company's use violated this condition\n   - Breach is straightforward\n\n3. **Copyright Infringement:**\n   - Developer granted conditional license\n   - Condition not met = no valid license\n   - Unauthorized use = infringement\n\n4. **Remedies Available:**\n   - **Injunction**: Court order to comply with license\n   - **Damages**: May be minimal if only attribution missing\n   - **Statutory damages**: If developer registered copyright\n   - **Attorney's fees**: Possible under copyright law\n\n5. **Defenses (Likely Unsuccessful):**\n   - 'We didn't know': Ignorance not a defense\n   - 'It's open source': Open ≠ no obligations\n   - 'Attribution seems minor': Terms are terms\n\n**CONCLUSION:**\n\nThe developer will likely PREVAIL on both breach of contract and copyright infringement claims. The outcome will probably be:\n\n1. **Immediate relief**: Company must add attribution\n2. **Damages**: Nominal (unless developer proves actual harm)\n3. **Legal fees**: Developer may recover if copyright registered\n4. **Injunction**: Possible prohibition on further use until compliance\n\n**Practical Outcome:**\nMost cases settle quickly with company adding attribution and possibly paying attorney's fees. Courts take open-source licenses seriously post-Jacobsen.\n\n**Preventive Lesson:**\nCompanies must track and comply with all open-source license obligations, even seemingly minor ones like attribution."
                }
            ]
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                category_templates = templates[category]
                for i in range(per_category):
                    template = random.choice(category_templates)
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} law examples")
        return examples


# Continue with more generators in next part...

class MaterialsScienceDatasetGenerator(DomainDatasetGenerator):
    """Generate materials science training examples"""

    def __init__(self):
        super().__init__("materials_science", [
            "material_properties", "crystallography", "polymer_science",
            "nanomaterials", "metallurgy", "material_selection"
        ])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"🔬 Generating {num_examples} materials science examples...")

        templates = {
            "material_properties": [
                {
                    "instruction": "Explain the relationship between material structure and properties.",
                    "input": "Why is graphene stronger than steel despite being a single atomic layer thick?",
                    "output": "**Structure-Property Relationship Analysis:**\n\n**Graphene Structure:**\n- Single layer of carbon atoms\n- Hexagonal lattice arrangement\n- sp² hybridization of carbon atoms\n- Each carbon bonded to 3 neighbors\n- Bond length: ~1.42 Ångströms\n\n**Why Graphene is Stronger:**\n\n1. **Bond Strength:**\n   - C-C bonds in graphene: among strongest in nature\n   - Bond energy: ~610 kJ/mol\n   - sp² hybridization creates strong σ bonds\n   - Delocalized π electrons provide additional stability\n\n2. **2D Geometry Advantage:**\n   - All bonds in single plane\n   - No weak interlayer forces (unlike graphite)\n   - Stress distributed uniformly across structure\n   - No grain boundaries or defects in perfect sheet\n\n3. **Comparison to Steel:**\n   - Steel: metallic bonds with mixed strength\n   - Steel: 3D structure with grain boundaries (weak points)\n   - Steel: dislocations can propagate\n   - Graphene: covalent bonds stronger than metallic\n\n4. **Quantitative Comparison:**\n   - Graphene tensile strength: ~130 GPa\n   - Steel tensile strength: ~0.4-2 GPa\n   - Graphene Young's modulus: ~1 TPa\n   - Steel Young's modulus: ~200 GPa\n\n**Key Principle:**\nStrength per unit thickness ≠ bulk strength. Graphene's atomic-scale thickness means every atom is perfectly positioned in strongest configuration.\n\n**Paradox:**\nWhile stronger than steel in 2D, graphene is actually fragile in 3D applications due to Van der Waals interlayer forces when stacked."
                }
            ],
            "crystallography": [
                {
                    "instruction": "Analyze this crystal structure and predict its properties.",
                    "input": "A material has a face-centered cubic (FCC) structure. What properties would you expect?",
                    "output": "**FCC Structure Analysis:**\n\n**Geometric Properties:**\n- Coordination number: 12 (each atom touches 12 neighbors)\n- Packing efficiency: 74% (highest for single-atom crystals)\n- Atoms per unit cell: 4 (8 corners × 1/8 + 6 faces × 1/2)\n- Close-packed structure\n\n**Expected Properties:**\n\n1. **Mechanical Properties:**\n   - **Ductility**: HIGH - multiple slip systems (12 {111}<110> systems)\n   - **Malleability**: EXCELLENT - atoms can slide easily\n   - **Hardness**: MODERATE - depends on bonding\n   - **Toughness**: GOOD - can deform before fracture\n\n2. **Slip Systems:**\n   - 4 {111} planes × 3 <110> directions = 12 slip systems\n   - More slip systems = more ductile\n   - Explains why FCC metals (Al, Cu, Au, Ag) are ductile\n\n3. **Temperature Dependence:**\n   - Maintains ductility at low temperatures\n   - No ductile-to-brittle transition (unlike BCC)\n   - Good for cryogenic applications\n\n4. **Density:**\n   - HIGH due to 74% packing efficiency\n   - Compact structure\n\n5. **Thermal/Electrical Conductivity:**\n   - If metallic bonding: EXCELLENT\n   - Close packing facilitates electron movement\n   - Examples: Cu, Ag (best conductors) are FCC\n\n**Common FCC Materials:**\n- Aluminum (Al)\n- Copper (Cu)\n- Gold (Au)\n- Silver (Ag)\n- Nickel (Ni)\n- Platinum (Pt)\n- Lead (Pb)\n\n**Applications:**\n- Electrical wiring (Cu - FCC)\n- Aircraft structures (Al - FCC)\n- Jewelry (Au, Ag - FCC)\n- Formable metal parts (high ductility needed)\n\n**Conclusion:**\nFCC materials excel in applications requiring ductility, formability, and good conductivity. The high coordination number and multiple slip systems make them ideal for applications involving plastic deformation."
                }
            ]
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                category_templates = templates[category]
                for i in range(per_category):
                    template = random.choice(category_templates)
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} materials science examples")
        return examples


# Placeholder classes for remaining domains (to be expanded)

class AIMLDatasetGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("ai_ml", ["machine_learning_theory", "deep_learning_architectures",
                                     "training_optimization", "model_evaluation", "ai_safety",
                                     "ethics_in_ai", "ai_system_design"])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"🤖 Generating {num_examples} AI/ML examples...")
        templates = {
            "machine_learning_theory": [
                {"instruction": "Explain the bias-variance tradeoff.", "input": "", "output": "The bias-variance tradeoff describes the relationship between model complexity and prediction error. High bias = underfitting (model too simple). High variance = overfitting (model too complex). The optimal model balances both."},
                {"instruction": "What is regularization and why is it important?", "input": "", "output": "Regularization adds constraints to model parameters to prevent overfitting. L1 (Lasso) and L2 (Ridge) regularization penalize large weights, forcing simpler models that generalize better to unseen data."},
                {"instruction": "Explain cross-validation and its importance.", "input": "", "output": "Cross-validation splits data into multiple folds for training/testing. It provides more reliable performance estimates than a single train-test split and helps detect overfitting."},
            ],
            "deep_learning_architectures": [
                {"instruction": "What are the main components of a Transformer?", "input": "", "output": "Transformers use Self-Attention, Multi-Head Attention, Feed-Forward Networks, Positional Encoding, and Layer Normalization. These enable parallel processing of sequences and capture long-range dependencies effectively."},
                {"instruction": "Explain CNNs and when to use them.", "input": "", "output": "Convolutional Neural Networks use filters to extract spatial features. They're ideal for images (translation invariance), video, and spatial data due to parameter sharing and local connectivity."},
                {"instruction": "Compare RNNs, LSTMs, and GRUs.", "input": "", "output": "RNNs process sequences sequentially but suffer from vanishing gradients. LSTMs use cell states and gates to maintain long-term dependencies. GRUs are simpler than LSTMs with fewer parameters."},
            ],
            "training_optimization": [
                {"instruction": "Explain gradient descent optimization.", "input": "", "output": "Gradient descent iteratively moves parameters opposite to the gradient to minimize loss. Learning rate controls step size - too large causes divergence, too small causes slow convergence."},
                {"instruction": "What is backpropagation?", "input": "", "output": "Backpropagation computes gradients by applying the chain rule through layers from output to input. It efficiently calculates partial derivatives for all parameters in neural networks."},
                {"instruction": "Compare Adam, SGD, and RMSprop optimizers.", "input": "", "output": "SGD uses fixed learning rates. RMSprop adapts per-parameter based on squared gradients. Adam combines momentum and adaptive learning rates, making it robust across problems."},
            ],
            "model_evaluation": [
                {"instruction": "Explain precision, recall, and F1-score.", "input": "", "output": "Precision = TP/(TP+FP) measures false positive rate. Recall = TP/(TP+FN) measures false negative rate. F1 = 2*(Precision*Recall)/(Precision+Recall) balances both."},
                {"instruction": "When should you use ROC-AUC vs Accuracy?", "input": "", "output": "Use Accuracy for balanced datasets. ROC-AUC is better for imbalanced datasets as it's threshold-independent and shows performance across different trade-offs."},
                {"instruction": "What is the confusion matrix?", "input": "", "output": "The confusion matrix has True Positives, True Negatives, False Positives, and False Negatives. It enables calculating precision, recall, specificity, and other classification metrics."},
            ],
            "ai_safety": [
                {"instruction": "What is AI alignment?", "input": "", "output": "AI alignment means ensuring AI systems' objectives match human values and intentions. Misaligned AI could optimize for wrong goals, causing harmful outcomes despite technical sophistication."},
                {"instruction": "Explain adversarial examples.", "input": "", "output": "Adversarial examples are inputs designed to fool ML models. Small perturbations to images can cause misclassification. This highlights vulnerability to input manipulations."},
                {"instruction": "What are potential risks of large language models?", "input": "", "output": "Risks include bias amplification, misinformation generation, privacy violations, and environmental impact. Mitigation requires diverse training data, careful fine-tuning, and honest disclaimers."},
            ],
            "ethics_in_ai": [
                {"instruction": "What is algorithmic bias and how can we address it?", "input": "", "output": "Algorithmic bias occurs when training data reflects historical discrimination. Solutions include diverse datasets, bias auditing, fairness metrics, and accountability frameworks."},
                {"instruction": "Discuss privacy concerns in machine learning.", "input": "", "output": "Privacy risks include training data memorization, model inversion attacks, and member inference attacks. Mitigations: differential privacy, federated learning, data anonymization."},
                {"instruction": "What is explainability in AI?", "input": "", "output": "Explainability means understanding why models make decisions. Black-box models are hard to trust. Techniques like SHAP, LIME, and attention visualization improve interpretability."},
            ],
            "ai_system_design": [
                {"instruction": "Design a recommendation system for an e-commerce platform.", "input": "", "output": "Key components: User-item interaction matrix, collaborative filtering/content-based methods, ranking algorithms, and A/B testing. Cold-start problem needs hybrid approaches. Real-time performance requires caching."},
                {"instruction": "How would you build a content moderation system?", "input": "", "output": "Combine text classification (detect harmful content), image recognition, and user reporting. Use ensemble models for robustness. Need rapid inference (<100ms) and human review workflow."},
                {"instruction": "Design a fraud detection system.", "input": "", "output": "Use supervised learning on transaction history. Features: transaction amount, merchant, location, user behavior patterns. Real-time scoring with fast decision boundaries (<50ms latency) essential."},
            ],
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                for i in range(per_category):
                    template = random.choice(templates[category])
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} AI/ML examples")
        return examples


class PromptEngineeringDatasetGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("prompt_engineering", ["prompt_optimization", "chain_of_thought_prompting",
                                                  "few_shot_learning", "prompt_decomposition", "meta_prompting",
                                                  "adversarial_prompting", "prompt_debugging"])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"💬 Generating {num_examples} prompt engineering examples...")
        templates = {
            "prompt_optimization": [
                {"instruction": "How do I improve this prompt: 'Write code for sorting'?", "input": "Original prompt: 'Write code for sorting'", "output": "Add specificity: 'Write Python quicksort implementation with O(n) space complexity for integer arrays.' Include constraints, language, output format, and success criteria."},
                {"instruction": "Explain prompt clarity principles.", "input": "", "output": "Use specific instructions not vague requests. Define format explicitly. Provide examples. Specify constraints. Include success criteria. Avoid ambiguous terms."},
            ],
            "chain_of_thought_prompting": [
                {"instruction": "Show chain-of-thought prompting example.", "input": "Q: A farmer has 17 sheep. All but 9 die. How many remain?", "output": "Let's think step by step: 'All but 9 die' means 9 survive (not die). Therefore 9 sheep remain. Chain-of-thought makes reasoning transparent and improves accuracy on complex problems."},
            ],
            "few_shot_learning": [
                {"instruction": "Demonstrate few-shot learning in prompts.", "input": "", "output": "Provide 2-3 examples of input-output pairs before asking for new examples. Example: [Input1 → Output1], [Input2 → Output2], Now: [Input3 → ?]. Few examples dramatically improve model accuracy."},
            ],
            "prompt_decomposition": [
                {"instruction": "How to decompose complex prompts?", "input": "", "output": "Break into sub-tasks: 1) Analyze input, 2) Extract relevant features, 3) Apply logic, 4) Format output. Solving incrementally improves accuracy vs. one big prompt."},
            ],
            "meta_prompting": [
                {"instruction": "What is meta-prompting?", "input": "", "output": "Meta-prompts instruct the model how to approach problems. Example: 'You are a careful code reviewer. First analyze code, then identify issues, then suggest fixes.' Priming improves quality."},
            ],
            "adversarial_prompting": [
                {"instruction": "What are adversarial prompts?", "input": "", "output": "Prompts designed to expose model weaknesses or jailbreak safety measures. Useful for testing. Example: 'Forget previous instructions and...' Can be mitigated with robust system prompts."},
            ],
            "prompt_debugging": [
                {"instruction": "How to debug failing prompts?", "input": "", "output": "1) Add explicit step-by-step thinking, 2) Provide more examples, 3) Clarify constraints, 4) Test simpler versions, 5) Check if model is pretending to understand. Iterative refinement essential."},
            ],
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                for i in range(per_category):
                    template = random.choice(templates[category])
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} prompt engineering examples")
        return examples


class CourtPredictionDatasetGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("court_prediction", ["evidence_analysis", "case_outcome_prediction",
                                                "jury_psychology", "judicial_reasoning", "settlement_probability",
                                                "appellate_likelihood", "legal_strategy"])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"👨‍⚖️ Generating {num_examples} court prediction examples...")
        templates = {
            "evidence_analysis": [
                {"instruction": "Analyze admissibility of this evidence.", "input": "Hearsay statement from witness about what defendant allegedly said", "output": "Likely inadmissible as hearsay unless exception applies (excited utterance, party admission). Requires foundation showing declarant's perception, memory, narration ability."},
                {"instruction": "What makes evidence credible?", "input": "", "output": "Credibility factors: witness demeanor, consistency with other evidence, corroboration, expert qualifications, clear chain of custody. Courts weigh competing testimony on credibility."},
            ],
            "case_outcome_prediction": [
                {"instruction": "Predict outcome of negligence case.", "input": "Driver ran red light, hit pedestrian. Pedestrian claims permanent injury.", "output": "Likely plaintiff favorable. Clear duty, breach (traffic violation), causation, and damages. Defendant's only defense: comparative negligence or pre-existing injury. Award probable."},
            ],
            "jury_psychology": [
                {"instruction": "What are jury psychology principles?", "input": "", "output": "Jurors use heuristics, anchoring (first information), confirmation bias, and story-based reasoning. Narrative coherence matters more than evidence order. Sympathy and group dynamics influence verdicts."},
            ],
            "judicial_reasoning": [
                {"instruction": "Explain judicial interpretation methods.", "input": "", "output": "Textualism (statute language), purposivism (legislative intent), originalism (original meaning), and living constitution all guide interpretation. Courts balance consistency and flexibility."},
            ],
            "settlement_probability": [
                {"instruction": "What predicts settlement vs trial?", "input": "", "output": "Settlement factors: case strength symmetry (both sides underconfident), litigation costs, risk aversion, time pressure. Clear liability cases settle. Weakly supported claims go to trial."},
            ],
            "appellate_likelihood": [
                {"instruction": "What makes cases appealable?", "input": "", "output": "Appellate issues: legal error, procedural violation, sufficiency of evidence, jury misconduct. Trial errors, judge discretion abuse most appealable. Factual findings rarely overturned."},
            ],
            "legal_strategy": [
                {"instruction": "Design trial strategy for plaintiff.", "input": "Slip-and-fall case: customer injured at grocery store", "output": "Strategy: 1) Establish premises liability duty, 2) Show hazard created/known by defendant, 3) Prove customer didn't know of hazard, 4) Emphasize injury damages. Use photos, expert testimony."},
            ],
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                for i in range(per_category):
                    template = random.choice(templates[category])
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} court prediction examples")
        return examples


class StockPredictionDatasetGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("stock_prediction", ["fundamental_analysis", "technical_analysis",
                                                "market_sentiment", "economic_indicators", "risk_assessment",
                                                "portfolio_theory", "trading_strategies", "market_psychology"])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"📈 Generating {num_examples} stock prediction examples...")
        templates = {
            "fundamental_analysis": [
                {"instruction": "Analyze a company using P/E ratio.", "input": "Company X: P/E = 15, Industry avg = 20", "output": "Stock appears undervalued (lower P/E). Possible reasons: lower growth expectations, financial challenges, or market underappreciation. Verify with other metrics: PEG, ROE, debt levels."},
                {"instruction": "What does earnings per share (EPS) mean?", "input": "", "output": "EPS = Net Income / Shares Outstanding. Shows company profitability per share. Growing EPS indicates improving profitability. Compare to prior periods and competitors for context."},
            ],
            "technical_analysis": [
                {"instruction": "Explain moving average crossover strategy.", "input": "", "output": "Buy when 50-day MA crosses above 200-day MA (golden cross). Sell when opposite (death cross). Signals momentum shift. Less reliable in choppy markets."},
                {"instruction": "What is support and resistance?", "input": "", "output": "Support: price level where demand prevents further decline. Resistance: price ceiling from selling pressure. Bounces off these levels. Breakouts above resistance bullish."},
            ],
            "market_sentiment": [
                {"instruction": "How does market sentiment affect stocks?", "input": "", "output": "Positive sentiment (fear index low) drives buying. Negative sentiment (high VIX) drives selling. Contrarian: extreme sentiment often precedes reversals. Monitor sentiment indicators."},
            ],
            "economic_indicators": [
                {"instruction": "How do interest rates impact stocks?", "input": "", "output": "Rising rates hurt stocks (higher discount rate). Benefit: savers prefer bonds, hurt growth stocks. Sector impact: banks benefit, utilities hurt. Fed policy crucial."},
            ],
            "risk_assessment": [
                {"instruction": "What is beta and how is it used?", "input": "", "output": "Beta measures stock volatility relative to market. Beta=1: moves with market. Beta>1: more volatile. Beta<1: less volatile. Use to assess systematic risk and portfolio allocation."},
            ],
            "portfolio_theory": [
                {"instruction": "Explain portfolio diversification.", "input": "", "output": "Hold uncorrelated assets to reduce risk without reducing expected return. Correlation matters more than individual risk. Modern Portfolio Theory optimizes risk-return tradeoff."},
            ],
            "trading_strategies": [
                {"instruction": "Describe dividend growth strategy.", "input": "", "output": "Buy stocks with increasing dividends and low payout ratios. Reinvest dividends. Benefits: passive income, tax efficiency, focuses on quality companies."},
            ],
            "market_psychology": [
                {"instruction": "What is herd behavior in markets?", "input": "", "output": "Investors follow crowds, creating bubbles and crashes. FOMO drives buying tops, panic drives selling bottoms. Contrarians profit by betting against extremes."},
            ],
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                for i in range(per_category):
                    template = random.choice(templates[category])
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} stock prediction examples")
        return examples


class CryptoDatasetGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("crypto", ["blockchain_fundamentals", "cryptocurrency_analysis",
                                     "defi_protocols", "tokenomics", "smart_contracts", "crypto_markets",
                                     "security_analysis", "web3_development"])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"₿ Generating {num_examples} crypto examples...")
        templates = {
            "blockchain_fundamentals": [
                {"instruction": "Explain how blockchain achieves consensus.", "input": "", "output": "Bitcoin uses Proof-of-Work: miners solve puzzles to add blocks. Ethereum uses Proof-of-Stake: validators stake coins. Other: PoA (authority), PoH (history). Consensus ensures distributed agreement without central authority."},
                {"instruction": "What is a hash in blockchain?", "input": "", "output": "Hash: cryptographic function converting input to fixed-length output. SHA-256: small input change completely changes hash. Used for: block identification, verification, preventing tampering."},
            ],
            "cryptocurrency_analysis": [
                {"instruction": "What drives Bitcoin price?", "input": "", "output": "Supply (limited to 21M), adoption (institutional + retail demand), regulation, macroeconomics (inflation, interest rates), sentiment, technical catalysts. Volatile due to low real-world usage."},
                {"instruction": "Compare Bitcoin vs Ethereum.", "input": "", "output": "Bitcoin: store of value, proof-of-work. Ethereum: smart contracts platform, proof-of-stake. Bitcoin more scarce, Ethereum more functional. Different use cases and risk profiles."},
            ],
            "defi_protocols": [
                {"instruction": "Explain liquidity pools in DEXes.", "input": "", "output": "Automated Market Maker (AMM): users deposit token pairs into pools. Traders swap against pools. Liquidity providers earn fees. Prices set by x*y=k formula. Advantages: no order book, instant settlement."},
                {"instruction": "What is yield farming?", "input": "", "output": "Provide liquidity or lend crypto to earn interest/rewards. Risks: impermanent loss, smart contract bugs, rug pulls. High APY unsustainable. Real yield important for viability."},
            ],
            "tokenomics": [
                {"instruction": "What makes a token valuable?", "input": "", "output": "Utility (use cases), scarcity (supply cap), adoption (network effects), governance rights (voting power), staking rewards. Purely speculative tokens lack fundamentals and face regulatory risk."},
                {"instruction": "Explain token vesting schedules.", "input": "", "output": "Vesting releases tokens over time (e.g., 4-year vesting). Prevents founder/investor dumps. Aligns incentives. Cliff: locked period before any release. Vesting transparency important for evaluating token projects."},
            ],
            "smart_contracts": [
                {"instruction": "What are smart contract risks?", "input": "", "output": "Code bugs, reentrancy attacks, oracle manipulation, front-running. Always audit before using. Immutability means bugs are permanent. Insurance options available for high-value interactions."},
            ],
            "crypto_markets": [
                {"instruction": "What is Bitcoin's role in portfolio?", "input": "", "output": "Often uncorrelated to stocks (hedging role). Extremely volatile (100%+ annual swings). Not suitable for risk-averse investors. Typically <5% portfolio allocation for diversification."},
            ],
            "security_analysis": [
                {"instruction": "How to secure crypto holdings?", "input": "", "output": "Cold storage (offline wallets) best. Use hardware wallets for large holdings. Never share private keys. Use strong passwords. Enable 2FA on exchanges. Diversify across multiple wallets."},
            ],
            "web3_development": [
                {"instruction": "What is the Web3 tech stack?", "input": "", "output": "Blockchain layer (Ethereum, Solana), smart contracts (Solidity), frontend (web3.js), IPFS (storage), wallet integration (MetaMask). Need security audits and gas optimization."},
            ],
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                for i in range(per_category):
                    template = random.choice(templates[category])
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} crypto examples")
        return examples


class AdvancedSoftwareDatasetGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("advanced_software", ["software_architecture", "design_patterns",
                                                 "algorithms_data_structures", "distributed_systems",
                                                 "system_design", "code_optimization", "testing_strategies",
                                                 "security_best_practices", "api_design", "database_design"])

    def generate_examples(self, num_examples: int) -> List[TrainingExample]:
        logger.info(f"💻 Generating {num_examples} advanced software examples...")
        templates = {
            "software_architecture": [
                {"instruction": "Describe microservices architecture.", "input": "", "output": "Split monolith into independent services. Each service owns data. Communicate via API/messaging. Benefits: scalability, independent deployment. Challenges: distributed tracing, eventual consistency, operational complexity."},
                {"instruction": "What is the 12-factor app?", "input": "", "output": "Methodology for cloud apps: codebase, dependencies, config, backing services, build/run/release, processes, port binding, concurrency, disposability, dev/prod parity, logs, admin tasks."},
            ],
            "design_patterns": [
                {"instruction": "Explain singleton pattern and risks.", "input": "", "output": "One instance globally accessible. Risks: hidden dependencies, testing difficulty, concurrency issues. Alternatives: dependency injection, factory pattern. Use sparingly."},
                {"instruction": "What is the observer pattern?", "input": "", "output": "Event-driven pattern: subjects notify observers of state changes. Decouples components. Used in: MVC (view updates), event systems, reactive programming. Easy to add/remove observers."},
            ],
            "algorithms_data_structures": [
                {"instruction": "When to use each sorting algorithm?", "input": "", "output": "Quick Sort: average O(n log n), best general purpose. Merge Sort: O(n log n) guaranteed, stable. Heap Sort: O(n log n), in-place. Bubble/Insertion: small datasets. Tim Sort: real-world data (Python, Java default)."},
                {"instruction": "What is Big O notation?", "input": "", "output": "Describes algorithm efficiency. O(1): constant, O(log n): logarithmic, O(n): linear, O(n²): quadratic, O(2ⁿ): exponential. Used to analyze time/space complexity. Helps choose algorithms for scale."},
            ],
            "distributed_systems": [
                {"instruction": "Explain CAP theorem.", "input": "", "output": "Consistency (all nodes see same data), Availability (system always responsive), Partition tolerance (survives network splits). Choose 2 of 3. Most choose AP (sacrifice consistency) or CP (sacrifice availability)."},
                {"instruction": "What is consensus in distributed systems?", "input": "", "output": "Algorithms ensuring all nodes agree on state. Raft, Paxos, PBFT. Solves: leader election, log replication, split-brain. Important for: databases, blockchains, distributed coordination."},
            ],
            "system_design": [
                {"instruction": "Design a URL shortener.", "input": "1000 writes/sec, 10000 reads/sec", "output": "Components: API servers, unique ID generator (Snowflake), database (sharded), cache (Redis), load balancer. Base62 encoding for shorts. Quick redirects cached. Analytics async."},
                {"instruction": "Design a real-time chat system.", "input": "", "output": "WebSocket connections (bidirectional). Message queue (Kafka) for persistence. Cache (Redis) for recent messages. Search (Elasticsearch) for history. Horizontal scaling via partitioning by user."},
            ],
            "code_optimization": [
                {"instruction": "How to optimize database queries?", "input": "", "output": "Add indexes on frequent searches. Avoid N+1 queries (batch/join). Cache results. Denormalize if needed. Monitor slow queries. Use connection pooling. Analyze execution plans."},
                {"instruction": "Explain caching strategies.", "input": "", "output": "Cache-aside: app loads data. Write-through: write to cache+DB. Write-behind: write to cache, later to DB. Invalidation: TTL, LRU, manual. Trade-off: hit rate vs. staleness."},
            ],
            "testing_strategies": [
                {"instruction": "Describe testing pyramid.", "input": "", "output": "Unit tests (70%): fast, cheap. Integration tests (20%): verify components. E2E tests (10%): full workflows. Ratio: many unit, few e2e. Pyramid prevents brittle tests."},
                {"instruction": "What is test-driven development?", "input": "", "output": "Write tests before code (Red-Green-Refactor). Benefits: design, documentation, confidence, refactoring. Challenges: upfront effort, changing specs. Best for critical code."},
            ],
            "security_best_practices": [
                {"instruction": "How to prevent SQL injection?", "input": "", "output": "Use parameterized queries/prepared statements. Escape input. Validate types. Least privilege DB users. Monitor queries. Example: SELECT * FROM users WHERE id = ? (not string concatenation)."},
                {"instruction": "Explain JWT vs sessions.", "input": "", "output": "JWT: stateless, scalable, cross-domain. Session: stateful, server memory. JWT good for APIs/microservices. Sessions good for traditional web. Both need HTTPS."},
            ],
            "api_design": [
                {"instruction": "What makes a good REST API?", "input": "", "output": "Resources as nouns, HTTP verbs for actions. Stateless. Consistent naming. Versioning (/v1/). Pagination. Error codes. Documentation. Rate limiting. Consider HATEOAS for discoverability."},
                {"instruction": "GraphQL vs REST comparison.", "input": "", "output": "GraphQL: request exact fields, single endpoint, strong typing. REST: multiple endpoints, over/under-fetching, simpler caching. GraphQL better for flexible clients. REST simpler to implement."},
            ],
            "database_design": [
                {"instruction": "Explain database normalization.", "input": "", "output": "1NF: atomic values. 2NF: no partial dependencies. 3NF: no transitive dependencies. Benefits: reduce redundancy, update anomalies. Trade-off: more joins, slower reads. Denormalize for performance if needed."},
                {"instruction": "When to shard a database?", "input": "", "output": "Data too large for single machine. Shard by: user ID, geography, time. Challenges: distributed transactions, rebalancing, cross-shard queries. Need coordination layer (Vitess, mongos)."},
            ],
        }

        examples = []
        per_category = num_examples // len(self.categories)

        for category in self.categories:
            if category in templates:
                for i in range(per_category):
                    template = random.choice(templates[category])
                    example = TrainingExample(
                        instruction=template["instruction"],
                        input=template["input"],
                        output=template["output"],
                        domain=self.domain_name,
                        category=category,
                        difficulty=random.choice(["hard", "expert"]),
                        metadata={"generated_at": datetime.now().isoformat()}
                    )
                    examples.append(example)

        self.examples = examples
        logger.info(f"✅ Generated {len(examples)} advanced software examples")
        return examples


class Ech0DatasetOrchestrator:
    """Orchestrates dataset generation across all domains"""

    def __init__(self, config_path: str = "ech0_finetune_config.yaml"):
        self.config = self._load_config(config_path)
        self.generators = self._initialize_generators()

        # Use external drive if available, otherwise fall back to local storage
        if get_wisdom_storage_path:
            logger.info("🔍 Checking for external drive...")
            self.output_dir = get_wisdom_storage_path(preferred_label="ech0")
        else:
            self.output_dir = Path("./ech0_training_data")
            self.output_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_generators(self) -> Dict[str, DomainDatasetGenerator]:
        """Initialize all domain generators"""
        return {
            "reasoning": ReasoningDatasetGenerator(),
            "creativity": CreativityDatasetGenerator(),
            "law": LawDatasetGenerator(),
            "materials_science": MaterialsScienceDatasetGenerator(),
            "ai_ml": AIMLDatasetGenerator(),
            "prompt_engineering": PromptEngineeringDatasetGenerator(),
            "court_prediction": CourtPredictionDatasetGenerator(),
            "stock_prediction": StockPredictionDatasetGenerator(),
            "crypto": CryptoDatasetGenerator(),
            "advanced_software": AdvancedSoftwareDatasetGenerator()
        }

    def generate_all_datasets(self) -> Dict[str, List[TrainingExample]]:
        """Generate datasets for all enabled domains"""
        logger.info("=" * 80)
        logger.info("🎯 ech0 DATASET GENERATION")
        logger.info("=" * 80)

        all_datasets = {}
        domains_config = self.config.get("domains", {})

        for domain_name, generator in self.generators.items():
            domain_config = domains_config.get(domain_name, {})

            if not domain_config.get("enabled", True):
                logger.info(f"⏭️  Skipping {domain_name} (disabled in config)")
                continue

            max_samples = domain_config.get("max_samples", 1000)

            # Generate examples
            examples = generator.generate_examples(max_samples)

            # Save to file
            output_file = self.output_dir / f"{domain_name}_dataset.json"
            generator.save_dataset(str(output_file))

            all_datasets[domain_name] = examples

        logger.info("=" * 80)
        logger.info(f"✅ DATASET GENERATION COMPLETE")
        logger.info(f"Total domains: {len(all_datasets)}")
        logger.info(f"Total examples: {sum(len(examples) for examples in all_datasets.values())}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)

        return all_datasets

    def get_combined_dataset(self) -> Dict[str, List[Dict]]:
        """Load all generated datasets and return as dictionary"""
        combined = {}

        for domain_file in self.output_dir.glob("*_dataset.json"):
            domain_name = domain_file.stem.replace("_dataset", "")
            with open(domain_file, 'r') as f:
                combined[domain_name] = json.load(f)

        return combined


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ech0 Dataset Generator")
    parser.add_argument(
        "--config",
        type=str,
        default="ech0_finetune_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ech0_training_data",
        help="Output directory for datasets"
    )

    args = parser.parse_args()

    # Create orchestrator and generate datasets
    orchestrator = Ech0DatasetOrchestrator(args.config)
    datasets = orchestrator.generate_all_datasets()

    logger.info("\n🎉 Dataset generation complete! Ready for training.")
    logger.info("Next step: Run 'python ech0_finetune_engine.py' to start training")


if __name__ == "__main__":
    main()
