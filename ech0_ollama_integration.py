#!/usr/bin/env python3
"""
ech0 Ollama Integration
Generate high-quality training examples using local Ollama LLMs
"""

import json
import logging
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests library not installed. Install with: pip install requests")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    base_url: str = "http://localhost:11434"
    model: str = "mistral"  # or "llama2", "codellama", "mixtral", etc.
    timeout: int = 300  # 5 minutes default (increased from 120)
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    temperature: float = 0.7
    num_ctx: int = 4096  # context window size
    stream: bool = False  # set to True for streaming responses


class OllamaGenerator:
    """Generate training examples using Ollama local LLM"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama generator"""
        self.config = config or OllamaConfig()
        self.base_url = self.config.base_url.rstrip('/')

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for Ollama integration")

        # Check if Ollama is available
        self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"✓ Ollama connected. Available models: {', '.join(model_names)}")

                # Check if configured model is available
                if not any(self.config.model in name for name in model_names):
                    logger.warning(f"⚠️  Model '{self.config.model}' not found. Available: {', '.join(model_names)}")
                    logger.warning(f"   Download with: ollama pull {self.config.model}")

                return True
            else:
                logger.error(f"❌ Ollama returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Cannot connect to Ollama at {self.base_url}")
            logger.error("   Make sure Ollama is running: 'ollama serve'")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking Ollama: {e}")
            return False

    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate a completion using Ollama

        Args:
            prompt: The prompt to complete
            system_prompt: Optional system prompt for context
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text or None if failed
        """
        temp = temperature if temperature is not None else self.config.temperature

        # Build the full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": self.config.stream,
            "options": {
                "temperature": temp,
                "num_ctx": self.config.num_ctx,
            }
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Generating completion (attempt {attempt + 1}/{self.config.max_retries})...")

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    completion = result.get('response', '').strip()

                    if completion:
                        logger.debug(f"✓ Generated {len(completion)} characters")
                        return completion
                    else:
                        logger.warning("Empty response from Ollama")

                else:
                    logger.warning(f"Ollama returned status {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"⏱️  Timeout on attempt {attempt + 1}/{self.config.max_retries}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"   Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)

            except requests.exceptions.ConnectionError as e:
                logger.error(f"❌ Connection error: {e}")
                logger.error("   Make sure Ollama is running: 'ollama serve'")
                return None

            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        logger.error("❌ All retry attempts failed")
        return None

    def generate_training_example(
        self,
        domain: str,
        category: str,
        difficulty: str = "medium"
    ) -> Optional[Dict[str, str]]:
        """
        Generate a training example for a specific domain

        Args:
            domain: Domain name (e.g., "reasoning", "creativity", "law")
            category: Category within domain
            difficulty: Difficulty level

        Returns:
            Dict with 'instruction', 'input', 'output' keys or None
        """
        system_prompt = f"""You are generating high-quality training data for an AI system.
Generate examples that are:
- Accurate and factually correct
- Detailed and comprehensive
- Educational and informative
- Well-structured and clear

Domain: {domain}
Category: {category}
Difficulty: {difficulty}"""

        prompt = f"""Generate a training example in the following format:

INSTRUCTION: [A clear instruction or question]
INPUT: [Optional context or input data, leave blank if not needed]
OUTPUT: [Detailed, accurate response]

Make the example challenging and educational for the {difficulty} difficulty level.
Focus on {category} within the {domain} domain.

Generate one complete example now:"""

        response = self.generate_completion(prompt, system_prompt=system_prompt)

        if not response:
            return None

        # Parse the response
        try:
            instruction = ""
            input_text = ""
            output = ""

            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()

                if line.upper().startswith('INSTRUCTION:'):
                    current_section = 'instruction'
                    instruction = line.split(':', 1)[1].strip()
                elif line.upper().startswith('INPUT:'):
                    current_section = 'input'
                    input_text = line.split(':', 1)[1].strip()
                elif line.upper().startswith('OUTPUT:'):
                    current_section = 'output'
                    output = line.split(':', 1)[1].strip()
                elif current_section and line:
                    if current_section == 'instruction':
                        instruction += ' ' + line
                    elif current_section == 'input':
                        input_text += ' ' + line
                    elif current_section == 'output':
                        output += ' ' + line

            if instruction and output:
                return {
                    'instruction': instruction.strip(),
                    'input': input_text.strip(),
                    'output': output.strip()
                }
            else:
                logger.warning("Could not parse Ollama response into instruction/output format")
                return None

        except Exception as e:
            logger.error(f"Error parsing Ollama response: {e}")
            return None

    def generate_batch(
        self,
        domain: str,
        category: str,
        num_examples: int,
        difficulty: str = "medium",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, str]]:
        """
        Generate a batch of training examples

        Args:
            domain: Domain name
            category: Category within domain
            num_examples: Number of examples to generate
            difficulty: Difficulty level
            progress_callback: Optional callback function(current, total)

        Returns:
            List of training examples
        """
        examples = []

        logger.info(f"🤖 Generating {num_examples} examples using Ollama ({self.config.model})...")

        for i in range(num_examples):
            if progress_callback:
                progress_callback(i + 1, num_examples)

            example = self.generate_training_example(domain, category, difficulty)

            if example:
                examples.append(example)
                logger.debug(f"  [{i+1}/{num_examples}] Generated: {example['instruction'][:60]}...")
            else:
                logger.warning(f"  [{i+1}/{num_examples}] Failed to generate example")

            # Small delay to avoid overwhelming Ollama
            if i < num_examples - 1:
                time.sleep(0.5)

        logger.info(f"✓ Successfully generated {len(examples)}/{num_examples} examples")
        return examples


def check_ollama_status() -> Dict[str, Any]:
    """
    Check Ollama status and return info

    Returns:
        Dict with status information
    """
    if not REQUESTS_AVAILABLE:
        return {
            'available': False,
            'error': 'requests library not installed'
        }

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return {
                'available': True,
                'models': models,
                'url': 'http://localhost:11434'
            }
        else:
            return {
                'available': False,
                'error': f'Ollama returned status {response.status_code}'
            }
    except requests.exceptions.ConnectionError:
        return {
            'available': False,
            'error': 'Cannot connect to Ollama. Make sure it is running (ollama serve)'
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test Ollama integration
    print("\n" + "="*80)
    print("ech0 Ollama Integration Test")
    print("="*80 + "\n")

    status = check_ollama_status()

    if status['available']:
        print(f"✓ Ollama is running at {status['url']}")
        print(f"✓ Available models: {', '.join(status['models'])}\n")

        # Test generation
        config = OllamaConfig(model="mistral", timeout=300)
        generator = OllamaGenerator(config)

        print("Testing example generation...")
        example = generator.generate_training_example(
            domain="reasoning",
            category="logical_reasoning",
            difficulty="medium"
        )

        if example:
            print("\n" + "="*80)
            print("Sample Generated Example:")
            print("="*80)
            print(f"\nInstruction: {example['instruction']}")
            if example['input']:
                print(f"Input: {example['input']}")
            print(f"Output: {example['output']}")
            print("\n" + "="*80)
        else:
            print("\n❌ Failed to generate example")
    else:
        print(f"❌ Ollama not available: {status['error']}")
        print("\nTo use Ollama:")
        print("  1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  2. Start: ollama serve")
        print("  3. Pull model: ollama pull mistral")
