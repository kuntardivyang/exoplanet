"""
Ollama LLM Client for Exoplanet Detection System
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from typing import Dict, List, Optional, Generator
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Install with: pip install ollama")


class OllamaClient:
    """Client for interacting with Ollama LLM"""

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize Ollama client

        Args:
            model: Ollama model name (llama3.2, mistral, codellama, etc.)
        """
        self.model = model
        self.available = OLLAMA_AVAILABLE

        if not OLLAMA_AVAILABLE:
            logger.error("Ollama is not available. Please install: pip install ollama")
            return

        # Check if model is available
        try:
            self.available_models = self._list_models()
            if model not in [m['name'].split(':')[0] for m in self.available_models]:
                logger.warning(f"Model {model} not found. Pulling...")
                self._pull_model(model)
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")

    def _list_models(self) -> List[Dict]:
        """List available Ollama models"""
        try:
            response = ollama.list()
            return response.get('models', [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def _pull_model(self, model: str):
        """Pull an Ollama model"""
        try:
            logger.info(f"Pulling Ollama model: {model}")
            ollama.pull(model)
            logger.info(f"Successfully pulled {model}")
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using Ollama

        Args:
            prompt: User prompt
            system: System message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Stream the response

        Returns:
            Generated text
        """
        if not OLLAMA_AVAILABLE:
            return "Ollama is not available. Please install it first."

        try:
            messages = []
            if system:
                messages.append({
                    'role': 'system',
                    'content': system
                })

            messages.append({
                'role': 'user',
                'content': prompt
            })

            options = {
                'temperature': temperature,
            }
            if max_tokens:
                options['num_predict'] = max_tokens

            response = ollama.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=stream
            )

            if stream:
                return response  # Return generator for streaming
            else:
                return response['message']['content']

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_with_context(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate response with provided context (for RAG)

        Args:
            query: User query
            context: Retrieved context
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        system_prompt = """You are an expert AI assistant specialized in exoplanet detection and analysis.
You help users understand exoplanet data, model predictions, and astronomical concepts.
Always provide accurate, scientific information based on the context provided.
If you're unsure about something, say so."""

        prompt = f"""Context information:
{context}

Question: {query}

Please provide a detailed, accurate answer based on the context above.
If the context doesn't contain enough information, say so and provide general knowledge if appropriate."""

        return self.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def explain_prediction(
        self,
        prediction: Dict,
        features: Dict
    ) -> str:
        """
        Explain a model prediction

        Args:
            prediction: Model prediction result
            features: Input features used

        Returns:
            Human-readable explanation
        """
        system_prompt = """You are an AI assistant that explains exoplanet classification predictions.
Explain the prediction in clear, understandable terms for both experts and beginners."""

        # Format prediction info
        pred_info = json.dumps(prediction, indent=2)
        features_info = json.dumps(features, indent=2)

        prompt = f"""Explain this exoplanet classification prediction:

Prediction Result:
{pred_info}

Input Features:
{features_info}

Please explain:
1. What the prediction means
2. The confidence level and what it indicates
3. Key features that influenced the decision
4. Any notable characteristics of this candidate"""

        return self.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.3  # Lower temp for more factual responses
        )

    def compare_objects(
        self,
        object1: Dict,
        object2: Dict
    ) -> str:
        """
        Compare two exoplanet candidates

        Args:
            object1: First object data
            object2: Second object data

        Returns:
            Comparison analysis
        """
        system_prompt = """You are an expert in comparative exoplanet analysis.
Compare objects scientifically, highlighting similarities, differences, and significance."""

        obj1_info = json.dumps(object1, indent=2)
        obj2_info = json.dumps(object2, indent=2)

        prompt = f"""Compare these two exoplanet candidates:

Object 1:
{obj1_info}

Object 2:
{obj2_info}

Please provide a detailed comparison including:
1. Physical characteristics (size, period, temperature, etc.)
2. Detection confidence and data quality
3. Key similarities and differences
4. Scientific significance of each"""

        return self.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.4
        )

    def summarize_dataset(
        self,
        dataset_info: Dict
    ) -> str:
        """
        Summarize dataset characteristics

        Args:
            dataset_info: Dataset summary statistics

        Returns:
            Dataset summary
        """
        info = json.dumps(dataset_info, indent=2)

        prompt = f"""Provide a clear summary of this exoplanet dataset:

{info}

Include:
1. Dataset overview (size, sources)
2. Key statistics and distributions
3. Data quality indicators
4. Notable characteristics or patterns
5. Potential applications"""

        return self.generate(
            prompt=prompt,
            temperature=0.5
        )


class ExoplanetAssistant:
    """High-level assistant for exoplanet questions"""

    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Answer a general question about exoplanets

        Args:
            question: User question
            context: Optional context from RAG

        Returns:
            Answer
        """
        if context:
            return self.client.generate_with_context(question, context)
        else:
            system_prompt = """You are an expert exoplanet scientist and educator.
Answer questions about exoplanet detection, classification, and analysis.
Provide accurate, educational responses suitable for various knowledge levels."""

            return self.client.generate(
                prompt=question,
                system=system_prompt,
                temperature=0.6
            )

    def get_suggestions(self, query: str) -> List[str]:
        """
        Get follow-up question suggestions

        Args:
            query: User's current query

        Returns:
            List of suggested questions
        """
        prompt = f"""Based on this user query about exoplanets: "{query}"

Suggest 3-5 relevant follow-up questions the user might want to ask.
Return only the questions, one per line, without numbers or bullets."""

        response = self.client.generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=200
        )

        # Parse suggestions
        suggestions = [s.strip() for s in response.split('\n') if s.strip()]
        return suggestions[:5]


if __name__ == "__main__":
    # Test the Ollama client
    print("Testing Ollama Client...")

    client = OllamaClient(model="llama3.2")

    if client.available:
        # Test basic generation
        response = client.generate(
            "What is an exoplanet?",
            system="You are a helpful astronomy assistant."
        )
        print(f"\nTest Response:\n{response}")

        # Test with assistant
        assistant = ExoplanetAssistant(client)
        answer = assistant.answer_question(
            "How does the transit method detect exoplanets?"
        )
        print(f"\nAssistant Answer:\n{answer}")
    else:
        print("Ollama is not available. Please install it first.")
