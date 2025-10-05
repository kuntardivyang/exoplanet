"""
RAG (Retrieval Augmented Generation) System for Exoplanet Assistant
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Optional
import json
from backend.llm.ollama_client import OllamaClient, ExoplanetAssistant
from backend.llm.vector_store import VectorStore, KnowledgeIndexer
from backend.utils.logger import setup_logger
from backend.config.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = setup_logger(__name__)


class RAGSystem:
    """Complete RAG system combining vector search and LLM"""

    def __init__(
        self,
        ollama_model: str = "llama3.2",
        vector_collection: str = "exoplanet_knowledge"
    ):
        """
        Initialize RAG system

        Args:
            ollama_model: Ollama model to use
            vector_collection: Vector store collection name
        """
        logger.info("Initializing RAG system...")

        # Initialize components
        self.ollama = OllamaClient(model=ollama_model)
        self.vector_store = VectorStore(collection_name=vector_collection)
        self.indexer = KnowledgeIndexer(self.vector_store)
        self.assistant = ExoplanetAssistant(self.ollama)

        self.available = self.ollama.available and self.vector_store.available

        if not self.available:
            logger.warning("RAG system not fully available. Check Ollama and ChromaDB installation.")

        logger.info(f"RAG system initialized. Available: {self.available}")

    def index_knowledge_base(self):
        """Index all available knowledge into vector store"""
        logger.info("Indexing knowledge base...")

        # Index exoplanet facts
        logger.info("Indexing exoplanet facts...")
        self.indexer.index_exoplanet_facts()

        # Index model results
        results_files = list(MODELS_DIR.glob("training_results_*.json"))
        for results_file in results_files:
            logger.info(f"Indexing {results_file.name}...")
            self.indexer.index_model_results(results_file)

        # Index dataset information
        try:
            from backend.utils.data_loader import DatasetLoader
            loader = DatasetLoader()
            datasets_info = [
                {'name': 'kepler', 'info': loader.get_dataset_info(loader.load_kepler(), 'kepler')},
                {'name': 'tess', 'info': loader.get_dataset_info(loader.load_tess(), 'tess')},
                {'name': 'k2', 'info': loader.get_dataset_info(loader.load_k2(), 'k2')}
            ]

            for ds in datasets_info:
                self.indexer.index_dataset_info(ds['name'], ds['info'])
        except Exception as e:
            logger.warning(f"Could not index datasets: {e}")

        # Index light curves
        try:
            from backend.utils.light_curve_loader import LightCurveLoader
            lc_loader = LightCurveLoader()
            summary = lc_loader.get_dataset_summary()
            summaries = summary.to_dict('records')
            self.indexer.index_light_curves(summaries)
        except Exception as e:
            logger.warning(f"Could not index light curves: {e}")

        stats = self.vector_store.get_stats()
        logger.info(f"Knowledge base indexed: {stats['document_count']} documents")

    def query(
        self,
        question: str,
        n_context: int = 3,
        include_context: bool = True
    ) -> Dict:
        """
        Query the RAG system

        Args:
            question: User question
            n_context: Number of context documents to retrieve
            include_context: Whether to include retrieved context in response

        Returns:
            Dict with answer and metadata
        """
        if not self.available:
            return {
                'answer': "RAG system is not available. Please ensure Ollama and ChromaDB are installed.",
                'sources': [],
                'error': True
            }

        # Retrieve relevant context
        logger.info(f"Querying: {question}")
        retrieved_docs = self.vector_store.search(question, n_results=n_context)

        # Format context
        context_text = "\n\n".join([
            f"Source {i+1}:\n{doc['document']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        # Generate answer with context
        if context_text and include_context:
            answer = self.ollama.generate_with_context(
                query=question,
                context=context_text,
                temperature=0.6
            )
        else:
            answer = self.assistant.answer_question(question)

        # Get follow-up suggestions
        suggestions = self.assistant.get_suggestions(question)

        return {
            'answer': answer,
            'sources': [
                {
                    'document': doc['document'][:200] + "...",
                    'metadata': doc['metadata'],
                    'relevance': 1.0 - (doc['distance'] or 0) if doc['distance'] else None
                }
                for doc in retrieved_docs
            ],
            'suggestions': suggestions,
            'error': False
        }

    def explain_prediction(
        self,
        prediction_result: Dict,
        features: Optional[Dict] = None
    ) -> str:
        """
        Explain a model prediction using RAG

        Args:
            prediction_result: Prediction output
            features: Input features

        Returns:
            Explanation text
        """
        if not self.available:
            return "RAG system not available"

        # Get relevant context about the model
        model_name = prediction_result.get('model', 'unknown')
        model_context = self.vector_store.search(
            f"model {model_name} performance metrics",
            n_results=1
        )

        context = model_context[0]['document'] if model_context else ""

        # Generate explanation
        system_prompt = f"""You are explaining an exoplanet classification prediction.
Use this model information if relevant:
{context}

Provide a clear, educational explanation."""

        pred_info = json.dumps(prediction_result, indent=2)
        features_info = json.dumps(features, indent=2) if features else "Not provided"

        prompt = f"""Explain this exoplanet classification:

Prediction: {pred_info}

Features: {features_info}

Explain:
1. What the prediction means
2. Confidence level interpretation
3. Key factors in the decision
4. Scientific significance"""

        return self.ollama.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.4
        )

    def compare_datasets(
        self,
        dataset1: str,
        dataset2: str
    ) -> str:
        """
        Compare two datasets

        Args:
            dataset1: First dataset name
            dataset2: Second dataset name

        Returns:
            Comparison text
        """
        # Retrieve dataset information
        context1 = self.vector_store.search(f"dataset {dataset1}", n_results=1)
        context2 = self.vector_store.search(f"dataset {dataset2}", n_results=1)

        context = ""
        if context1:
            context += f"{dataset1.upper()}:\n{context1[0]['document']}\n\n"
        if context2:
            context += f"{dataset2.upper()}:\n{context2[0]['document']}\n\n"

        prompt = f"""Compare the {dataset1.upper()} and {dataset2.upper()} exoplanet datasets:

{context}

Provide a detailed comparison covering:
1. Dataset sizes and coverage
2. Key differences in features
3. Scientific objectives
4. Strengths and limitations
5. Best use cases for each"""

        return self.ollama.generate(prompt=prompt, temperature=0.5)

    def get_model_recommendations(
        self,
        use_case: str
    ) -> str:
        """
        Get model recommendations for a use case

        Args:
            use_case: Description of use case

        Returns:
            Recommendations
        """
        # Get all model information
        model_docs = self.vector_store.search(
            "model performance accuracy metrics",
            n_results=5,
            filter_metadata={'type': 'model_results'}
        )

        models_info = "\n\n".join([doc['document'] for doc in model_docs])

        prompt = f"""Based on these model performances:

{models_info}

Recommend the best model(s) for this use case: {use_case}

Consider:
1. Accuracy requirements
2. Speed/efficiency
3. Interpretability
4. Specific strengths

Provide specific recommendations with reasoning."""

        return self.ollama.generate(prompt=prompt, temperature=0.5)


class ChatHistory:
    """Manage conversation history"""

    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to history"""
        self.history.append({
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'timestamp': None  # Add timestamp if needed
        })

        # Trim history if too long
        if len(self.history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.history = self.history[-self.max_history * 2:]

    def get_context(self, n_messages: int = 6) -> str:
        """Get recent context as string"""
        recent = self.history[-n_messages:] if len(self.history) > n_messages else self.history
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

    def clear(self):
        """Clear history"""
        self.history = []


if __name__ == "__main__":
    # Test RAG system
    print("Testing RAG System...")

    rag = RAGSystem(ollama_model="llama3.2")

    if rag.available:
        # Index knowledge base
        print("\n1. Indexing knowledge base...")
        rag.index_knowledge_base()

        # Test query
        print("\n2. Testing query...")
        result = rag.query("How does the transit method detect exoplanets?")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {len(result['sources'])}")
        print(f"Suggestions: {result['suggestions']}")

        # Test dataset comparison
        print("\n3. Testing dataset comparison...")
        comparison = rag.compare_datasets("kepler", "tess")
        print(f"\nComparison:\n{comparison[:500]}...")

    else:
        print("RAG system not available. Please install Ollama and ChromaDB.")
