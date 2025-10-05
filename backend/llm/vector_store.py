"""
Vector Store for RAG (Retrieval Augmented Generation)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import List, Dict, Optional, Tuple
import json
import pickle
from backend.utils.logger import setup_logger
from backend.config.config import BASE_DIR

logger = setup_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")


class VectorStore:
    """Vector database for storing and retrieving context"""

    def __init__(
        self,
        collection_name: str = "exoplanet_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store

        Args:
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
        """
        self.collection_name = collection_name
        self.db_path = BASE_DIR / "data" / "vector_db"
        self.db_path.mkdir(parents=True, exist_ok=True)

        if not CHROMA_AVAILABLE or not EMBEDDINGS_AVAILABLE:
            logger.error("ChromaDB or SentenceTransformers not available")
            self.available = False
            return

        self.available = True

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Exoplanet detection knowledge base"}
        )

        logger.info(f"Vector store initialized. Collection: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to vector store

        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
        """
        if not self.available:
            logger.error("Vector store not available")
            return

        if not documents:
            return

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents).tolist()

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of matching documents with scores
        """
        if not self.available:
            logger.error("Vector store not available")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return formatted_results

    def delete_collection(self):
        """Delete the entire collection"""
        if self.available:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        if not self.available:
            return {"available": False}

        count = self.collection.count()
        return {
            "available": True,
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": "all-MiniLM-L6-v2"
        }


class KnowledgeIndexer:
    """Index exoplanet knowledge into vector store"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def index_model_results(self, results_path: Path):
        """Index model training results"""
        logger.info(f"Indexing model results from {results_path}")

        with open(results_path) as f:
            results = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for model_name, model_results in results.items():
            # Create document for each model
            doc = f"""Model: {model_name}
Training Time: {model_results.get('training_time', 'N/A')} seconds
Train Accuracy: {model_results.get('train_accuracy', 'N/A')}

Validation Metrics:
- Accuracy: {model_results.get('validation', {}).get('accuracy', 'N/A')}
- Precision: {model_results.get('validation', {}).get('precision', 'N/A')}
- Recall: {model_results.get('validation', {}).get('recall', 'N/A')}
- F1 Score: {model_results.get('validation', {}).get('f1_score', 'N/A')}

Test Metrics:
- Accuracy: {model_results.get('test', {}).get('accuracy', 'N/A')}
- Precision: {model_results.get('test', {}).get('precision', 'N/A')}
- Recall: {model_results.get('test', {}).get('recall', 'N/A')}
- F1 Score: {model_results.get('test', {}).get('f1_score', 'N/A')}"""

            documents.append(doc)
            metadatas.append({
                'type': 'model_results',
                'model_name': model_name,
                'accuracy': model_results.get('test', {}).get('accuracy', 0)
            })
            ids.append(f"model_{model_name}")

        self.vector_store.add_documents(documents, metadatas, ids)

    def index_dataset_info(self, dataset_name: str, info: Dict):
        """Index dataset information"""
        doc = f"""Dataset: {dataset_name}
Rows: {info.get('rows', 'N/A')}
Columns: {info.get('columns', 'N/A')}
Features: {', '.join(info.get('features', [])[:20])}

Target Distribution: {json.dumps(info.get('target_distribution', {}), indent=2)}"""

        self.vector_store.add_documents(
            [doc],
            [{'type': 'dataset', 'name': dataset_name}],
            [f"dataset_{dataset_name}"]
        )

    def index_light_curves(self, light_curve_summaries: List[Dict]):
        """Index light curve information"""
        documents = []
        metadatas = []
        ids = []

        for i, summary in enumerate(light_curve_summaries):
            doc = f"""Light Curve: {summary.get('star_id')}
Transit Depth: {summary.get('transit_depth', 'N/A')}
Transit Depth (ppm): {summary.get('transit_depth_ppm', 'N/A')}
Orbital Period: {summary.get('period', 'N/A')} days
Number of Data Points: {summary.get('n_points', 'N/A')}
Min Flux: {summary.get('min_flux', 'N/A')}
Max Flux: {summary.get('max_flux', 'N/A')}"""

            documents.append(doc)
            metadatas.append({
                'type': 'light_curve',
                'star_id': summary.get('star_id'),
                'period': summary.get('period')
            })
            ids.append(f"lightcurve_{i}")

        self.vector_store.add_documents(documents, metadatas, ids)

    def index_exoplanet_facts(self):
        """Index general exoplanet knowledge"""
        facts = [
            {
                'doc': """The transit method detects exoplanets by measuring the dimming of a star's light when a planet passes in front of it. The depth of the transit reveals the planet's size relative to its host star, while the duration indicates the planet's orbital characteristics.""",
                'meta': {'type': 'knowledge', 'topic': 'transit_method'}
            },
            {
                'doc': """Kepler Space Telescope discovered over 2,700 confirmed exoplanets by continuously monitoring the brightness of over 150,000 stars. It revolutionized our understanding of planetary systems and showed that planets are common in our galaxy.""",
                'meta': {'type': 'knowledge', 'topic': 'kepler_mission'}
            },
            {
                'doc': """TESS (Transiting Exoplanet Survey Satellite) surveys the entire sky for transiting exoplanets around nearby bright stars. Unlike Kepler, TESS focuses on closer, brighter stars that are ideal targets for follow-up observations.""",
                'meta': {'type': 'knowledge', 'topic': 'tess_mission'}
            },
            {
                'doc': """Hot Jupiters are gas giant exoplanets that orbit very close to their host stars, completing an orbit in just a few days. They were unexpected discoveries that challenged our understanding of planetary formation and migration.""",
                'meta': {'type': 'knowledge', 'topic': 'hot_jupiters'}
            },
            {
                'doc': """The habitable zone is the region around a star where conditions might allow liquid water to exist on a planet's surface. The location and width of this zone depends on the star's temperature and luminosity.""",
                'meta': {'type': 'knowledge', 'topic': 'habitable_zone'}
            },
            {
                'doc': """Machine learning models can identify exoplanets by learning patterns in light curves, stellar parameters, and other features. Common algorithms include Random Forests, XGBoost, and neural networks, achieving 95-99% accuracy.""",
                'meta': {'type': 'knowledge', 'topic': 'ml_detection'}
            }
        ]

        documents = [f['doc'] for f in facts]
        metadatas = [f['meta'] for f in facts]
        ids = [f"fact_{f['meta']['topic']}" for f in facts]

        self.vector_store.add_documents(documents, metadatas, ids)


if __name__ == "__main__":
    # Test vector store
    print("Testing Vector Store...")

    store = VectorStore()

    if store.available:
        # Add some test documents
        docs = [
            "Exoplanets are planets that orbit stars outside our solar system.",
            "The transit method detects planets by measuring brightness dips.",
            "Hot Jupiters are gas giants very close to their stars."
        ]

        store.add_documents(
            docs,
            [{'type': 'test'} for _ in docs]
        )

        # Search
        results = store.search("How do we detect exoplanets?", n_results=2)
        print(f"\nSearch Results:")
        for r in results:
            print(f"- {r['document'][:100]}...")

        # Stats
        stats = store.get_stats()
        print(f"\nStats: {stats}")
    else:
        print("Vector store not available")
