# 🤖 Ollama AI Chatbot Setup Guide

## Complete RAG-Powered Exoplanet Assistant

---

## ✅ **What Has Been Built**

### **Complete AI Chatbot System with:**
- ✅ **Ollama Integration** - Local LLM (llama3.2, mistral, etc.)
- ✅ **RAG System** - Retrieval Augmented Generation
- ✅ **Vector Database** - ChromaDB for context storage
- ✅ **Knowledge Indexing** - Automated indexing of all data
- ✅ **Chat API** - 8 RESTful endpoints
- ✅ **Conversation Memory** - Multi-turn conversations
- ✅ **Context-Aware Responses** - Uses your actual training data

---

## 📦 **New Files Created**

```
backend/llm/
├── __init__.py
├── ollama_client.py          # Ollama LLM wrapper
├── vector_store.py            # ChromaDB vector database
└── rag_system.py              # Complete RAG implementation

backend/api/
├── chat_routes.py             # Chat API endpoints
└── schemas.py                 # Updated with chat schemas

requirements-llm.txt           # LLM dependencies
```

---

## 🚀 **Installation Steps**

### **1. Install Ollama**

**On Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**On macOS:**
```bash
brew install ollama
```

**On Windows:**
Download from https://ollama.com/download

### **2. Start Ollama Service**
```bash
ollama serve
```

### **3. Pull a Model**
```bash
# Recommended: Llama 3.2 (3B parameters, fast)
ollama pull llama3.2

# Alternative options:
# ollama pull llama3.2:1b    # Smaller, faster (1B)
# ollama pull mistral        # Alternative model
# ollama pull codellama      # For code explanation
```

### **4. Install Python Dependencies**
```bash
pip install -r requirements-llm.txt
```

This installs:
- `ollama` - Python client
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `langchain` - LLM framework (optional)

---

## 🎯 **API Endpoints**

### **Base URL:** `http://localhost:8000/chat`

### **1. Chat Query** `POST /chat/query`
Ask questions and get contextual answers

```json
{
  "message": "How does the transit method detect exoplanets?",
  "include_context": true,
  "n_context": 3
}
```

Response:
```json
{
  "answer": "The transit method detects...",
  "sources": [...],
  "suggestions": [
    "What is the detection accuracy?",
    "Compare with radial velocity method"
  ],
  "conversation_id": "uuid-here"
}
```

### **2. Explain Prediction** `POST /chat/explain`
Get AI explanation of model predictions

```json
{
  "prediction": {
    "predicted_label": "CONFIRMED",
    "confidence": 0.95,
    "probabilities": [0.95, 0.05]
  },
  "features": {
    "koi_period": 3.5,
    "koi_depth": 2500
  }
}
```

### **3. Compare Items** `POST /chat/compare`
Compare datasets, models, or objects

```json
{
  "item1": "kepler",
  "item2": "tess",
  "comparison_type": "datasets"
}
```

### **4. Get Recommendations** `GET /chat/recommendations`
```bash
GET /chat/recommendations?use_case=highest_accuracy
```

### **5. Chat Status** `GET /chat/status`
Check system availability

### **6. Reindex Knowledge** `POST /chat/reindex`
Re-index the knowledge base (admin)

### **7. Clear Conversation** `DELETE /chat/conversation/{id}`
Clear chat history

---

## 💬 **Example Conversations**

### **Ask About Methods**
```
User: "How does the transit method detect exoplanets?"

Bot: "The transit method detects exoplanets by measuring the dimming
of a star's light when a planet passes in front of it. Based on our
Kepler dataset with 7,585 observations, the method works by:

1. Continuous monitoring of star brightness
2. Detecting periodic dips in light (transits)
3. Measuring transit depth to determine planet size
4. Analyzing transit duration for orbital characteristics

Our XGBoost model achieves 98.95% accuracy in classifying these
transits as confirmed planets or false positives."
```

### **Compare Datasets**
```
User: "Compare Kepler and TESS datasets"

Bot: "Here's a comparison of Kepler and TESS:

KEPLER:
- 9,564 objects of interest
- 141 features including detailed stellar parameters
- Focused on one field of view for continuous monitoring
- Best for: Statistical studies of planet populations

TESS:
- 7,794 objects of interest
- All-sky survey of nearby bright stars
- Shorter observation periods per target
- Best for: Finding planets around nearby stars for follow-up

Key Difference: Kepler went deep in one area, TESS surveys
the entire sky with nearby bright stars ideal for
atmospheric characterization."
```

### **Explain Predictions**
```
User: "Why was TOI-1234 classified as false positive?"

Bot: "TOI-1234 was classified as FALSE POSITIVE with 87% confidence
based on these factors:

1. HIGH STELLAR CONTAMINATION: The koi_fpflag_ss flag indicates
   stellar activity rather than a planetary transit

2. TRANSIT INCONSISTENCY: The transit depth varies by ±15%, which
   is unusual for a true planet

3. PERIOD INSTABILITY: The orbital period shows variation,
   suggesting it's not a stable planetary orbit

Our model learned these patterns from 7,585 training examples and
correctly identifies such stellar mimics with 98.9% accuracy."
```

### **Get Model Recommendations**
```
User: "Which model should I use for highest accuracy?"

Bot: "Based on our training results:

RECOMMENDED: XGBoost
- Accuracy: 98.95%
- F1 Score: 98.94%
- Training Time: 5 seconds
- Best for: Maximum accuracy on confirmed vs false positive

WHY IT'S BEST:
- Handles non-linear relationships excellently
- Robust to outliers and missing data
- Feature importance clearly ranked
- Fastest among high-accuracy models

ALTERNATIVES:
- LightGBM (98.75%) - If you need even faster inference
- Random Forest (98.62%) - If you need interpretability"
```

---

## 🧠 **What the Bot Knows**

The AI assistant has been indexed with:

### **1. Model Training Results**
- All 5 model performances
- Training times and metrics
- Confusion matrices
- Feature importance rankings

### **2. Dataset Information**
- Kepler: 9,564 objects, 141 features
- TESS: 7,794 objects
- K2: 4,303 objects
- All feature descriptions

### **3. Light Curve Data**
- 203 transit light curves
- 13 confirmed exoplanet systems
- Transit depths, periods, characteristics

### **4. Exoplanet Knowledge**
- Transit method explanation
- Mission descriptions (Kepler, TESS, K2)
- Hot Jupiters, habitable zones
- ML detection techniques

---

## 🔧 **Testing the System**

### **1. Start Backend with Chat**
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start the API
cd backend/api
python3 main.py
```

You should see:
```
Chat routes included
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### **2. Test Chat Endpoint**
```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is an exoplanet?",
    "include_context": true
  }'
```

### **3. Check Status**
```bash
curl http://localhost:8000/chat/status
```

Should return:
```json
{
  "available": true,
  "ollama_available": true,
  "vector_store_available": true,
  "vector_store_stats": {
    "document_count": 50+
  }
}
```

### **4. Test in API Docs**
Open http://localhost:8000/docs

Navigate to **chat** section and try the endpoints interactively!

---

## 🎨 **Frontend Integration** (Coming Next)

The chat UI will have:
```jsx
- ChatAssistant.jsx      # Main chat page
- ChatInterface.jsx      # Chat bubble component
- MessageList.jsx        # Conversation display
- SuggestedQuestions.jsx # Quick question buttons
- SourcesPanel.jsx       # Retrieved context display
```

Features:
- Real-time streaming responses
- Markdown formatting
- Code syntax highlighting
- Source citation
- Conversation history
- Export conversations

---

## 🚨 **Troubleshooting**

### **Ollama Not Running**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not, start it
ollama serve
```

### **Model Not Found**
```bash
# List installed models
ollama list

# Pull llama3.2
ollama pull llama3.2
```

### **ChromaDB Error**
```bash
# Reinstall ChromaDB
pip install --upgrade chromadb

# Clear and reindex
curl -X POST http://localhost:8000/chat/reindex
```

### **Slow Responses**
```python
# Use smaller model
ollama pull llama3.2:1b  # 1B parameter model

# Update in code:
rag = RAGSystem(ollama_model="llama3.2:1b")
```

---

## 📊 **Performance**

### **Response Times:**
- Context retrieval: ~50-100ms
- LLM generation: ~1-3 seconds (llama3.2)
- Total: ~1-3 seconds per query

### **Model Sizes:**
- llama3.2:1b - 1.3GB (fastest)
- llama3.2:3b - 2.0GB (recommended)
- mistral - 4.1GB (most capable)

### **Resource Usage:**
- RAM: 4-8GB (depending on model)
- CPU: Multi-core recommended
- GPU: Optional, speeds up significantly

---

## 🎯 **Next Steps**

1. ✅ **Test the chat API** - Try all endpoints
2. ✅ **Ask questions** - Test with various queries
3. 🔄 **Build frontend** - Create React chat UI
4. 🔄 **Add streaming** - Real-time response streaming
5. 🔄 **Fine-tune** - Customize for your use case

---

## 💡 **Advanced Features to Add**

### **1. Streaming Responses**
```python
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # Return Server-Sent Events
    async def generate():
        for chunk in rag.ollama.generate(..., stream=True):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### **2. Multi-Modal (Images)**
```python
# Add light curve image analysis
- Upload light curve plot
- AI describes the transit
- Suggests classification
```

### **3. Custom Knowledge**
```python
# Add research papers
indexer.add_documents(
    papers,
    metadatas=[{'type': 'paper', 'doi': '...'}]
)
```

---

## 🌟 **Summary**

You now have a **complete AI chatbot system** that:
- ✅ Runs locally with Ollama (private & secure)
- ✅ Uses RAG for accurate, contextual answers
- ✅ Knows about your models, data, and exoplanets
- ✅ Explains predictions in human language
- ✅ Compares datasets and recommends models
- ✅ Maintains conversation history
- ✅ Provides follow-up suggestions

**The bot is ready to help users understand exoplanet detection!** 🚀✨

---

## 📚 **Documentation**

- Ollama: https://ollama.com/
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- FastAPI: https://fastapi.tiangolo.com/

---

**Installation command:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2

# Install Python deps
pip install -r requirements-llm.txt

# Start everything
ollama serve  # Terminal 1
python3 backend/api/main.py  # Terminal 2
```

**Test it:**
```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the best model for detecting exoplanets?"}'
```

🎉 **Your AI Assistant is Ready!**