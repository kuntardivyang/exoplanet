# 🚀 Quick Start Guide - Exoplanet Detection System

Get your complete AI-powered exoplanet detection system running in **3 simple steps**!

---

## ⚡ TL;DR - Start Everything Now

```bash
# Start all services
./START_SYSTEM.sh

# Visit the app
# Open browser: http://localhost:3000
```

That's it! The system will automatically start:
- ✅ Backend API (FastAPI)
- ✅ Frontend UI (React)
- ✅ AI Chat (Ollama - if installed)

---

## 📋 Prerequisites

### Required
- **Python 3.12+** (you have this)
- **Node.js & npm** (for frontend)
- **pip** (Python package manager)

### Optional (for AI Chat)
- **Ollama** - Local LLM for chat features

---

## 🎯 Step-by-Step Setup

### 1️⃣ Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install LLM dependencies (optional, for chat)
pip install -r requirements-llm.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2️⃣ Train Models (First Time Only)

```bash
# Train on Kepler dataset (~2 minutes)
python3 backend/models/train_pipeline.py
```

Expected output:
```
✓ XGBoost: 98.95% accuracy (best)
✓ LightGBM: 98.75%
✓ Random Forest: 98.62%
✓ Neural Network: 98.42%
✓ Gradient Boosting: 98.42%
```

### 3️⃣ Install Ollama (Optional - for Chat Features)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2
```

---

## 🏃 Running the System

### Option A: **Automatic Start (Recommended)**

```bash
./START_SYSTEM.sh
```

This script:
- ✅ Checks all prerequisites
- ✅ Starts backend, frontend, and Ollama
- ✅ Shows you all access URLs
- ✅ Creates log files

### Option B: **Manual Start (3 Terminals)**

**Terminal 1 - Backend:**
```bash
python3 backend/api/main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Ollama (optional):**
```bash
ollama serve
```

---

## 🌐 Access the System

Once running, open these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| 🎨 **Main App** | http://localhost:3000 | Full web interface |
| 📚 **API Docs** | http://localhost:8000/docs | Interactive API docs |
| 💚 **Health Check** | http://localhost:8000/health | System status |
| 💬 **Chat Status** | http://localhost:8000/chat/status | AI chat status |

---

## 🎮 What You Can Do

### 1. **Dashboard**
View system overview, dataset statistics, and model performance

### 2. **Make Predictions**
- **Single:** Enter exoplanet parameters manually
- **Batch:** Test multiple samples at once
- **Upload:** Upload CSV files for bulk predictions

### 3. **Train Models**
- Select dataset (Kepler, TESS, K2)
- Watch real-time training progress
- Compare model performance

### 4. **Explore Data**
- Browse NASA datasets
- View statistics and distributions
- Examine sample data

### 5. **Manage Models**
- View all trained models
- Switch between models
- See performance metrics

### 6. **AI Assistant** ✨
Ask questions like:
- "How does the transit method work?"
- "Which model is most accurate?"
- "Compare Kepler and TESS datasets"
- "Explain this false positive classification"

---

## 🛑 Stopping the System

```bash
./STOP_SYSTEM.sh
```

Or manually press `Ctrl+C` in each terminal.

---

## 🧪 Test the System

Run comprehensive tests:

```bash
python3 test_system.py
```

This will verify:
- ✅ Models are trained
- ✅ Predictions work
- ✅ Light curves load
- ✅ Datasets load
- ✅ API is healthy
- ✅ Chat system works

---

## 📊 System Status Check

### Check if Backend is Running
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "datasets_available": ["kepler", "tess", "k2"]
}
```

### Check if Frontend is Running
```bash
curl http://localhost:3000
```

Should return HTML.

### Check if Ollama is Running
```bash
curl http://localhost:11434/api/version
```

---

## 🐛 Troubleshooting

### Backend won't start

**Problem:** Port 8000 already in use
```bash
# Find what's using the port
lsof -i:8000

# Kill it
kill -9 <PID>
```

**Problem:** Models not found
```bash
# Check if models exist
ls -lh data/models/

# Retrain if needed
python3 backend/models/train_pipeline.py
```

### Frontend won't start

**Problem:** Node modules missing
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Problem:** Port 3000 in use
```bash
# Edit frontend/vite.config.js
# Change port to 3001
```

### Chat not working

**Problem:** Ollama not running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start it
ollama serve
```

**Problem:** Model not pulled
```bash
# Check available models
ollama list

# Pull llama3.2 if missing
ollama pull llama3.2
```

---

## 📁 Project Structure

```
exoplanet/
├── backend/              # Python backend
│   ├── api/             # FastAPI endpoints
│   ├── models/          # ML models & training
│   ├── utils/           # Data loading, preprocessing
│   └── llm/             # Ollama & RAG system
│
├── frontend/            # React frontend
│   └── src/
│       ├── pages/       # 6 pages
│       └── services/    # API client
│
├── data/
│   ├── raw/            # NASA datasets (CSV)
│   ├── processed/      # Preprocessed data
│   └── models/         # Trained models
│
├── lightcurve/         # 203 light curve files
│
└── logs/               # Runtime logs
```

---

## 🔥 Pro Tips

1. **First time?** Run `./START_SYSTEM.sh` - it does everything for you

2. **Training slow?** The first training takes ~2 minutes. Subsequent trainings are cached.

3. **Chat not working?** It's optional! The system works fine without Ollama. Chat just adds extra features.

4. **Want to customize?** Check `backend/config/config.py` for all settings.

5. **Logs?** All logs are in the `logs/` directory:
   - `logs/backend.log` - API logs
   - `logs/frontend.log` - Frontend logs
   - `logs/ollama.log` - Chat logs

6. **Performance?** The system can process:
   - Single predictions: <100ms
   - Batch: ~10,000 samples/minute
   - Chat responses: 1-3 seconds

---

## 📚 Next Steps

- **Read the full documentation:** [COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md)
- **Explore the API:** http://localhost:8000/docs
- **Try the chat assistant:** Ask about exoplanet detection methods
- **Upload your own data:** Use the Predict page to upload CSV files
- **Train on different datasets:** Try TESS and K2 datasets

---

## 🆘 Need Help?

- **Documentation:** See `Documentation/` folder
- **API Reference:** http://localhost:8000/docs
- **System Overview:** [COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md)
- **Test Script:** Run `python3 test_system.py`

---

## ✅ Verification Checklist

Before you start using the system:

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Models trained (`python3 backend/models/train_pipeline.py`)
- [ ] Frontend dependencies installed (`cd frontend && npm install`)
- [ ] Backend running (http://localhost:8000/health returns healthy)
- [ ] Frontend running (http://localhost:3000 loads)
- [ ] Ollama installed and running (optional, for chat)

---

## 🎉 You're All Set!

Your complete exoplanet detection system is ready to:
- 🔍 Detect exoplanets with 98.95% accuracy
- 📊 Analyze NASA mission data
- 💬 Answer questions with AI
- 🚀 Process thousands of predictions per minute

**Happy exoplanet hunting! 🪐✨**

---

*Built for NASA Space Apps Challenge 2025*
