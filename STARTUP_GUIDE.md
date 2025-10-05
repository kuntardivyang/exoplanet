# 🚀 How to Start the Exoplanet Detection System

## ⚡ Quick Start (Recommended)

Use the **automated startup script**:

```bash
./START_SYSTEM.sh
```

This script:
- ✅ Checks all prerequisites
- ✅ Starts backend API (port 8000)
- ✅ Starts frontend (port 3000)
- ✅ Starts Ollama if available
- ✅ Shows all access URLs
- ✅ Creates log files

---

## 📋 Prerequisites Check

The system needs:
- ✅ Python 3.12+ (you have this)
- ✅ Node.js & npm (for frontend)
- ✅ Ollama (optional, for AI chat)

**Already installed:**
- FastAPI, scikit-learn, XGBoost, LightGBM, etc.
- React, Tailwind CSS, etc.
- Ollama with llama3.2 model

---

## 🎯 Step-by-Step Startup

### **Option 1: Automatic (Best)**

```bash
./START_SYSTEM.sh
```

Wait 10-15 seconds for everything to start, then visit:
- 🎨 **Frontend:** http://localhost:3000
- 📡 **API:** http://localhost:8000/docs

---

### **Option 2: Manual**

If you prefer manual control:

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

## 🛑 How to Stop

**If using START_SYSTEM.sh:**
```bash
./STOP_SYSTEM.sh
```

**If manual:**
Press `Ctrl+C` in each terminal.

---

## ✅ Verify It's Running

```bash
# Check backend
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000

# Check Ollama
curl http://localhost:11434/api/version

# Check chat
curl http://localhost:8000/chat/status
```

---

## 🐛 Troubleshooting

### **Error: Port already in use**

```bash
# Find what's using port 8000
lsof -i:8000

# Kill it
kill -9 <PID>
```

### **Error: Module not found**

```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements-llm.txt
```

### **Error: npm not found**

```bash
# Install Node.js first
# Then install frontend dependencies
cd frontend
npm install
```

### **Frontend not loading**

```bash
# Rebuild frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## 📊 What Gets Started

| Service | Port | URL | Status |
|---------|------|-----|--------|
| **Backend API** | 8000 | http://localhost:8000 | Running ✅ |
| **Frontend** | 3000 | http://localhost:3000 | Ready ✅ |
| **Ollama LLM** | 11434 | http://localhost:11434 | Ready ✅ |

---

## 📝 Log Files

Logs are saved in `logs/` directory:

```bash
# View backend logs
tail -f logs/backend.log

# View frontend logs
tail -f logs/frontend.log

# View Ollama logs
tail -f logs/ollama.log
```

---

## 🎨 Access the Application

Once started, open your browser:

**Main Application:** http://localhost:3000

**Pages available:**
1. 🏠 **Dashboard** - System overview
2. 🔮 **Predictions** - Make predictions
3. 🎓 **Training** - Train models
4. 📊 **Data Explorer** - Browse datasets
5. 🤖 **Models** - Manage models
6. 💬 **AI Assistant** - Chat with AI

---

## ⚙️ Configuration

**Backend:** `backend/config/config.py`
**Frontend:** `frontend/vite.config.js`

---

## 🚀 Ready to Go!

Your system is ready. Just run:

```bash
./START_SYSTEM.sh
```

Then visit **http://localhost:3000** and start exploring! 🪐✨

---

## 📚 More Help

- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **Complete Guide:** [COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md)
- **System Ready:** [SYSTEM_READY.md](SYSTEM_READY.md)
- **Final Summary:** [FINAL_SUMMARY.txt](FINAL_SUMMARY.txt)

---

**Built for NASA Space Apps Challenge 2025** 🌌
