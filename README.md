# 🌌 Exoplanet Detection System

**AI-Powered Exoplanet Discovery Platform**

A complete, production-ready machine learning system for detecting exoplanets using NASA mission data. Built for NASA Space Apps Challenge 2025.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.95%25-success.svg)](.)

## ✨ Features

### 🧠 **Machine Learning**
- **5 State-of-the-Art Models** (XGBoost, LightGBM, Random Forest, Neural Network, Gradient Boosting)
- **98.95% Accuracy** on Kepler dataset
- **Automated Training Pipeline** with cross-validation
- **Feature Engineering** from 141 to 50 optimal features

### 📊 **NASA Data Integration**
- **Kepler Mission:** 9,564 objects, 141 features
- **TESS Mission:** 7,794 objects
- **K2 Mission:** 4,303 objects
- **203 Light Curves** from 13 exoplanet systems

### 🤖 **AI Assistant**
- **RAG-Powered Chatbot** using Ollama (llama3.2)
- **Vector Database** (ChromaDB) for context retrieval
- **Intelligent Q&A** about exoplanets, models, and data
- **Source Citations** for all answers

### 🌐 **Full-Stack Web Application**
- **6 Interactive Pages** (Dashboard, Predictions, Training, Data Explorer, Models, AI Chat)
- **Real-Time Updates** during training
- **Batch Processing** for thousands of predictions
- **CSV Upload** for bulk predictions

### 🚀 **Production Ready**
- **RESTful API** with 20+ endpoints
- **Comprehensive Error Handling**
- **Logging & Monitoring**
- **Input Validation** with Pydantic
- **CORS Enabled** for cross-origin requests

## 📁 Project Structure

```
exoplanet/
├── backend/
│   ├── api/                    # FastAPI endpoints
│   │   ├── main.py            # Main API application
│   │   └── schemas.py         # Pydantic models
│   ├── models/                # ML models
│   │   ├── train_pipeline.py  # Training pipeline
│   │   ├── model_trainer.py   # Model training logic
│   │   └── predictor.py       # Prediction service
│   ├── utils/                 # Utilities
│   │   ├── data_loader.py     # Dataset loading
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── data_exploration.py # Data analysis
│   │   └── logger.py          # Logging utilities
│   └── config/                # Configuration
│       └── config.py          # Settings
├── frontend/                  # React application
│   ├── src/
│   │   ├── pages/            # UI pages
│   │   ├── components/       # React components
│   │   ├── services/         # API services
│   │   └── styles/           # CSS styles
│   └── package.json
├── data/
│   ├── raw/                  # Raw CSV datasets
│   ├── processed/            # Processed data
│   └── models/               # Saved models
├── notebooks/                # Jupyter notebooks
└── logs/                     # Application logs
```

## 🎯 Quick Start

### **1. Install & Setup**
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-llm.txt  # Optional, for chat

# Install frontend
cd frontend && npm install && cd ..

# Install Ollama (optional, for AI chat)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

### **2. Train Models**
```bash
# Train on Kepler dataset (~2 minutes)
python3 backend/models/train_pipeline.py
```

### **3. Start System**
```bash
# Automatic start (recommended)
./START_SYSTEM.sh

# Or manual start (3 terminals)
# Terminal 1: python3 backend/api/main.py
# Terminal 2: cd frontend && npm run dev
# Terminal 3: ollama serve
```

### **4. Access the System**
- 🎨 **Main App:** http://localhost:3000
- 📚 **API Docs:** http://localhost:8000/docs
- 💚 **Health Check:** http://localhost:8000/health

📖 **Full Guide:** See [QUICK_START.md](QUICK_START.md)

## 🎯 Usage

### 1. Dashboard
- View system status and model performance
- Explore dataset statistics
- See feature importance charts

### 2. Make Predictions
- **Single Prediction**: Enter feature values manually
- **Batch Prediction**: Add multiple samples
- **CSV Upload**: Upload a CSV file for bulk predictions

### 3. Train Models
- Select dataset (Kepler, TESS, or K2)
- Start training process
- Monitor real-time progress
- View training results and metrics

### 4. Data Explorer
- Browse available datasets
- View sample data
- Explore feature distributions

### 5. Model Management
- View all trained models
- Compare model performance
- Load different models
- View feature importance

## 🔧 API Endpoints

### Health & Status
- `GET /health` - API health check
- `GET /` - Root endpoint

### Predictions
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /predict/upload` - CSV file upload

### Models
- `GET /models` - List all models
- `GET /models/{name}/info` - Model details
- `POST /models/load/{name}` - Load specific model

### Training
- `POST /train` - Start training
- `GET /train/status` - Training status

### Features & Data
- `GET /features/importance` - Feature importance
- `GET /datasets` - List datasets
- `GET /datasets/{name}/sample` - Dataset sample

## 📊 Datasets

### Kepler (KOI)
- **Samples**: ~9,500
- **Features**: 141 columns
- **Classes**: CONFIRMED, FALSE POSITIVE, CANDIDATE

### TESS (TOI)
- **Samples**: ~7,800
- **Features**: Multiple transit and stellar parameters

### K2
- **Samples**: ~4,300
- **Features**: Planet and candidate data

## 🤖 ML Models

### Random Forest
- Ensemble of decision trees
- Robust to overfitting
- Good baseline performance

### XGBoost
- Gradient boosting framework
- High accuracy with proper tuning
- Handles missing data well

### LightGBM
- Fast training on large datasets
- Memory efficient
- Excellent for high-dimensional data

### Neural Network
- Multi-layer perceptron
- Captures complex patterns
- Requires sufficient training data

### Gradient Boosting
- Sequential ensemble method
- Good generalization
- Handles various data types

## 📈 Performance Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

## 🔬 Technical Details

### Data Preprocessing
1. **Cleaning**: Remove duplicates, handle missing values
2. **Feature Selection**: Correlation analysis, top-k selection
3. **Scaling**: StandardScaler or RobustScaler
4. **Outlier Handling**: IQR-based clipping
5. **Train/Val/Test Split**: 70/10/20

### Feature Engineering
- Automatic feature selection based on correlation
- Missing value imputation (median/mean/KNN)
- Outlier detection and handling
- Feature importance ranking

## 🎨 Frontend Technology

- **React 18**: Modern UI framework
- **Tailwind CSS**: Utility-first styling
- **Recharts**: Data visualization
- **Axios**: API communication
- **React Router**: Navigation
- **Lucide Icons**: Beautiful icons

## 🔐 Security

- CORS protection
- Input validation with Pydantic
- Error handling and logging
- Rate limiting ready

## 🚀 Deployment

### Backend (Production)
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend (Production)
```bash
npm run build
# Serve the dist/ folder with your preferred web server
```

## 📝 License

This project is built for NASA Space Apps Challenge 2025.

## 🤝 Contributing

This is a challenge submission. For NASA Space Apps Challenge participants only.

## 📧 Support

For issues and questions, please refer to the challenge documentation.

## 🌟 Acknowledgments

- NASA Exoplanet Archive for datasets
- Kepler, TESS, and K2 missions
- NASA Space Apps Challenge organizers
- Open-source ML community

---

**Built with ❤️ for exoplanet discovery and the search for life beyond Earth**
