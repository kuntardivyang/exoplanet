# 🌟 Exoplanet Detection System - Complete Overview

## NASA Space Apps Challenge 2025
**Challenge**: A World Away - Hunting for Exoplanets with AI

---

## 🎯 Project Summary

A complete end-to-end AI-powered system for detecting and classifying exoplanets from NASA's open-source datasets. The system features:

- **5 Machine Learning Models** trained in parallel
- **3 NASA Datasets** (Kepler, TESS, K2)
- **Full-stack Web Application** with React + FastAPI
- **Real-time Training Monitoring**
- **Batch Prediction Support**
- **Interactive Visualizations**

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  Dashboard | Predictions | Training | Data Explorer | Models │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Endpoints                                        │  │
│  │  /predict | /train | /models | /datasets | /features │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  ML Pipeline                                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ Data Loader│─▶│Preprocessor│─▶│  Trainer   │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Kepler (9.5k)│  │  TESS (7.8k) │  │   K2 (4.3k)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Loading & Exploration
- **DatasetLoader**: Loads CSV files from NASA Exoplanet Archive
- **DataExplorer**: Analyzes distributions, correlations, outliers
- Handles 3 datasets: Kepler KOI, TESS TOI, K2 Planets

### 2. Preprocessing Pipeline
**ExoplanetPreprocessor** performs:
- ✅ Data cleaning (duplicates, constants)
- ✅ Missing value imputation (median/mean/KNN)
- ✅ Outlier detection and handling (IQR method)
- ✅ Feature selection (correlation-based)
- ✅ Feature scaling (Standard/Robust scaler)
- ✅ Train/Validation/Test split (70/10/20)

### 3. Model Training
**ExoplanetModelTrainer** trains 5 models:

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Random Forest** | Ensemble of decision trees | Robust, interpretable |
| **XGBoost** | Gradient boosting | High accuracy, handles missing data |
| **LightGBM** | Fast gradient boosting | Efficient, memory-friendly |
| **Neural Network** | Multi-layer perceptron | Complex patterns |
| **Gradient Boosting** | Sequential ensemble | Good generalization |

### 4. Evaluation Metrics
Each model is evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean
- **ROC AUC**: Discrimination ability
- **Confusion Matrix**: Detailed breakdown

---

## 🎨 Frontend Features

### Dashboard
- System health monitoring
- Dataset statistics
- Feature importance visualization
- Target distribution charts
- Quick action cards

### Predictions
- **Single Prediction**: Manual feature input
- **Batch Prediction**: Multiple samples at once
- **CSV Upload**: Bulk file processing
- Confidence scores and class probabilities
- Real-time results

### Training
- Dataset selection (Kepler/TESS/K2)
- Real-time progress tracking
- Training status updates
- Results visualization
- Model comparison

### Data Explorer
- Browse all datasets
- Sample data preview
- Feature lists
- Distribution statistics

### Model Management
- List all trained models
- Load/switch models
- View performance metrics
- Feature importance charts
- Model comparison

---

## 🔧 Backend API

### Endpoints Overview

#### Health & Status
```
GET  /health          - API health check
GET  /                - Root endpoint
```

#### Predictions
```
POST /predict         - Single prediction
POST /predict/batch   - Batch predictions
POST /predict/upload  - CSV upload
```

#### Models
```
GET  /models                  - List all models
GET  /models/{name}/info      - Model details
POST /models/load/{name}      - Load model
```

#### Training
```
POST /train          - Start training
GET  /train/status   - Training progress
```

#### Features & Data
```
GET  /features/importance         - Feature importance
GET  /datasets                    - List datasets
GET  /datasets/{name}/sample      - Dataset sample
```

---

## 📁 Complete File Structure

```
exoplanet/
│
├── backend/
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   ├── schemas.py              # Pydantic models
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── train_pipeline.py      # Complete training pipeline
│   │   ├── model_trainer.py       # Model training logic
│   │   ├── predictor.py           # Prediction service
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── data_loader.py         # Dataset loading
│   │   ├── preprocessing.py       # Data preprocessing
│   │   ├── data_exploration.py    # Data analysis
│   │   ├── logger.py              # Logging utilities
│   │   └── __init__.py
│   │
│   ├── config/
│   │   ├── config.py              # Configuration settings
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Main dashboard
│   │   │   ├── Predict.jsx        # Prediction interface
│   │   │   ├── Training.jsx       # Training interface
│   │   │   ├── DataExplorer.jsx   # Data exploration
│   │   │   └── Models.jsx         # Model management
│   │   │
│   │   ├── services/
│   │   │   └── api.js             # API client
│   │   │
│   │   ├── styles/
│   │   │   └── index.css          # Global styles
│   │   │
│   │   ├── App.jsx                # Main app component
│   │   └── main.jsx               # Entry point
│   │
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── data/
│   ├── raw/                        # CSV datasets
│   │   ├── cumulative_2025.10.04_10.17.04.csv (Kepler)
│   │   ├── TOI_2025.10.04_10.17.23.csv (TESS)
│   │   └── k2pandc_2025.10.04_10.17.32.csv (K2)
│   │
│   ├── processed/                  # Preprocessed data
│   │   └── preprocessor_*.pkl
│   │
│   └── models/                     # Trained models
│       ├── random_forest_*.pkl
│       ├── xgboost_*.pkl
│       ├── lightgbm_*.pkl
│       ├── neural_network_*.pkl
│       ├── gradient_boosting_*.pkl
│       ├── model_comparison_*.csv
│       ├── training_results_*.json
│       └── metadata_*.json
│
├── logs/                           # Application logs
│   └── exoplanet_*.log
│
├── notebooks/                      # Jupyter notebooks (optional)
│
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
├── PROJECT_OVERVIEW.md            # This file
├── .gitignore
│
└── Scripts:
    ├── setup.sh                    # Setup script
    ├── run.sh                      # Run both servers
    └── train.sh                    # Train models
```

---

## 🚀 Getting Started

### Option 1: Quick Start (Recommended)
```bash
./setup.sh    # Install dependencies
./train.sh    # Train models
./run.sh      # Start system
```

### Option 2: Manual Setup
```bash
# Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend/models && python train_pipeline.py
cd ../api && python main.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## 📈 Performance Expectations

### Training Time
- **Kepler**: 10-15 minutes
- **TESS**: 8-12 minutes
- **K2**: 5-8 minutes

### Model Accuracy (Expected)
- **Random Forest**: 90-95%
- **XGBoost**: 92-97%
- **LightGBM**: 91-96%
- **Neural Network**: 88-93%
- **Gradient Boosting**: 90-94%

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core recommended for faster training
- **Disk**: 2GB free space
- **GPU**: Optional, speeds up neural network training

---

## 🌟 Key Features Implemented

### Core Functionality ✅
- [x] Multi-dataset support (Kepler, TESS, K2)
- [x] 5 different ML algorithms
- [x] Automated preprocessing pipeline
- [x] Feature selection and engineering
- [x] Model comparison and selection
- [x] Real-time training monitoring
- [x] Batch predictions
- [x] CSV file upload
- [x] RESTful API
- [x] Interactive web interface

### Advanced Features ✅
- [x] Feature importance visualization
- [x] Model performance metrics
- [x] Cross-validation
- [x] Hyperparameter configuration
- [x] Data exploration tools
- [x] Confidence scores
- [x] Class probability distributions
- [x] Logging and monitoring
- [x] Error handling
- [x] CORS support

### UI/UX Features ✅
- [x] Responsive design
- [x] Dark theme with space aesthetics
- [x] Interactive charts (Recharts)
- [x] Real-time updates
- [x] Progress indicators
- [x] Multi-page navigation
- [x] Form validation
- [x] Loading states
- [x] Error messages
- [x] Success notifications

---

## 🔬 Technical Highlights

### Backend
- **Framework**: FastAPI (modern, fast, async)
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Validation**: Pydantic schemas
- **Logging**: Custom colored logger

### Frontend
- **Framework**: React 18 with Hooks
- **Build Tool**: Vite (fast HMR)
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Routing**: React Router v6

### Data Pipeline
- **Cleaning**: Automated duplicate/constant removal
- **Imputation**: Multiple strategies (median/mean/KNN)
- **Scaling**: StandardScaler/RobustScaler
- **Selection**: Correlation-based feature selection
- **Validation**: Stratified train/val/test split

---

## 📊 Datasets Information

### Kepler Objects of Interest (KOI)
- **Source**: Kepler Space Telescope
- **Samples**: 9,564 objects
- **Features**: 141 columns
- **Target Classes**: CONFIRMED, FALSE POSITIVE, CANDIDATE
- **Best For**: Binary classification (confirmed vs false positive)

### TESS Objects of Interest (TOI)
- **Source**: Transiting Exoplanet Survey Satellite
- **Samples**: 7,794 objects
- **Features**: Multiple transit and stellar parameters
- **Status**: Active mission, regularly updated

### K2 Planets and Candidates
- **Source**: K2 Mission (Kepler extended)
- **Samples**: 4,303 objects
- **Features**: Planet and candidate data
- **Coverage**: Multiple campaign fields

---

## 🎓 How It Works

### 1. Data Preprocessing
```python
Preprocessor:
  ├── Load CSV → Remove duplicates
  ├── Handle missing values (median imputation)
  ├── Remove outliers (IQR clipping)
  ├── Select features (correlation > threshold)
  ├── Scale features (StandardScaler)
  └── Split data (70/10/20)
```

### 2. Model Training
```python
Trainer:
  ├── Initialize 5 models
  ├── Train on training set
  ├── Validate on validation set
  ├── Evaluate on test set
  ├── Compare metrics
  └── Save best model
```

### 3. Prediction
```python
Predictor:
  ├── Load trained model
  ├── Load preprocessor
  ├── Transform input data
  ├── Make prediction
  ├── Calculate confidence
  └── Return probabilities
```

---

## 🔮 Future Enhancements

Potential improvements:
- [ ] Deep learning models (CNN, LSTM)
- [ ] Time series analysis for light curves
- [ ] Ensemble stacking
- [ ] AutoML hyperparameter tuning
- [ ] Model explainability (SHAP, LIME)
- [ ] User authentication
- [ ] Database for predictions history
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] Real-time data updates

---

## 📚 References

### Datasets
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Kepler Mission: https://www.nasa.gov/mission_pages/kepler
- TESS Mission: https://tess.mit.edu/
- K2 Mission: https://www.nasa.gov/mission_pages/kepler/main/k2-mission

### Technologies
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/

---

## 🏆 Challenge Compliance

This project fulfills all NASA Space Apps Challenge requirements:

✅ **AI/ML Model**: 5 different models trained
✅ **NASA Datasets**: Kepler, TESS, K2 integrated
✅ **Web Interface**: Full-featured React application
✅ **User Interaction**: Upload, predict, train, explore
✅ **Model Accuracy**: Displayed in UI
✅ **Hyperparameter Control**: Configurable via code
✅ **Data Upload**: CSV upload supported
✅ **Manual Entry**: Single prediction form
✅ **Statistics**: Comprehensive metrics dashboard
✅ **Open Source**: All code available

---

## 💡 Innovation Points

1. **Multi-Model Ensemble**: Train 5 models simultaneously
2. **Real-Time Monitoring**: Watch training progress live
3. **Complete Pipeline**: From raw data to predictions
4. **Professional UI**: Production-ready interface
5. **Comprehensive API**: 15+ endpoints
6. **Feature Engineering**: Automated selection
7. **Batch Processing**: Handle large datasets
8. **Model Comparison**: Side-by-side metrics
9. **Visualization**: Interactive charts
10. **Developer Tools**: Scripts for easy setup

---

**Built with passion for space exploration and AI! 🚀🌌**

*For NASA Space Apps Challenge 2025*
