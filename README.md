# 🚀 Exoplanet Detection System

AI-powered exoplanet detection and classification system built for NASA Space Apps Challenge 2025.

## 🌟 Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Neural Networks, Gradient Boosting
- **Three NASA Datasets**: Kepler, TESS, and K2 missions
- **Complete Pipeline**: Data preprocessing, feature engineering, model training, and evaluation
- **Web Interface**: Interactive React-based UI for predictions and model management
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Real-time Training**: Monitor model training progress in real-time
- **Batch Predictions**: Support for CSV upload and batch processing
- **Feature Importance**: Visualize which features matter most for detection

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

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip
- npm or yarn

### Backend Setup

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train your first model**:
```bash
cd backend/models
python train_pipeline.py
```

This will:
- Load the Kepler dataset
- Preprocess and clean the data
- Train 5 different ML models
- Evaluate and save the best model
- Generate comparison reports

4. **Start the API server**:
```bash
cd backend/api
python main.py
```

API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Start development server**:
```bash
npm run dev
```

Frontend will be available at `http://localhost:3000`

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
