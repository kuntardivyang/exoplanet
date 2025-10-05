# 🚀 Quick Start Guide

Get up and running with the Exoplanet Detection System in 5 minutes!

## Step 1: Setup (One Time)

Run the setup script to install all dependencies:

```bash
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all Python dependencies (scikit-learn, XGBoost, FastAPI, etc.)
- Install all Node.js dependencies (React, Vite, Tailwind, etc.)

## Step 2: Train Your First Model

Train models on the Kepler dataset:

```bash
./train.sh
```

This will:
- Load the Kepler dataset (~9,500 exoplanet candidates)
- Preprocess the data (clean, normalize, feature selection)
- Train 5 different ML models (Random Forest, XGBoost, LightGBM, Neural Network, Gradient Boosting)
- Evaluate and compare all models
- Save the best performing model

**Training takes 5-15 minutes depending on your hardware.**

## Step 3: Run the System

Start both backend and frontend servers:

```bash
./run.sh
```

This will start:
- **Backend API** at http://localhost:8000
- **Frontend UI** at http://localhost:3000

Open your browser and go to: **http://localhost:3000**

## What You Can Do

### 1. Dashboard
- View system status and statistics
- See dataset information
- View feature importance charts

### 2. Make Predictions
Three ways to predict:
- **Single**: Enter feature values manually
- **Batch**: Add multiple samples
- **Upload**: Upload a CSV file

### 3. Train Models
- Select dataset (Kepler, TESS, K2)
- Monitor training progress in real-time
- View results and metrics

### 4. Explore Data
- Browse all three NASA datasets
- View sample data
- Explore features

### 5. Manage Models
- View all trained models
- Load different models
- Compare performance
- View feature importance

## Manual Steps (Alternative)

If you prefer to run commands manually:

### Backend
```bash
# Activate virtual environment
source venv/bin/activate

# Train model
cd backend/models
python train_pipeline.py

# Start API
cd ../api
python main.py
```

### Frontend
```bash
# In a new terminal
cd frontend
npm run dev
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Port Already in Use
If port 8000 or 3000 is already in use:

Backend:
```bash
# Edit backend/config/config.py and change API_PORT
```

Frontend:
```bash
# Edit frontend/vite.config.js and change server.port
```

### Missing Dependencies
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### Model Not Loading
Make sure you've trained at least one model first:
```bash
./train.sh
```

## Next Steps

1. **Explore the datasets** - Go to Data Explorer
2. **Make predictions** - Try the Predict page with example data
3. **Train on different datasets** - Try TESS or K2
4. **Compare models** - Train multiple times and compare results
5. **Upload your own data** - Use the CSV upload feature

## File Structure

```
exoplanet/
├── backend/         # Python backend
├── frontend/        # React frontend
├── data/
│   ├── raw/        # Your CSV datasets
│   ├── processed/  # Preprocessed data
│   └── models/     # Trained models
├── logs/           # Training logs
└── notebooks/      # Jupyter notebooks
```

## System Requirements

- **Python**: 3.9 or higher
- **Node.js**: 18 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB free space
- **OS**: Linux, macOS, or Windows (with WSL)

## Tips

1. **Start with Kepler** - It has the most complete data
2. **Monitor training** - Use the Training page to watch progress
3. **Check logs** - Look in `logs/` directory for detailed information
4. **Try batch predictions** - Upload CSV files for bulk processing
5. **Compare models** - Train on different datasets and compare results

## Support

- Check the README.md for detailed documentation
- Review API docs at /docs endpoint
- Check logs directory for error messages

---

**Happy exoplanet hunting! 🪐✨**
