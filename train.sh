#!/bin/bash

# Exoplanet Detection System - Training Script
# Quick script to train models

echo "🤖 Exoplanet Model Training"
echo "==========================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Ask for dataset selection
echo "Select dataset to train on:"
echo "1) Kepler (default)"
echo "2) TESS"
echo "3) K2"
echo ""
read -p "Enter choice (1-3) [1]: " choice
choice=${choice:-1}

dataset="kepler"
case $choice in
    1) dataset="kepler" ;;
    2) dataset="tess" ;;
    3) dataset="k2" ;;
    *)
        echo "Invalid choice, using Kepler"
        dataset="kepler"
        ;;
esac

echo ""
echo "🚀 Training models on $dataset dataset..."
echo ""

# Run training
cd backend/models
python train_pipeline.py

echo ""
echo "✅ Training Complete!"
echo ""
echo "📊 Check the following directories for results:"
echo "   - data/models/       (saved models)"
echo "   - data/processed/    (preprocessed data)"
echo "   - logs/              (training logs)"
echo ""
echo "🎯 Next: Start the API server to use the trained model"
echo "   ./run.sh"
