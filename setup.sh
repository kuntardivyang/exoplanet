#!/bin/bash

# Exoplanet Detection System - Setup Script
# NASA Space Apps Challenge 2025

echo "🚀 Exoplanet Detection System Setup"
echo "===================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check Node.js version
echo "📋 Checking Node.js version..."
node_version=$(node --version 2>&1)
echo "   Found Node.js $node_version"

echo ""
echo "🔧 Setting up Backend..."
echo "------------------------"

# Create virtual environment
echo "   Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "   Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "   Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎨 Setting up Frontend..."
echo "------------------------"

# Install Node dependencies
cd frontend
echo "   Installing Node.js dependencies..."
npm install

cd ..

echo ""
echo "✅ Setup Complete!"
echo "=================="
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Train your first model:"
echo "   source venv/bin/activate"
echo "   cd backend/models"
echo "   python train_pipeline.py"
echo ""
echo "2. Start the backend API:"
echo "   cd backend/api"
echo "   python main.py"
echo ""
echo "3. Start the frontend (in a new terminal):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "🌟 Access the application:"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Happy exoplanet hunting! 🪐✨"
