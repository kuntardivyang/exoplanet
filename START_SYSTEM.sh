#!/bin/bash

# Exoplanet Detection System - Startup Script
# This script starts all components of the system

echo "============================================================"
echo "🌌 EXOPLANET DETECTION SYSTEM - STARTUP"
echo "============================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "backend/api/main.py" ]; then
    echo -e "${RED}Error: Please run this script from the exoplanet project root directory${NC}"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i:$1 >/dev/null 2>&1
}

echo "Step 1: Checking Prerequisites..."
echo "-----------------------------------------------------------"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
else
    python_version=$(python3 --version)
    echo -e "${GREEN}✓ $python_version${NC}"
fi

# Check Node/npm
if ! command_exists npm; then
    echo -e "${YELLOW}⚠ npm not found - frontend won't start${NC}"
    SKIP_FRONTEND=true
else
    node_version=$(node --version)
    echo -e "${GREEN}✓ Node $node_version${NC}"
fi

# Check Ollama (optional)
if ! command_exists ollama; then
    echo -e "${YELLOW}⚠ Ollama not found - chat features will be limited${NC}"
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
    OLLAMA_AVAILABLE=false
else
    echo -e "${GREEN}✓ Ollama installed${NC}"
    OLLAMA_AVAILABLE=true
fi

echo ""
echo "Step 2: Checking Models..."
echo "-----------------------------------------------------------"

if [ -d "data/models" ] && [ "$(ls -A data/models/*.pkl 2>/dev/null)" ]; then
    model_count=$(ls data/models/*.pkl 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $model_count trained model files${NC}"
else
    echo -e "${YELLOW}⚠ No trained models found${NC}"
    echo "  Run: python3 backend/models/train_pipeline.py"
fi

echo ""
echo "Step 3: Starting Services..."
echo "-----------------------------------------------------------"

# Check if backend is already running
if port_in_use 8000; then
    echo -e "${YELLOW}⚠ Backend already running on port 8000${NC}"
else
    echo "Starting Backend API..."
    python3 backend/api/main.py > logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
    sleep 3
fi

# Start Ollama if available and not running
if [ "$OLLAMA_AVAILABLE" = true ]; then
    if port_in_use 11434; then
        echo -e "${GREEN}✓ Ollama already running${NC}"
    else
        echo "Starting Ollama..."
        ollama serve > logs/ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo -e "${GREEN}✓ Ollama started (PID: $OLLAMA_PID)${NC}"
        sleep 2

        # Check if llama3.2 is pulled
        if ollama list 2>/dev/null | grep -q "llama3.2"; then
            echo -e "${GREEN}✓ llama3.2 model available${NC}"
        else
            echo -e "${YELLOW}⚠ Pulling llama3.2 model (this may take a while)...${NC}"
            ollama pull llama3.2 &
            PULL_PID=$!
            echo "  Running in background (PID: $PULL_PID)"
        fi
    fi
fi

# Start Frontend if npm available
if [ "$SKIP_FRONTEND" != true ]; then
    if port_in_use 3000; then
        echo -e "${YELLOW}⚠ Frontend already running on port 3000${NC}"
    else
        echo "Starting Frontend..."
        cd frontend

        # Check if node_modules exists
        if [ ! -d "node_modules" ]; then
            echo "  Installing npm dependencies..."
            npm install > ../logs/npm_install.log 2>&1
        fi

        npm run dev > ../logs/frontend.log 2>&1 &
        FRONTEND_PID=$!
        cd ..
        echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
        sleep 3
    fi
fi

echo ""
echo "============================================================"
echo "✅ SYSTEM STARTED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Access Points:"
echo "-----------------------------------------------------------"
echo -e "${GREEN}📊 Frontend:${NC}     http://localhost:3000"
echo -e "${GREEN}🔧 API Docs:${NC}     http://localhost:8000/docs"
echo -e "${GREEN}📖 ReDoc:${NC}        http://localhost:8000/redoc"
echo -e "${GREEN}💚 Health Check:${NC} http://localhost:8000/health"
echo ""

if [ "$OLLAMA_AVAILABLE" = true ]; then
    echo -e "${GREEN}💬 Chat Status:${NC}  http://localhost:8000/chat/status"
    echo ""
fi

echo "Logs:"
echo "-----------------------------------------------------------"
echo "Backend:  tail -f logs/backend.log"
echo "Frontend: tail -f logs/frontend.log"
if [ "$OLLAMA_AVAILABLE" = true ]; then
    echo "Ollama:   tail -f logs/ollama.log"
fi
echo ""

echo "Stop All Services:"
echo "-----------------------------------------------------------"
echo "Run: ./STOP_SYSTEM.sh"
echo ""

# Save PIDs for stopping later
echo "$BACKEND_PID" > logs/backend.pid 2>/dev/null
echo "$FRONTEND_PID" > logs/frontend.pid 2>/dev/null
echo "$OLLAMA_PID" > logs/ollama.pid 2>/dev/null

echo "============================================================"
echo "🚀 Happy exoplanet hunting!"
echo "============================================================"
