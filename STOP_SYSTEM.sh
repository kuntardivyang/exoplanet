#!/bin/bash

# Exoplanet Detection System - Stop Script
# This script stops all components of the system

echo "============================================================"
echo "🛑 EXOPLANET DETECTION SYSTEM - SHUTDOWN"
echo "============================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

stopped_count=0

# Function to stop a process by PID file
stop_by_pidfile() {
    local service_name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null
            echo -e "${GREEN}✓ Stopped $service_name (PID: $PID)${NC}"
            ((stopped_count++))
        else
            echo -e "${YELLOW}⚠ $service_name process not found${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠ No PID file for $service_name${NC}"
    fi
}

# Function to kill processes by port
kill_by_port() {
    local port=$1
    local service_name=$2

    PID=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        kill $PID 2>/dev/null
        echo -e "${GREEN}✓ Stopped $service_name on port $port (PID: $PID)${NC}"
        ((stopped_count++))
    fi
}

# Stop by PID files first
echo "Stopping services by PID files..."
echo "-----------------------------------------------------------"
stop_by_pidfile "Backend" "logs/backend.pid"
stop_by_pidfile "Frontend" "logs/frontend.pid"
stop_by_pidfile "Ollama" "logs/ollama.pid"

echo ""
echo "Checking ports..."
echo "-----------------------------------------------------------"

# Kill any remaining processes on known ports
kill_by_port 8000 "Backend (port 8000)"
kill_by_port 3000 "Frontend (port 3000)"
kill_by_port 11434 "Ollama (port 11434)"

echo ""
echo "Cleaning up..."
echo "-----------------------------------------------------------"

# Kill any python processes running the backend
pkill -f "backend/api/main.py" 2>/dev/null && echo -e "${GREEN}✓ Killed backend processes${NC}" && ((stopped_count++))

# Kill any npm/vite processes
pkill -f "vite" 2>/dev/null && echo -e "${GREEN}✓ Killed frontend processes${NC}" && ((stopped_count++))

echo ""
echo "============================================================"
if [ $stopped_count -gt 0 ]; then
    echo -e "${GREEN}✅ SHUTDOWN COMPLETE ($stopped_count processes stopped)${NC}"
else
    echo -e "${YELLOW}⚠ No running processes found${NC}"
fi
echo "============================================================"
