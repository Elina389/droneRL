#!/bin/bash
# Swarm Simulation Launcher
# This script helps you run different parts of your swarm simulation

# Set the correct Python path
PYTHON_CMD="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
export PYTHONPATH="swarm-env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚁 Swarm Simulation Launcher${NC}"
echo "=========================================="
echo ""

if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./run_simulation.sh web          # Start web visualization (recommended)"
    echo "                                   # Open http://127.0.0.1:8001 in browser"
    echo "                                   # Choose from modes:"
    echo "                                   #   - known_map: 2D with known obstacles"
    echo "                                   #   - unknown_terrain: 2D disaster discovery"
    echo "                                   #   - ppo_3d: 3D altitude-aware search"
    echo "  ./run_simulation.sh desktop      # Run desktop matplotlib visualization"  
    echo "  ./run_simulation.sh test         # Test consensus implementation"
    echo "  ./run_simulation.sh train-2d     # Train 2D consensus model"
    echo "  ./run_simulation.sh train-3d     # Train 3D consensus model"
    echo "  ./run_simulation.sh train-both   # Train both models"
    echo "  ./run_simulation.sh ablation     # Run ablation study"
    echo ""
    exit 1
fi

case "$1" in
    web)
        echo -e "${GREEN}🌐 Starting web server...${NC}"
        echo "Open http://127.0.0.1:8001 in your browser"
        echo "Press Ctrl+C to stop"
        echo ""
        $PYTHON_CMD -m uvicorn server:app --reload --host 127.0.0.1 --port 8001
        ;;
        
    desktop)
        echo -e "${GREEN}📊 Running desktop visualization...${NC}"
        $PYTHON_CMD visualize_swarm.py
        ;;
        
    test)
        echo -e "${YELLOW}🧪 Testing consensus implementation...${NC}"
        $PYTHON_CMD test_consensus_implementation.py
        ;;
        
    train-2d)
        echo -e "${GREEN}🤖 Training 2D consensus model...${NC}"
        $PYTHON_CMD train_consensus_models.py --model 2d --consensus
        ;;
        
    train-3d)
        echo -e "${GREEN}🚁 Training 3D consensus model...${NC}"
        $PYTHON_CMD train_consensus_models.py --model 3d --consensus
        ;;
        
    train-both)
        echo -e "${GREEN}🤖🚁 Training both 2D and 3D models...${NC}"
        $PYTHON_CMD train_consensus_models.py --model both --consensus
        ;;
        
    ablation)
        echo -e "${YELLOW}🔬 Running ablation studies...${NC}"
        $PYTHON_CMD train_consensus_models.py --model both --ablation
        ;;
        
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Run without arguments to see usage"
        exit 1
        ;;
esac