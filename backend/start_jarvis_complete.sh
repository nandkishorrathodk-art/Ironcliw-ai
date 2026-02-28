#!/bin/bash
# Start Ironcliw with Voice Feedback Support
# ========================================

echo "🚀 Starting Ironcliw Complete System"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if already running
check_service() {
    local port=$1
    local name=$2
    
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name already running on port $port${NC}"
        return 0
    else
        return 1
    fi
}

# Kill existing processes
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up existing processes...${NC}"
    
    # Kill Ironcliw
    pkill -f "python main.py" 2>/dev/null
    
    # Kill WebSocket server
    pkill -f "websocket_server.py" 2>/dev/null
    
    sleep 2
}

# Start services
start_services() {
    echo -e "\n${YELLOW}🔧 Starting services...${NC}"
    
    # Start WebSocket server for voice unlock
    if ! check_service 8765 "Voice Unlock WebSocket"; then
        echo "Starting Voice Unlock WebSocket server..."
        cd voice_unlock/objc/server
        python websocket_server.py > /tmp/voice_unlock_ws.log 2>&1 &
        cd ../../../
        sleep 2
        
        if check_service 8765 "Voice Unlock WebSocket"; then
            echo -e "${GREEN}✅ Voice Unlock WebSocket started${NC}"
        else
            echo -e "${RED}❌ Failed to start Voice Unlock WebSocket${NC}"
            cat /tmp/voice_unlock_ws.log
        fi
    fi
    
    # Start Ironcliw
    if ! check_service 8000 "Ironcliw Backend"; then
        echo "Starting Ironcliw backend..."
        python main.py > /tmp/jarvis.log 2>&1 &
        
        echo "Waiting for Ironcliw to initialize..."
        for i in {1..10}; do
            if check_service 8000 "Ironcliw Backend"; then
                echo -e "${GREEN}✅ Ironcliw backend started${NC}"
                break
            fi
            echo -n "."
            sleep 2
        done
    fi
}

# Show status
show_status() {
    echo -e "\n${YELLOW}📊 System Status${NC}"
    echo "=================="
    
    check_service 8000 "Ironcliw Backend"
    check_service 8765 "Voice Unlock WebSocket"
    
    # Check if frontend is running
    if check_service 3000 "React Frontend"; then
        echo -e "${GREEN}✅ Frontend running at http://localhost:3000${NC}"
    else
        echo -e "${YELLOW}ℹ️  Frontend not running. Start with:${NC}"
        echo "   cd ../frontend && npm start"
    fi
}

# Show instructions
show_instructions() {
    echo -e "\n${GREEN}✅ Ironcliw is ready!${NC}"
    echo -e "\n📝 ${YELLOW}Voice Feedback Test Instructions:${NC}"
    echo "1. Open the React app at http://localhost:3000"
    echo "2. Lock your screen (Cmd+Ctrl+Q)"
    echo "3. Click the microphone and say: 'Ironcliw, open Safari and search for dogs'"
    echo "4. Listen for Ironcliw to say:"
    echo "   'I see your screen is locked. I'll unlock it now by typing in your password so I can search for dogs.'"
    echo ""
    echo -e "${YELLOW}🔊 Key improvements:${NC}"
    echo "- Ironcliw now detects when your screen is locked"
    echo "- Provides voice feedback BEFORE unlocking"
    echo "- Explains what it's about to do"
    echo "- Then unlocks and executes your command"
    echo ""
    echo -e "${YELLOW}📋 Logs:${NC}"
    echo "- Ironcliw: tail -f /tmp/jarvis.log"
    echo "- WebSocket: tail -f /tmp/voice_unlock_ws.log"
}

# Main execution
main() {
    clear
    
    case "${1:-}" in
        "clean")
            cleanup
            echo -e "${GREEN}✅ Cleanup complete${NC}"
            ;;
        "restart")
            cleanup
            start_services
            show_status
            show_instructions
            ;;
        *)
            start_services
            show_status
            show_instructions
            ;;
    esac
}

# Run
main "$@"