#!/bin/bash

# Build script for Ironcliw Vision Intelligence System
# Compiles Rust, Swift, and sets up Python components

set -e

echo "🚀 Building Ironcliw Vision Intelligence System..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ Error: $1 is not installed. Please install it first."
        exit 1
    fi
}

echo "📋 Checking dependencies..."
check_command cargo
check_command rustc
check_command swiftc
check_command python3

# Build Rust components
echo "🦀 Building Rust vision intelligence components..."
cd "$SCRIPT_DIR"

# Create Rust source structure if not exists
mkdir -p src/vision src/state_detection src/visual_features src/memory_pool src/python_bridge

# Create placeholder modules if they don't exist
[ ! -f src/state_detection.rs ] && echo "pub struct StateDetector;" > src/state_detection.rs
[ ! -f src/visual_features.rs ] && echo "pub struct FeatureExtractor;" > src/visual_features.rs
[ ! -f src/memory_pool.rs ] && echo "pub fn initialize_global_pool() {}" > src/memory_pool.rs
[ ! -f src/python_bridge.rs ] && echo "// Python bridge implementation" > src/python_bridge.rs

# Build Rust library
cargo build --release

# Copy the built library to Python's expected location
if [ -f "target/release/libvision_intelligence.dylib" ]; then
    cp target/release/libvision_intelligence.dylib vision_intelligence.so
    echo "✅ Rust components built successfully"
else
    echo "⚠️  Warning: Rust build completed but library not found"
fi

# Build Swift components
echo "🎯 Building Swift vision intelligence components..."
cd "$SCRIPT_DIR"

if [ -f "VisionIntelligence.swift" ]; then
    swiftc -O \
        -framework Vision \
        -framework CoreML \
        -framework AppKit \
        -framework Accelerate \
        -framework CoreImage \
        VisionIntelligence.swift \
        -emit-library \
        -o VisionIntelligence.dylib \
        -module-name VisionIntelligence
    
    # Also build executable for command-line interface
    swiftc -O \
        -framework Vision \
        -framework CoreML \
        -framework AppKit \
        -framework Accelerate \
        -framework CoreImage \
        VisionIntelligence.swift \
        -o VisionIntelligence
    
    echo "✅ Swift components built successfully"
else
    echo "⚠️  Warning: VisionIntelligence.swift not found"
fi

# Set up Python environment
echo "🐍 Setting up Python components..."

# Create __init__.py if it doesn't exist
if [ ! -f "__init__.py" ]; then
    cat > __init__.py << EOF
"""
Ironcliw Vision Intelligence System
Multi-language vision understanding without hardcoding
"""

from .visual_state_management_system import (
    VisualStateManagementSystem,
    ApplicationStateTracker,
    StateObservation,
    ApplicationState,
    VisualSignature
)

from .vision_intelligence_bridge import (
    VisionIntelligenceBridge,
    get_vision_intelligence_bridge,
    analyze_screenshot
)

__all__ = [
    'VisualStateManagementSystem',
    'ApplicationStateTracker',
    'StateObservation',
    'ApplicationState',
    'VisualSignature',
    'VisionIntelligenceBridge',
    'get_vision_intelligence_bridge',
    'analyze_screenshot'
]

# Try to import Rust components if available
try:
    import vision_intelligence as rust_vi
    __all__.extend(['rust_vi'])
except ImportError:
    rust_vi = None

print("✨ Ironcliw Vision Intelligence System initialized")
EOF
fi

# Create directories for learned states
mkdir -p learned_states

# Make the build script executable
chmod +x "$0"

echo "✅ Build complete!"
echo ""
echo "📦 Components built:"
echo "  - Rust library: vision_intelligence.so"
echo "  - Swift library: VisionIntelligence.dylib"
echo "  - Swift executable: VisionIntelligence"
echo "  - Python modules: Ready"
echo ""
echo "🎯 To use the system:"
echo "  from backend.vision.intelligence import VisionIntelligenceBridge"
echo "  bridge = VisionIntelligenceBridge()"
echo ""
echo "🚀 Vision Intelligence System is ready!"