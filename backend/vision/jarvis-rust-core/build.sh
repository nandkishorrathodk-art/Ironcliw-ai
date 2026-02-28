#!/bin/bash

# Build script for Ironcliw Rust core

echo "Building Ironcliw Rust core with advanced features..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build in release mode with all optimizations
echo "Building release version..."
maturin build --release

# Develop mode for testing
echo "Installing in development mode..."
maturin develop --release

echo "Build complete!"

# Run tests
echo "Running tests..."
cargo test --release

echo "✅ Ironcliw Rust core built successfully!"