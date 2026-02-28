#!/usr/bin/env python3
"""
Fix AVFoundation installation
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_pyobjc_avfoundation():
    """Install only the necessary pyobjc frameworks"""
    logger.info("📦 Installing pyobjc-framework-AVFoundation...")
    
    # These are the essential packages for video capture
    packages = [
        "pyobjc-framework-AVFoundation",
        "pyobjc-framework-Cocoa",
        "pyobjc-framework-CoreMedia",
        "pyobjc-framework-Quartz"
    ]
    
    success = True
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            cmd = [sys.executable, "-m", "pip", "install", package]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ {package} installed successfully")
            else:
                logger.warning(f"⚠️  {package} installation failed: {result.stderr}")
                success = False
                
        except Exception as e:
            logger.error(f"❌ Error installing {package}: {e}")
            success = False
    
    # Test the imports
    if success:
        logger.info("\n📋 Testing imports...")
        test_imports = [
            "AVFoundation",
            "CoreMedia",
            "Quartz",
            "Cocoa"
        ]
        
        all_good = True
        for module in test_imports:
            try:
                __import__(module)
                logger.info(f"✅ {module} import successful")
            except ImportError as e:
                logger.error(f"❌ {module} import failed: {e}")
                all_good = False
        
        return all_good
    
    return False

def main():
    """Main function"""
    logger.info("🔧 Fixing AVFoundation installation...")
    
    if install_pyobjc_avfoundation():
        logger.info("\n✅ AVFoundation fixed successfully!")
        logger.info("Video streaming will now use native macOS capture with purple indicator")
        logger.info("Restart Ironcliw to use the native video capture: python start_system.py")
        return True
    else:
        logger.warning("\n⚠️  AVFoundation installation incomplete")
        logger.info("Ironcliw will use fallback video capture mode")
        logger.info("This still works but may have reduced performance")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)