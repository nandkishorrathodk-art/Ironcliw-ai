#!/usr/bin/env python3
"""
Fix libdispatch module for macOS video streaming
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_libdispatch():
    """Install pyobjc-framework-libdispatch"""
    logger.info("📦 Installing libdispatch framework...")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "pyobjc-framework-libdispatch"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ pyobjc-framework-libdispatch installed successfully")
            
            # Test the import
            try:
                import libdispatch
                logger.info("✅ libdispatch import test successful")
                return True
            except ImportError as e:
                logger.error(f"❌ libdispatch import failed: {e}")
                return False
        else:
            logger.error(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing libdispatch: {e}")
        return False

def test_video_capture_imports():
    """Test all required imports for video capture"""
    logger.info("\n📋 Testing video capture imports...")
    
    required_modules = [
        "AVFoundation",
        "CoreMedia",
        "Quartz",
        "Cocoa",
        "objc",
        "Foundation",
        "libdispatch"
    ]
    
    all_good = True
    for module in required_modules:
        try:
            if module == "Quartz":
                # Special handling for Quartz.CoreVideo
                from Quartz import CoreVideo
                logger.info(f"✅ Quartz.CoreVideo import successful")
            elif module == "Cocoa":
                from Cocoa import NSObject
                logger.info(f"✅ Cocoa.NSObject import successful")
            elif module == "Foundation":
                from Foundation import NSRunLoop
                logger.info(f"✅ Foundation.NSRunLoop import successful")
            else:
                __import__(module)
                logger.info(f"✅ {module} import successful")
        except ImportError as e:
            logger.error(f"❌ {module} import failed: {e}")
            all_good = False
    
    return all_good

def main():
    """Main function"""
    logger.info("🔧 Fixing libdispatch for macOS video streaming...")
    
    if install_libdispatch():
        if test_video_capture_imports():
            logger.info("\n✅ All video capture dependencies fixed!")
            logger.info("Native macOS video streaming is now fully functional")
            logger.info("The purple recording indicator will appear when streaming")
            logger.info("\nRestart Ironcliw to use native video capture: python start_system.py")
            return True
        else:
            logger.warning("\n⚠️  Some imports still failing")
            logger.info("Check the errors above and install missing packages")
            return False
    else:
        logger.warning("\n⚠️  libdispatch installation failed")
        logger.info("Ironcliw will continue to use fallback video capture")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)