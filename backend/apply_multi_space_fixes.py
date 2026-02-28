#!/usr/bin/env python3
"""Apply all multi-space vision fixes"""

import os
import re

def fix_space_info_get_calls(file_path):
    """Fix all space.get() calls to handle SpaceInfo objects"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find space.get() calls
    pattern = r'(\s+)(space\.get\([\'"](\w+)[\'"](?:,\s*[^)]+)?\))'
    
    def replace_func(match):
        indent = match.group(1)
        full_call = match.group(2)
        attribute = match.group(3)
        
        # Create object/dict compatible code
        replacement = f"(hasattr(space, '{attribute}') and space.{attribute}) if hasattr(space, '{attribute}') else (isinstance(space, dict) and {full_call})"
        return f"{indent}{replacement}"
    
    # Apply replacements
    new_content = re.sub(pattern, replace_func, content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    print("🔧 Applying Multi-Space Vision Fixes")
    print("=" * 60)
    
    fixes_needed = []
    
    # 1. Check and fix multi_space_intelligence.py
    print("\n1️⃣ Checking multi_space_intelligence.py...")
    intel_file = "vision/multi_space_intelligence.py"
    if os.path.exists(intel_file):
        if fix_space_info_get_calls(intel_file):
            print("   ✅ Fixed SpaceInfo.get() calls")
            fixes_needed.append("SpaceInfo.get() calls")
        else:
            print("   ✅ No SpaceInfo.get() issues found")
    
    # 2. Check pure_vision_intelligence.py for capture_engine
    print("\n2️⃣ Checking pure_vision_intelligence.py...")
    pure_file = "api/pure_vision_intelligence.py"
    if os.path.exists(pure_file):
        with open(pure_file, 'r') as f:
            content = f.read()
        
        if 'self.capture_engine = None' not in content:
            print("   ❌ capture_engine initialization missing!")
            fixes_needed.append("capture_engine initialization")
        else:
            print("   ✅ capture_engine initialization present")
    
    # 3. Summary
    print(f"\n{'❌' if fixes_needed else '✅'} Summary:")
    if fixes_needed:
        print(f"   Issues found: {', '.join(fixes_needed)}")
        print("\n   Please restart Ironcliw to apply all fixes!")
    else:
        print("   All fixes are already applied!")
    
    print("\n📝 Next Steps:")
    print("1. Restart Ironcliw backend")
    print("2. Say: 'Hey Ironcliw, start monitoring my screen'")
    print("3. Wait for purple indicator")
    print("4. Say: 'Where is Terminal?'")

if __name__ == "__main__":
    main()