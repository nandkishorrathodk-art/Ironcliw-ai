#!/usr/bin/env python3
"""
Ironcliw Cross-Platform Dependency Verification Script
Verifies that all required dependencies are installed for the current platform.
Version: 1.0.0
"""

import sys
import platform
import importlib
from typing import List, Tuple, Dict
import os

# Set UTF-8 encoding for Windows console
if platform.system() == 'Windows':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Check marks (ASCII fallback for Windows)
if platform.system() == 'Windows':
    CHECK_MARK = '[OK]'
    CROSS_MARK = '[FAIL]'
else:
    CHECK_MARK = '✓'
    CROSS_MARK = '✗'

def get_platform() -> str:
    """Detect the current platform."""
    system = platform.system()
    if system == 'Darwin':
        return 'macOS'
    elif system == 'Windows':
        return 'Windows'
    elif system == 'Linux':
        return 'Linux'
    else:
        return 'Unknown'

def check_dependency(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Check if a Python module can be imported.
    
    Args:
        module_name: Name of the module to import
        package_name: Name of the package (for display purposes)
    
    Returns:
        Tuple of (success, message)
    """
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        return True, f"{Colors.GREEN}{CHECK_MARK}{Colors.RESET} {package_name}"
    except ImportError as e:
        return False, f"{Colors.RED}{CROSS_MARK}{Colors.RESET} {package_name} - {str(e)}"

def check_common_dependencies() -> List[Tuple[bool, str]]:
    """Check dependencies required on all platforms."""
    deps = [
        ('aiohttp', 'aiohttp'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('onnxruntime', 'onnxruntime'),
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('psutil', 'psutil'),
        ('pydantic', 'pydantic'),
        ('anthropic', 'anthropic'),
        ('openai', 'openai'),
    ]
    
    return [check_dependency(module, package) for package, module in deps]

def check_macos_dependencies() -> List[Tuple[bool, str]]:
    """Check macOS-specific dependencies."""
    deps = [
        ('coremltools', 'coremltools'),
    ]
    
    return [check_dependency(module, package) for package, module in deps]

def check_windows_dependencies() -> List[Tuple[bool, str]]:
    """Check Windows-specific dependencies."""
    deps = [
        ('wmi', 'wmi'),
        ('win32api', 'pywin32'),  # Note: module is win32api, package is pywin32
        ('comtypes', 'comtypes'),
        ('mss', 'mss'),
        ('pyttsx3', 'pyttsx3'),
        ('pyperclip', 'pyperclip'),
        ('pyautogui', 'pyautogui'),
        ('pynput', 'pynput'),
        ('pystray', 'pystray'),
    ]
    
    return [check_dependency(module, package) for module, package in deps]

def check_linux_dependencies() -> List[Tuple[bool, str]]:
    """Check Linux-specific dependencies."""
    deps = [
        ('Xlib', 'python-xlib'),
        ('mss', 'mss'),
        ('pyttsx3', 'pyttsx3'),
        ('pyperclip', 'pyperclip'),
        ('pyautogui', 'pyautogui'),
        ('pynput', 'pynput'),
        ('pystray', 'pystray'),
    ]
    
    return [check_dependency(module, package) for package, module in deps]

def check_cross_platform_dependencies() -> List[Tuple[bool, str]]:
    """Check cross-platform alternatives to macOS-specific tools."""
    deps = [
        ('mss', 'mss (screen capture)'),
        ('pyttsx3', 'pyttsx3 (TTS)'),
        ('pyperclip', 'pyperclip (clipboard)'),
        ('pyautogui', 'pyautogui (GUI automation)'),
        ('pynput', 'pynput (keyboard/mouse)'),
        ('pystray', 'pystray (system tray)'),
    ]
    
    return [check_dependency(module, package) for module, package in deps]

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")

def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{'-' * len(text)}")

def main():
    """Main verification routine."""
    current_platform = get_platform()
    system_info = platform.platform()
    python_version = sys.version.split()[0]
    
    print_header("Ironcliw Dependency Verification")
    
    print(f"{Colors.BOLD}Platform:{Colors.RESET} {current_platform}")
    print(f"{Colors.BOLD}System:{Colors.RESET} {system_info}")
    print(f"{Colors.BOLD}Python:{Colors.RESET} {python_version}")
    
    all_results: Dict[str, List[Tuple[bool, str]]] = {}
    
    # Check common dependencies
    print_section("Common Dependencies (All Platforms)")
    common_results = check_common_dependencies()
    all_results['Common'] = common_results
    for success, message in common_results:
        print(f"  {message}")
    
    # Check cross-platform dependencies
    print_section("Cross-Platform Dependencies (Windows/Linux alternatives)")
    cross_platform_results = check_cross_platform_dependencies()
    all_results['Cross-Platform'] = cross_platform_results
    for success, message in cross_platform_results:
        print(f"  {message}")
    
    # Check platform-specific dependencies
    if current_platform == 'macOS':
        print_section("macOS-Specific Dependencies")
        macos_results = check_macos_dependencies()
        all_results['macOS'] = macos_results
        for success, message in macos_results:
            print(f"  {message}")
    
    elif current_platform == 'Windows':
        print_section("Windows-Specific Dependencies")
        windows_results = check_windows_dependencies()
        all_results['Windows'] = windows_results
        for success, message in windows_results:
            print(f"  {message}")
    
    elif current_platform == 'Linux':
        print_section("Linux-Specific Dependencies")
        linux_results = check_linux_dependencies()
        all_results['Linux'] = linux_results
        for success, message in linux_results:
            print(f"  {message}")
    
    # Calculate statistics
    total_checks = sum(len(results) for results in all_results.values())
    successful_checks = sum(sum(1 for success, _ in results if success) for results in all_results.values())
    failed_checks = total_checks - successful_checks
    
    # Print summary
    print_section("Summary")
    print(f"  Total checks: {total_checks}")
    print(f"  {Colors.GREEN}Successful: {successful_checks}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {failed_checks}{Colors.RESET}")
    
    success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
    print(f"\n  Success rate: {success_rate:.1f}%")
    
    # Print installation instructions for missing dependencies
    if failed_checks > 0:
        print_section("Installation Instructions")
        print(f"\n{Colors.YELLOW}To install missing dependencies, run:{Colors.RESET}\n")
        print(f"  pip install -r requirements.txt")
        
        if current_platform == 'Linux':
            print(f"\n{Colors.YELLOW}For Linux system dependencies:{Colors.RESET}\n")
            print(f"  # Ubuntu/Debian:")
            print(f"  sudo apt-get install espeak xclip wmctrl xdotool")
            print(f"\n  # Fedora:")
            print(f"  sudo dnf install espeak xclip wmctrl xdotool")
            print(f"\n  # Arch:")
            print(f"  sudo pacman -S espeak-ng xclip wmctrl xdotool")
    
    print()
    
    # Exit with appropriate code
    sys.exit(0 if failed_checks == 0 else 1)

if __name__ == '__main__':
    main()
