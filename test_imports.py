import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []
successes = []

def test_import(name, module_path):
    try:
        parts = module_path.split(".")
        mod = __import__(module_path, fromlist=[parts[-1]])
        successes.append(name)
        return mod
    except Exception as e:
        errors.append((name, str(e)))
        return None

test_import("platform_abstraction", "backend.core.platform_abstraction")
test_import("system_commands", "backend.core.system_commands")
test_import("voice_unlock_api", "backend.api.voice_unlock_api")
test_import("fastapi", "fastapi")
test_import("anthropic", "anthropic")
test_import("mss", "mss")
test_import("pyttsx3", "pyttsx3")
test_import("pyperclip", "pyperclip")
test_import("pyautogui", "pyautogui")
test_import("pynput", "pynput")
test_import("pystray", "pystray")
test_import("aiohttp", "aiohttp")

print(f"Successes ({len(successes)}): {', '.join(successes)}")
print()
print(f"Errors ({len(errors)}):")
for name, err in errors:
    print(f"  [{name}]: {err[:120]}")
