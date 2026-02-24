import sys
print("Python version:", sys.version)
print()

packages = [
    "torch",
    "mss",
    "pyttsx3",
    "fastapi",
    "anthropic",
    "transformers"
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {pkg}: {version}")
    except ImportError as e:
        print(f"[FAIL] {pkg}: NOT INSTALLED")
