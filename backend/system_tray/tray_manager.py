"""Windows/Linux system tray icon for Ironcliw-AI."""
import sys
import threading
import logging

logger = logging.getLogger(__name__)


def start_tray_icon(on_show=None, on_quit=None):
    """Start Ironcliw-AI system tray icon (Windows/Linux via pystray)."""
    if sys.platform not in ("win32", "linux"):
        logger.info("System tray not supported on this platform")
        return None

    try:
        import pystray
        from PIL import Image, ImageDraw

        def create_icon():
            img = Image.new("RGB", (64, 64), color=(0, 100, 200))
            draw = ImageDraw.Draw(img)
            draw.ellipse([8, 8, 56, 56], fill=(0, 150, 255))
            return img

        menu = pystray.Menu(
            pystray.MenuItem("Open Ironcliw-AI", on_show or (lambda: None)),
            pystray.MenuItem("Quit", on_quit or (lambda: None)),
        )

        icon = pystray.Icon("Ironcliw-AI", create_icon(), "Ironcliw-AI", menu)

        tray_thread = threading.Thread(target=icon.run, daemon=True)
        tray_thread.start()

        logger.info("System tray icon started")
        return icon

    except ImportError:
        logger.warning(
            "pystray or Pillow not installed — no system tray. "
            "pip install pystray Pillow"
        )
        return None
    except Exception as e:
        logger.warning(f"System tray failed: {e}")
        return None
