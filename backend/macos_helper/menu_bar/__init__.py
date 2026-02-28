"""
Ironcliw macOS Helper - Menu Bar Status Indicator

Native macOS menu bar integration for Ironcliw status display and control.

Features:
- Real-time status indicator (icon + text)
- Dynamic menu with live updates
- Permission status display
- Quick actions (pause, resume, settings)
- Activity monitor integration
- System tray notifications

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Menu Bar Status Indicator                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │ 🤖 Ironcliw ▼                                                 ││
    │  ├─────────────────────────────────────────────────────────────┤│
    │  │ Status: Online                            [●]                ││
    │  │ ─────────────────────────────────────────                   ││
    │  │ Monitoring                                                   ││
    │  │   Apps: 12 active                                            ││
    │  │   Windows: 34 tracked                                        ││
    │  │   Notifications: 8 pending                                   ││
    │  │ ─────────────────────────────────────────                   ││
    │  │ Quick Actions                                                ││
    │  │   ⏸ Pause Monitoring                                         ││
    │  │   🔄 Restart Service                                         ││
    │  │   ⚙️ Open Settings                                           ││
    │  │ ─────────────────────────────────────────                   ││
    │  │ Permissions                                                  ││
    │  │   ✅ Accessibility                                            ││
    │  │   ✅ Screen Recording                                         ││
    │  │   ⚠️ Microphone (click to grant)                             ││
    │  │ ─────────────────────────────────────────                   ││
    │  │ About Ironcliw                                                 ││
    │  │ Quit                                                         ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘

Apple Compliance:
- Uses NSStatusBar (standard macOS API)
- No private frameworks
- Respects system appearance (dark/light mode)
- Proper memory management via ARC

Usage:
    from macos_helper.menu_bar import get_menu_bar, start_menu_bar

    # Start the menu bar indicator
    menu_bar = await start_menu_bar()

    # Update status
    menu_bar.set_status("Online", activity="Processing request...")

    # Stop
    await stop_menu_bar()
"""

from .status_indicator import (
    MenuBarIndicator,
    MenuBarState,
    StatusIcon,
    get_menu_bar,
    start_menu_bar,
    stop_menu_bar,
)

__all__ = [
    "MenuBarIndicator",
    "MenuBarState",
    "StatusIcon",
    "get_menu_bar",
    "start_menu_bar",
    "stop_menu_bar",
]
