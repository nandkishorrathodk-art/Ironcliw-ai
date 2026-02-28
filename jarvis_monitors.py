#!/usr/bin/env python3
"""
Ironcliw Multi-Monitor CLI

Command-line interface for multi-monitor support functionality.
Provides easy access to display detection, space mapping, and capture operations.

Usage:
    python jarvis_monitors.py --detect
    python jarvis_monitors.py --capture
    python jarvis_monitors.py --summary
    python jarvis_monitors.py --performance

Author: Derek Russell
Date: 2025-01-14
Branch: multi-monitor-support
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from vision.multi_monitor_detector import (
    MultiMonitorDetector,
    detect_all_monitors,
    capture_multi_monitor_screenshots,
    get_monitor_summary,
    MACOS_AVAILABLE
)


def print_banner():
    """Print CLI banner"""
    print("🖥️  Ironcliw Multi-Monitor Support CLI")
    print("=" * 50)


def print_displays(displays, detailed=False):
    """Print display information in a formatted way"""
    if not displays:
        print("❌ No displays detected")
        return
    
    print(f"📺 Found {len(displays)} display(s):")
    print()
    
    for i, display in enumerate(displays, 1):
        primary_marker = " [PRIMARY]" if display.is_primary else ""
        print(f"  {i}. {display.name}{primary_marker}")
        print(f"     Resolution: {display.resolution[0]}x{display.resolution[1]}")
        print(f"     Position: ({display.position[0]}, {display.position[1]})")
        
        if detailed:
            print(f"     Refresh Rate: {display.refresh_rate} Hz")
            print(f"     Color Depth: {display.color_depth} bits")
            print(f"     Active Space: {display.active_space}")
            print(f"     Spaces: {display.spaces}")
            print(f"     Last Updated: {time.ctime(display.last_updated)}")
        
        print()


def print_space_mappings(mappings):
    """Print space-display mappings"""
    if not mappings:
        print("❌ No space mappings available")
        return
    
    print(f"🗺️  Space-Display Mappings:")
    print()
    
    for space_id, display_id in mappings.items():
        print(f"  Space {space_id} → Display {display_id}")
    
    print()


def print_capture_result(result):
    """Print capture result information"""
    if result.success:
        print(f"✅ Capture successful!")
        print(f"   Displays captured: {len(result.displays_captured)}/{result.total_displays}")
        print(f"   Capture time: {result.capture_time:.2f}s")
        
        if result.displays_captured:
            print(f"   Screenshot details:")
            for display_id, screenshot in result.displays_captured.items():
                print(f"     Display {display_id}: {screenshot.shape} ({screenshot.dtype})")
        
        if result.failed_displays:
            print(f"   Failed displays: {result.failed_displays}")
    else:
        print(f"❌ Capture failed: {result.error}")
    
    print()


def print_performance_stats(stats):
    """Print performance statistics"""
    print("📊 Performance Statistics:")
    print()
    
    capture_stats = stats["capture_stats"]
    print(f"  Capture Operations:")
    print(f"    Total captures: {capture_stats['total_captures']}")
    print(f"    Successful: {capture_stats['successful_captures']}")
    print(f"    Failed: {capture_stats['failed_captures']}")
    print(f"    Average time: {capture_stats['average_capture_time']:.2f}s")
    print()
    
    print(f"  Cache Information:")
    print(f"    Displays cached: {stats['displays_cached']}")
    print(f"    Space mappings cached: {stats['space_mappings_cached']}")
    print(f"    Cache age: {stats['cache_age']:.1f}s")
    print()
    
    print(f"  System Information:")
    print(f"    macOS available: {MACOS_AVAILABLE}")
    print(f"    Last detection: {time.ctime(stats['last_detection_time'])}")


async def cmd_detect(args):
    """Handle detect command"""
    print("🔍 Detecting displays...")
    
    try:
        displays = await detect_all_monitors()
        print_displays(displays, detailed=args.detailed)
        
        if args.json:
            json_output = [
                {
                    "id": d.display_id,
                    "name": d.name,
                    "resolution": d.resolution,
                    "position": d.position,
                    "is_primary": d.is_primary,
                    "refresh_rate": d.refresh_rate,
                    "color_depth": d.color_depth,
                    "spaces": d.spaces,
                    "active_space": d.active_space
                }
                for d in displays
            ]
            print(json.dumps(json_output, indent=2))
        
    except Exception as e:
        print(f"❌ Error detecting displays: {e}")
        return 1
    
    return 0


async def cmd_capture(args):
    """Handle capture command"""
    print("📸 Capturing screenshots from all displays...")
    
    try:
        result = await capture_multi_monitor_screenshots()
        print_capture_result(result)
        
        if args.json:
            json_output = {
                "success": result.success,
                "displays_captured": len(result.displays_captured),
                "failed_displays": result.failed_displays,
                "capture_time": result.capture_time,
                "total_displays": result.total_displays,
                "error": result.error,
                "metadata": result.metadata
            }
            print(json.dumps(json_output, indent=2))
        
    except Exception as e:
        print(f"❌ Error capturing displays: {e}")
        return 1
    
    return 0


async def cmd_summary(args):
    """Handle summary command"""
    print("📋 Getting display summary...")
    
    try:
        summary = await get_monitor_summary()
        
        print(f"Total displays: {summary.get('total_displays', 0)}")
        print()
        
        displays = summary.get('displays', [])
        for display in displays:
            primary_marker = " [PRIMARY]" if display.get('is_primary', False) else ""
            print(f"  {display.get('name', 'Unknown')}{primary_marker}")
            print(f"    Resolution: {display.get('resolution', [0, 0])[0]}x{display.get('resolution', [0, 0])[1]}")
            print(f"    Spaces: {display.get('spaces', [])}")
        
        print()
        
        mappings = summary.get('space_mappings', {})
        print_space_mappings(mappings)
        
        if args.json:
            print(json.dumps(summary, indent=2))
        
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
        return 1
    
    return 0


async def cmd_performance(args):
    """Handle performance command"""
    print("📊 Getting performance statistics...")
    
    try:
        detector = MultiMonitorDetector()
        stats = detector.get_performance_stats()
        print_performance_stats(stats)
        
        if args.json:
            print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"❌ Error getting performance stats: {e}")
        return 1
    
    return 0


async def cmd_test(args):
    """Handle test command - run comprehensive tests"""
    print("🧪 Running multi-monitor tests...")
    
    try:
        detector = MultiMonitorDetector()
        
        print("1. Testing display detection...")
        displays = await detector.detect_displays()
        print(f"   ✅ Detected {len(displays)} displays")
        
        print("2. Testing space mappings...")
        mappings = await detector.get_space_display_mapping()
        print(f"   ✅ Mapped {len(mappings)} spaces")
        
        print("3. Testing capture...")
        result = await detector.capture_all_displays()
        if result.success:
            print(f"   ✅ Captured {len(result.displays_captured)} displays")
        else:
            print(f"   ⚠️  Capture failed: {result.error}")
        
        print("4. Testing summary...")
        summary = await detector.get_display_summary()
        print(f"   ✅ Summary generated with {summary.get('total_displays', 0)} displays")
        
        print("5. Testing performance stats...")
        stats = detector.get_performance_stats()
        print(f"   ✅ Stats generated with {stats['capture_stats']['total_captures']} total captures")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Ironcliw Multi-Monitor Support CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --detect                    # Detect all displays
  %(prog)s --detect --detailed         # Detailed display information
  %(prog)s --capture                   # Capture screenshots
  %(prog)s --summary                   # Get display summary
  %(prog)s --performance               # Get performance stats
  %(prog)s --test                      # Run comprehensive tests
  %(prog)s --detect --json             # Output JSON format
        """
    )
    
    # Commands
    parser.add_argument("--detect", action="store_true", help="Detect all displays")
    parser.add_argument("--capture", action="store_true", help="Capture screenshots from all displays")
    parser.add_argument("--summary", action="store_true", help="Get display summary")
    parser.add_argument("--performance", action="store_true", help="Get performance statistics")
    parser.add_argument("--test", action="store_true", help="Run comprehensive tests")
    
    # Options
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress banner and extra output")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_banner()
    
    # Check if macOS is available
    if not MACOS_AVAILABLE:
        print("❌ macOS frameworks not available")
        print("   Multi-monitor support requires macOS with Core Graphics")
        return 1
    
    # Determine which command to run
    commands = [args.detect, args.capture, args.summary, args.performance, args.test]
    if sum(commands) == 0:
        parser.print_help()
        return 1
    
    if sum(commands) > 1:
        print("❌ Please specify only one command")
        return 1
    
    # Run the appropriate command
    try:
        if args.detect:
            return asyncio.run(cmd_detect(args))
        elif args.capture:
            return asyncio.run(cmd_capture(args))
        elif args.summary:
            return asyncio.run(cmd_summary(args))
        elif args.performance:
            return asyncio.run(cmd_performance(args))
        elif args.test:
            return asyncio.run(cmd_test(args))
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
