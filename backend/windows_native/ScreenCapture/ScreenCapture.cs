using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace JarvisWindowsNative.ScreenCapture
{
    /// <summary>
    /// Provides screen capture functionality for Windows using GDI+ and Windows.Graphics.Capture.
    /// </summary>
    public class ScreenCaptureEngine
    {
        #region Windows API Imports

        [DllImport("user32.dll")]
        private static extern IntPtr GetDesktopWindow();

        [DllImport("user32.dll")]
        private static extern IntPtr GetDC(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);

        [DllImport("gdi32.dll")]
        private static extern IntPtr CreateCompatibleDC(IntPtr hdc);

        [DllImport("gdi32.dll")]
        private static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int nWidth, int nHeight);

        [DllImport("gdi32.dll")]
        private static extern IntPtr SelectObject(IntPtr hdc, IntPtr hgdiobj);

        [DllImport("gdi32.dll")]
        private static extern bool BitBlt(IntPtr hdcDest, int nXDest, int nYDest, int nWidth, int nHeight,
            IntPtr hdcSrc, int nXSrc, int nYSrc, int dwRop);

        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr hObject);

        [DllImport("gdi32.dll")]
        private static extern bool DeleteDC(IntPtr hdc);

        [DllImport("user32.dll")]
        private static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [DllImport("user32.dll")]
        private static extern int GetSystemMetrics(int nIndex);

        [StructLayout(LayoutKind.Sequential)]
        private struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        private const int SRCCOPY = 0x00CC0020;
        private const int SM_CXSCREEN = 0;
        private const int SM_CYSCREEN = 1;

        #endregion

        /// <summary>
        /// Capture the entire primary screen and return as byte array (PNG).
        /// </summary>
        public byte[] CaptureScreen()
        {
            int screenWidth = GetSystemMetrics(SM_CXSCREEN);
            int screenHeight = GetSystemMetrics(SM_CYSCREEN);

            return CaptureRegion(0, 0, screenWidth, screenHeight);
        }

        /// <summary>
        /// Capture a specific region of the screen.
        /// </summary>
        public byte[] CaptureRegion(int x, int y, int width, int height)
        {
            IntPtr desktopHwnd = GetDesktopWindow();
            IntPtr desktopDc = GetDC(desktopHwnd);
            IntPtr memoryDc = CreateCompatibleDC(desktopDc);
            IntPtr bitmap = CreateCompatibleBitmap(desktopDc, width, height);
            IntPtr oldBitmap = SelectObject(memoryDc, bitmap);

            try
            {
                BitBlt(memoryDc, 0, 0, width, height, desktopDc, x, y, SRCCOPY);

                using (Bitmap image = Image.FromHbitmap(bitmap))
                {
                    using (MemoryStream ms = new MemoryStream())
                    {
                        image.Save(ms, ImageFormat.Png);
                        return ms.ToArray();
                    }
                }
            }
            finally
            {
                SelectObject(memoryDc, oldBitmap);
                DeleteObject(bitmap);
                DeleteDC(memoryDc);
                ReleaseDC(desktopHwnd, desktopDc);
            }
        }

        /// <summary>
        /// Capture a specific window by its handle.
        /// </summary>
        public byte[] CaptureWindow(IntPtr windowHandle)
        {
            if (!GetWindowRect(windowHandle, out RECT rect))
            {
                throw new ArgumentException("Failed to get window rectangle");
            }

            int width = rect.Right - rect.Left;
            int height = rect.Bottom - rect.Top;

            if (width <= 0 || height <= 0)
            {
                throw new ArgumentException("Invalid window dimensions");
            }

            return CaptureRegion(rect.Left, rect.Top, width, height);
        }

        /// <summary>
        /// Save screen capture to file.
        /// </summary>
        public bool SaveScreenToFile(string filePath)
        {
            try
            {
                byte[] imageData = CaptureScreen();
                File.WriteAllBytes(filePath, imageData);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save screen capture: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Save window capture to file.
        /// </summary>
        public bool SaveWindowToFile(IntPtr windowHandle, string filePath)
        {
            try
            {
                byte[] imageData = CaptureWindow(windowHandle);
                File.WriteAllBytes(filePath, imageData);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save window capture: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Get screen dimensions.
        /// </summary>
        public (int width, int height) GetScreenSize()
        {
            return (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
        }

        /// <summary>
        /// Capture screen at specified intervals (for video/monitoring).
        /// </summary>
        public async Task CaptureScreenContinuous(
            Action<byte[]> frameCallback,
            int intervalMs = 100,
            System.Threading.CancellationToken cancellationToken = default)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    byte[] frame = CaptureScreen();
                    frameCallback(frame);
                    await Task.Delay(intervalMs, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in continuous capture: {ex.Message}");
                }
            }
        }
    }

    /// <summary>
    /// Monitor information for multi-monitor setups.
    /// </summary>
    public class MonitorInfo
    {
        public int Index { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public bool IsPrimary { get; set; }
    }

    /// <summary>
    /// Multi-monitor screen capture support.
    /// </summary>
    public class MultiMonitorCapture
    {
        [DllImport("user32.dll")]
        private static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr lprcClip,
            MonitorEnumDelegate lpfnEnum, IntPtr dwData);

        [DllImport("user32.dll")]
        private static extern bool GetMonitorInfo(IntPtr hMonitor, ref MONITORINFO lpmi);

        private delegate bool MonitorEnumDelegate(IntPtr hMonitor, IntPtr hdcMonitor,
            ref RECT lprcMonitor, IntPtr dwData);

        [StructLayout(LayoutKind.Sequential)]
        private struct MONITORINFO
        {
            public int Size;
            public RECT Monitor;
            public RECT WorkArea;
            public uint Flags;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        private const uint MONITORINFOF_PRIMARY = 1;

        /// <summary>
        /// Get all monitor information.
        /// </summary>
        public List<MonitorInfo> GetAllMonitors()
        {
            List<MonitorInfo> monitors = new List<MonitorInfo>();
            int index = 0;

            EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero,
                (IntPtr hMonitor, IntPtr hdcMonitor, ref RECT lprcMonitor, IntPtr dwData) =>
                {
                    MONITORINFO mi = new MONITORINFO();
                    mi.Size = Marshal.SizeOf(mi);

                    if (GetMonitorInfo(hMonitor, ref mi))
                    {
                        monitors.Add(new MonitorInfo
                        {
                            Index = index++,
                            X = mi.Monitor.Left,
                            Y = mi.Monitor.Top,
                            Width = mi.Monitor.Right - mi.Monitor.Left,
                            Height = mi.Monitor.Bottom - mi.Monitor.Top,
                            IsPrimary = (mi.Flags & MONITORINFOF_PRIMARY) != 0
                        });
                    }
                    return true;
                }, IntPtr.Zero);

            return monitors;
        }

        /// <summary>
        /// Capture a specific monitor.
        /// </summary>
        public byte[] CaptureMonitor(int monitorIndex)
        {
            List<MonitorInfo> monitors = GetAllMonitors();
            
            if (monitorIndex < 0 || monitorIndex >= monitors.Count)
            {
                throw new ArgumentException($"Invalid monitor index: {monitorIndex}");
            }

            MonitorInfo monitor = monitors[monitorIndex];
            ScreenCaptureEngine engine = new ScreenCaptureEngine();
            
            return engine.CaptureRegion(monitor.X, monitor.Y, monitor.Width, monitor.Height);
        }
    }
}
