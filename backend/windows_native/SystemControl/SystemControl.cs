using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JarvisWindowsNative.SystemControl
{
    /// <summary>
    /// Provides system control functionality for Windows including window management,
    /// volume control, and notifications.
    /// </summary>
    public class SystemController
    {
        #region Windows API Imports

        [DllImport("user32.dll")]
        private static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);

        [DllImport("user32.dll")]
        private static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

        [DllImport("user32.dll")]
        private static extern int GetWindowTextLength(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        [DllImport("user32.dll")]
        private static extern bool CloseWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

        [DllImport("winmm.dll")]
        private static extern int waveOutGetVolume(IntPtr hwo, out uint dwVolume);

        [DllImport("winmm.dll")]
        private static extern int waveOutSetVolume(IntPtr hwo, uint dwVolume);

        private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        private const int SW_HIDE = 0;
        private const int SW_SHOW = 5;
        private const int SW_MINIMIZE = 6;
        private const int SW_MAXIMIZE = 3;
        private const int SW_RESTORE = 9;

        #endregion

        #region Window Management

        /// <summary>
        /// Get all visible windows with their handles and titles.
        /// </summary>
        public List<WindowInfo> GetAllWindows()
        {
            var windows = new List<WindowInfo>();
            
            EnumWindows((hWnd, lParam) =>
            {
                if (IsWindowVisible(hWnd))
                {
                    int length = GetWindowTextLength(hWnd);
                    if (length > 0)
                    {
                        StringBuilder builder = new StringBuilder(length + 1);
                        GetWindowText(hWnd, builder, builder.Capacity);
                        
                        GetWindowThreadProcessId(hWnd, out uint processId);
                        string processName = "";
                        
                        try
                        {
                            Process process = Process.GetProcessById((int)processId);
                            processName = process.ProcessName;
                        }
                        catch
                        {
                            processName = "Unknown";
                        }
                        
                        windows.Add(new WindowInfo
                        {
                            Handle = hWnd,
                            Title = builder.ToString(),
                            ProcessId = processId,
                            ProcessName = processName
                        });
                    }
                }
                return true;
            }, IntPtr.Zero);
            
            return windows;
        }

        /// <summary>
        /// Get the currently focused window.
        /// </summary>
        public WindowInfo? GetFocusedWindow()
        {
            IntPtr hWnd = GetForegroundWindow();
            if (hWnd == IntPtr.Zero)
                return null;

            int length = GetWindowTextLength(hWnd);
            if (length == 0)
                return null;

            StringBuilder builder = new StringBuilder(length + 1);
            GetWindowText(hWnd, builder, builder.Capacity);
            
            GetWindowThreadProcessId(hWnd, out uint processId);
            string processName = "";
            
            try
            {
                Process process = Process.GetProcessById((int)processId);
                processName = process.ProcessName;
            }
            catch
            {
                processName = "Unknown";
            }

            return new WindowInfo
            {
                Handle = hWnd,
                Title = builder.ToString(),
                ProcessId = processId,
                ProcessName = processName
            };
        }

        /// <summary>
        /// Focus a window by its handle.
        /// </summary>
        public bool FocusWindow(IntPtr handle)
        {
            return SetForegroundWindow(handle);
        }

        /// <summary>
        /// Minimize a window by its handle.
        /// </summary>
        public bool MinimizeWindow(IntPtr handle)
        {
            return ShowWindow(handle, SW_MINIMIZE);
        }

        /// <summary>
        /// Maximize a window by its handle.
        /// </summary>
        public bool MaximizeWindow(IntPtr handle)
        {
            return ShowWindow(handle, SW_MAXIMIZE);
        }

        /// <summary>
        /// Restore a window by its handle.
        /// </summary>
        public bool RestoreWindow(IntPtr handle)
        {
            return ShowWindow(handle, SW_RESTORE);
        }

        /// <summary>
        /// Hide a window by its handle.
        /// </summary>
        public bool HideWindow(IntPtr handle)
        {
            return ShowWindow(handle, SW_HIDE);
        }

        /// <summary>
        /// Show a window by its handle.
        /// </summary>
        public bool ShowWindowHandle(IntPtr handle)
        {
            return ShowWindow(handle, SW_SHOW);
        }

        /// <summary>
        /// Close a window by its handle.
        /// </summary>
        public bool CloseWindowHandle(IntPtr handle)
        {
            return CloseWindow(handle);
        }

        #endregion

        #region Volume Control

        /// <summary>
        /// Get the system volume (0-100).
        /// </summary>
        public int GetVolume()
        {
            uint currentVolume;
            waveOutGetVolume(IntPtr.Zero, out currentVolume);
            
            // Convert from 0-0xFFFF to 0-100
            ushort calcVolume = (ushort)(currentVolume & 0x0000ffff);
            return (int)(calcVolume / (float)0xFFFF * 100);
        }

        /// <summary>
        /// Set the system volume (0-100).
        /// </summary>
        public bool SetVolume(int volume)
        {
            if (volume < 0 || volume > 100)
                return false;

            // Convert from 0-100 to 0-0xFFFF
            uint newVolume = (uint)((volume / 100.0) * 0xFFFF);
            uint bothChannels = (newVolume << 16) | newVolume;
            
            return waveOutSetVolume(IntPtr.Zero, bothChannels) == 0;
        }

        /// <summary>
        /// Increase volume by specified amount.
        /// </summary>
        public bool IncreaseVolume(int amount = 10)
        {
            int currentVolume = GetVolume();
            int newVolume = Math.Min(100, currentVolume + amount);
            return SetVolume(newVolume);
        }

        /// <summary>
        /// Decrease volume by specified amount.
        /// </summary>
        public bool DecreaseVolume(int amount = 10)
        {
            int currentVolume = GetVolume();
            int newVolume = Math.Max(0, currentVolume - amount);
            return SetVolume(newVolume);
        }

        #endregion

        #region Notifications

        /// <summary>
        /// Show a Windows toast notification.
        /// </summary>
        public async Task<bool> ShowNotification(string title, string message, int durationMs = 5000)
        {
            try
            {
                // Use PowerShell to show notification
                string script = $@"
                    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                    [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

                    $APP_ID = 'JarvisAI'

                    $template = @""<toast>
                        <visual>
                            <binding template='ToastText02'>
                                <text id='1'>{title}</text>
                                <text id='2'>{message}</text>
                            </binding>
                        </visual>
                    </toast>""@

                    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
                    $xml.LoadXml($template)
                    $toast = New-Object Windows.UI.Notifications.ToastNotification $xml
                    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($APP_ID).Show($toast)
                ";

                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "powershell.exe",
                        Arguments = $"-NoProfile -ExecutionPolicy Bypass -Command \"{script}\"",
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true
                    }
                };

                process.Start();
                await process.WaitForExitAsync();
                
                return process.ExitCode == 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to show notification: {ex.Message}");
                return false;
            }
        }

        #endregion
    }

    /// <summary>
    /// Represents information about a window.
    /// </summary>
    public class WindowInfo
    {
        public IntPtr Handle { get; set; }
        public string Title { get; set; } = "";
        public uint ProcessId { get; set; }
        public string ProcessName { get; set; } = "";
    }
}
