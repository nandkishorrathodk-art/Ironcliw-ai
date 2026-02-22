using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace JarvisWindowsNative.AudioEngine
{
    /// <summary>
    /// Provides WASAPI audio functionality for Windows including recording, playback, and device management.
    /// </summary>
    public class AudioEngine
    {
        private MMDeviceEnumerator? deviceEnumerator;
        private WasapiCapture? currentRecording;
        private WasapiOut? currentPlayback;
        private readonly object recordingLock = new object();
        private readonly object playbackLock = new object();

        public AudioEngine()
        {
            deviceEnumerator = new MMDeviceEnumerator();
        }

        #region Device Management

        /// <summary>
        /// Get all available audio input devices.
        /// </summary>
        public List<AudioDeviceInfo> GetInputDevices()
        {
            if (deviceEnumerator == null)
                return new List<AudioDeviceInfo>();

            var devices = new List<AudioDeviceInfo>();
            var collection = deviceEnumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);

            foreach (var device in collection)
            {
                devices.Add(new AudioDeviceInfo
                {
                    Id = device.ID,
                    Name = device.FriendlyName,
                    IsDefault = device.ID == deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Console).ID
                });
            }

            return devices;
        }

        /// <summary>
        /// Get all available audio output devices.
        /// </summary>
        public List<AudioDeviceInfo> GetOutputDevices()
        {
            if (deviceEnumerator == null)
                return new List<AudioDeviceInfo>();

            var devices = new List<AudioDeviceInfo>();
            var collection = deviceEnumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active);

            foreach (var device in collection)
            {
                devices.Add(new AudioDeviceInfo
                {
                    Id = device.ID,
                    Name = device.FriendlyName,
                    IsDefault = device.ID == deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console).ID
                });
            }

            return devices;
        }

        /// <summary>
        /// Get the default input device.
        /// </summary>
        public AudioDeviceInfo? GetDefaultInputDevice()
        {
            if (deviceEnumerator == null)
                return null;

            try
            {
                var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Console);
                return new AudioDeviceInfo
                {
                    Id = device.ID,
                    Name = device.FriendlyName,
                    IsDefault = true
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Get the default output device.
        /// </summary>
        public AudioDeviceInfo? GetDefaultOutputDevice()
        {
            if (deviceEnumerator == null)
                return null;

            try
            {
                var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
                return new AudioDeviceInfo
                {
                    Id = device.ID,
                    Name = device.FriendlyName,
                    IsDefault = true
                };
            }
            catch
            {
                return null;
            }
        }

        #endregion

        #region Audio Recording

        /// <summary>
        /// Start recording audio from the default input device.
        /// </summary>
        public bool StartRecording(Action<byte[]> dataCallback, int sampleRate = 16000, int bitDepth = 16)
        {
            lock (recordingLock)
            {
                if (currentRecording != null)
                {
                    StopRecording();
                }

                try
                {
                    if (deviceEnumerator == null)
                        return false;

                    var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Console);
                    currentRecording = new WasapiCapture(device);

                    currentRecording.DataAvailable += (sender, e) =>
                    {
                        if (e.BytesRecorded > 0)
                        {
                            byte[] buffer = new byte[e.BytesRecorded];
                            Array.Copy(e.Buffer, buffer, e.BytesRecorded);
                            dataCallback(buffer);
                        }
                    };

                    currentRecording.RecordingStopped += (sender, e) =>
                    {
                        if (e.Exception != null)
                        {
                            Console.WriteLine($"Recording stopped with exception: {e.Exception.Message}");
                        }
                    };

                    currentRecording.StartRecording();
                    return true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to start recording: {ex.Message}");
                    currentRecording?.Dispose();
                    currentRecording = null;
                    return false;
                }
            }
        }

        /// <summary>
        /// Stop recording audio.
        /// </summary>
        public void StopRecording()
        {
            lock (recordingLock)
            {
                if (currentRecording != null)
                {
                    try
                    {
                        currentRecording.StopRecording();
                    }
                    catch { }
                    finally
                    {
                        currentRecording.Dispose();
                        currentRecording = null;
                    }
                }
            }
        }

        /// <summary>
        /// Check if currently recording.
        /// </summary>
        public bool IsRecording
        {
            get
            {
                lock (recordingLock)
                {
                    return currentRecording != null;
                }
            }
        }

        #endregion

        #region Audio Playback

        /// <summary>
        /// Play audio data through the default output device.
        /// </summary>
        public bool PlayAudio(byte[] audioData, int sampleRate = 16000, int channels = 1, int bitDepth = 16)
        {
            lock (playbackLock)
            {
                try
                {
                    if (deviceEnumerator == null)
                        return false;

                    var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
                    
                    // Create a wave format for the audio data
                    var waveFormat = new WaveFormat(sampleRate, bitDepth, channels);
                    
                    // Create a wave provider from the byte array
                    var provider = new RawSourceWaveStream(new System.IO.MemoryStream(audioData), waveFormat);
                    
                    currentPlayback = new WasapiOut(device, AudioClientShareMode.Shared, false, 100);
                    currentPlayback.Init(provider);
                    
                    currentPlayback.PlaybackStopped += (sender, e) =>
                    {
                        lock (playbackLock)
                        {
                            currentPlayback?.Dispose();
                            currentPlayback = null;
                        }
                    };
                    
                    currentPlayback.Play();
                    return true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to play audio: {ex.Message}");
                    currentPlayback?.Dispose();
                    currentPlayback = null;
                    return false;
                }
            }
        }

        /// <summary>
        /// Stop audio playback.
        /// </summary>
        public void StopPlayback()
        {
            lock (playbackLock)
            {
                if (currentPlayback != null)
                {
                    try
                    {
                        currentPlayback.Stop();
                    }
                    catch { }
                    finally
                    {
                        currentPlayback.Dispose();
                        currentPlayback = null;
                    }
                }
            }
        }

        /// <summary>
        /// Check if currently playing.
        /// </summary>
        public bool IsPlaying
        {
            get
            {
                lock (playbackLock)
                {
                    return currentPlayback != null && currentPlayback.PlaybackState == PlaybackState.Playing;
                }
            }
        }

        /// <summary>
        /// Get current playback volume (0.0 - 1.0).
        /// </summary>
        public float GetVolume()
        {
            if (deviceEnumerator == null)
                return 0.0f;

            try
            {
                var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
                return device.AudioEndpointVolume.MasterVolumeLevelScalar;
            }
            catch
            {
                return 0.0f;
            }
        }

        /// <summary>
        /// Set playback volume (0.0 - 1.0).
        /// </summary>
        public bool SetVolume(float volume)
        {
            if (deviceEnumerator == null)
                return false;

            if (volume < 0.0f || volume > 1.0f)
                return false;

            try
            {
                var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
                device.AudioEndpointVolume.MasterVolumeLevelScalar = volume;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to set volume: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Check if output is muted.
        /// </summary>
        public bool IsMuted()
        {
            if (deviceEnumerator == null)
                return false;

            try
            {
                var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
                return device.AudioEndpointVolume.Mute;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Mute or unmute output.
        /// </summary>
        public bool SetMute(bool mute)
        {
            if (deviceEnumerator == null)
                return false;

            try
            {
                var device = deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
                device.AudioEndpointVolume.Mute = mute;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to set mute: {ex.Message}");
                return false;
            }
        }

        #endregion

        #region Cleanup

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            StopRecording();
            StopPlayback();
            deviceEnumerator?.Dispose();
            deviceEnumerator = null;
        }

        #endregion
    }

    /// <summary>
    /// Represents information about an audio device.
    /// </summary>
    public class AudioDeviceInfo
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public bool IsDefault { get; set; }
    }
}
