/**
 * test_main.m
 * Ironcliw Voice Unlock System
 *
 * Test harness for Voice Unlock components
 */

#import <Foundation/Foundation.h>
#import "IroncliwVoiceUnlockDaemon.h"
#import "IroncliwVoiceAuthenticator.h"
#import "IroncliwScreenUnlockManager.h"
#import "IroncliwVoiceMonitor.h"
#import "IroncliwWebSocketBridge.h"
#import "IroncliwPythonBridge.h"
#import "IroncliwPermissionManager.h"

// Test delegate implementations
@interface TestDelegate : NSObject <IroncliwWebSocketBridgeDelegate, IroncliwVoiceMonitorDelegate, IroncliwPermissionManagerDelegate, IroncliwScreenUnlockDelegate>
@property (nonatomic, strong) NSMutableArray *logs;
@end

@implementation TestDelegate

- (instancetype)init {
    self = [super init];
    if (self) {
        _logs = [NSMutableArray array];
    }
    return self;
}

// WebSocket delegate
- (void)webSocketDidConnect {
    [self.logs addObject:@"WebSocket connected"];
    NSLog(@"✅ WebSocket connected");
}

- (void)webSocketDidDisconnect:(NSError *)error {
    [self.logs addObject:[NSString stringWithFormat:@"WebSocket disconnected: %@", error]];
    NSLog(@"❌ WebSocket disconnected: %@", error);
}

// Voice monitor delegate
- (void)voiceMonitorDidStartListening {
    [self.logs addObject:@"Voice monitor started"];
    NSLog(@"🎤 Voice monitor started listening");
}

- (void)voiceMonitorDidDetectVoice:(IroncliwAudioBufferInfo *)bufferInfo {
    [self.logs addObject:[NSString stringWithFormat:@"Voice detected: %.2f confidence", bufferInfo.voiceConfidence]];
    NSLog(@"🗣️ Voice detected with confidence: %.2f", bufferInfo.voiceConfidence);
}

// Permission manager delegate
- (void)allRequiredPermissionsGranted {
    [self.logs addObject:@"All permissions granted"];
    NSLog(@"✅ All required permissions granted");
}

- (void)missingRequiredPermissions:(NSArray<IroncliwPermissionInfo *> *)permissions {
    NSMutableArray *names = [NSMutableArray array];
    for (IroncliwPermissionInfo *perm in permissions) {
        [names addObject:perm.displayName];
    }
    [self.logs addObject:[NSString stringWithFormat:@"Missing permissions: %@", [names componentsJoinedByString:@", "]]];
    NSLog(@"⚠️ Missing permissions: %@", [names componentsJoinedByString:@", "]);
}

// Screen unlock delegate
- (void)screenStateDidChange:(IroncliwScreenState)newState {
    NSString *stateString = @"Unknown";
    switch (newState) {
        case IroncliwScreenStateUnlocked:
            stateString = @"Unlocked";
            break;
        case IroncliwScreenStateLocked:
            stateString = @"Locked";
            break;
        case IroncliwScreenStateScreensaver:
            stateString = @"Screensaver";
            break;
        case IroncliwScreenStateSleeping:
            stateString = @"Sleeping";
            break;
        default:
            break;
    }
    [self.logs addObject:[NSString stringWithFormat:@"Screen state: %@", stateString]];
    NSLog(@"🖥️ Screen state changed to: %@", stateString);
}

@end

// Test functions
void testPermissions() {
    NSLog(@"\n=== Testing Permissions ===");
    
    IroncliwPermissionManager *permManager = [[IroncliwPermissionManager alloc] init];
    TestDelegate *delegate = [[TestDelegate alloc] init];
    permManager.delegate = delegate;
    
    // Check all permissions
    NSLog(@"Checking permissions...");
    for (IroncliwPermissionInfo *perm in permManager.allPermissions) {
        IroncliwPermissionStatus status = [permManager statusForPermission:perm.type];
        NSString *statusString = @"Unknown";
        switch (status) {
            case IroncliwPermissionStatusAuthorized:
                statusString = @"✅ Authorized";
                break;
            case IroncliwPermissionStatusDenied:
                statusString = @"❌ Denied";
                break;
            case IroncliwPermissionStatusNotDetermined:
                statusString = @"❓ Not Determined";
                break;
            case IroncliwPermissionStatusRestricted:
                statusString = @"🚫 Restricted";
                break;
        }
        NSLog(@"  %@: %@%@", perm.displayName, statusString, perm.isRequired ? @" (Required)" : @"");
    }
    
    NSLog(@"Has all required permissions: %@", permManager.hasAllRequiredPermissions ? @"YES" : @"NO");
}

void testVoiceAuthenticator() {
    NSLog(@"\n=== Testing Voice Authenticator ===");
    
    IroncliwVoiceAuthenticator *auth = [[IroncliwVoiceAuthenticator alloc] init];
    
    // Check enrolled users
    NSArray *users = [auth enrolledUsers];
    NSLog(@"Enrolled users: %@", users.count > 0 ? [users componentsJoinedByString:@", "] : @"None");
    
    // Test audio quality check
    NSData *testAudio = [NSData dataWithBytes:"test" length:4];
    NSDictionary *quality = [auth analyzeAudioQuality:testAudio];
    NSLog(@"Audio quality analysis: %@", quality);
    
    // Test feature extraction
    NSData *longerAudio = [NSMutableData dataWithLength:4096];
    NSArray *features = [auth extractFeaturesFromAudio:longerAudio];
    NSLog(@"Extracted %lu features", (unsigned long)features.count);
}

void testScreenUnlock() {
    NSLog(@"\n=== Testing Screen Unlock Manager ===");
    
    IroncliwScreenUnlockManager *unlockManager = [[IroncliwScreenUnlockManager alloc] init];
    TestDelegate *delegate = [[TestDelegate alloc] init];
    unlockManager.delegate = delegate;
    
    // Check screen state
    BOOL locked = [unlockManager isScreenLocked];
    BOOL screensaver = [unlockManager isScreensaverActive];
    IroncliwScreenState state = [unlockManager detectScreenState];
    
    NSLog(@"Screen locked: %@", locked ? @"YES" : @"NO");
    NSLog(@"Screensaver active: %@", screensaver ? @"YES" : @"NO");
    NSLog(@"Screen state: %ld", (long)state);
    NSLog(@"Can unlock screen: %@", unlockManager.canUnlockScreen ? @"YES" : @"NO");
    NSLog(@"Has secure token: %@", unlockManager.hasSecureToken ? @"YES" : @"NO");
}

void testWebSocketBridge() {
    NSLog(@"\n=== Testing WebSocket Bridge ===");
    
    IroncliwWebSocketBridge *wsbridge = [[IroncliwWebSocketBridge alloc] init];
    TestDelegate *delegate = [[TestDelegate alloc] init];
    wsbridge.delegate = delegate;
    
    NSLog(@"WebSocket configuration:");
    NSLog(@"  Host: %@", wsbridge.serverHost);
    NSLog(@"  Port: %lu", (unsigned long)wsbridge.serverPort);
    NSLog(@"  Auto-reconnect: %@", wsbridge.enableAutoReconnect ? @"YES" : @"NO");
    
    // Test connection (won't actually connect without server)
    [wsbridge startWithPort:8765];
    
    // Wait a bit
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:1.0]];
    
    NSDictionary *connInfo = [wsbridge connectionInfo];
    NSLog(@"Connection info: %@", connInfo);
    
    [wsbridge stop];
}

void testPythonBridge() {
    NSLog(@"\n=== Testing Python Bridge ===");
    
    IroncliwPythonBridge *pybridge = [[IroncliwPythonBridge alloc] init];
    
    NSLog(@"Python configuration:");
    NSLog(@"  Python path: %@", pybridge.pythonPath);
    NSLog(@"  Scripts directory: %@", pybridge.scriptsDirectory);
    
    // Test starting bridge
    NSError *error = nil;
    BOOL started = [pybridge startBridgeWithError:&error];
    
    if (started) {
        NSLog(@"✅ Python bridge started successfully");
        
        // Get Python version
        NSString *version = [pybridge getPythonVersion];
        NSLog(@"Python version: %@", version);
        
        [pybridge stopBridge];
    } else {
        NSLog(@"❌ Failed to start Python bridge: %@", error);
    }
}

void testVoiceMonitor() {
    NSLog(@"\n=== Testing Voice Monitor ===");
    
    IroncliwVoiceMonitor *monitor = [[IroncliwVoiceMonitor alloc] init];
    TestDelegate *delegate = [[TestDelegate alloc] init];
    monitor.delegate = delegate;
    
    NSLog(@"Voice monitor configuration:");
    NSLog(@"  Processing mode: %ld", (long)monitor.processingMode);
    NSLog(@"  Silence timeout: %.1f seconds", monitor.silenceTimeout);
    NSLog(@"  Max recording duration: %.1f seconds", monitor.maxRecordingDuration);
    NSLog(@"  Voice detection threshold: %.3f", monitor.voiceDetectionThreshold);
    
    // Test starting monitor
    if ([monitor startMonitoring]) {
        NSLog(@"✅ Voice monitor started");
        
        // Monitor for 2 seconds
        NSLog(@"Monitoring for 2 seconds...");
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2.0]];
        
        [monitor stopMonitoring];
        NSLog(@"Voice monitor stopped");
    } else {
        NSLog(@"❌ Failed to start voice monitor");
    }
}

void testDaemonStatus() {
    NSLog(@"\n=== Testing Daemon Status ===");
    
    IroncliwVoiceUnlockDaemon *daemon = [IroncliwVoiceUnlockDaemon sharedDaemon];
    
    NSDictionary *status = [daemon getStatus];
    NSLog(@"Daemon status: %@", status);
    
    NSLog(@"Is monitoring: %@", daemon.isMonitoring ? @"YES" : @"NO");
    NSLog(@"Is screen locked: %@", daemon.isScreenLocked ? @"YES" : @"NO");
    NSLog(@"Enrolled user: %@", daemon.enrolledUserIdentifier ?: @"None");
    NSLog(@"Failed attempts: %lu", (unsigned long)daemon.failedAttemptCount);
    
    // Test configuration
    NSDictionary *config = [daemon currentConfiguration];
    NSLog(@"Current configuration: %@", config);
    
    NSArray *phrases = daemon.unlockPhrases;
    NSLog(@"Unlock phrases: %@", [phrases componentsJoinedByString:@", "]);
}

void runIntegrationTest() {
    NSLog(@"\n=== Running Integration Test ===");
    
    IroncliwVoiceUnlockDaemon *daemon = [IroncliwVoiceUnlockDaemon sharedDaemon];
    daemon.options |= IroncliwVoiceUnlockOptionEnableDebugLogging;
    
    // Check if user is enrolled
    if (![daemon isUserEnrolled]) {
        NSLog(@"⚠️ No user enrolled for voice unlock");
        NSLog(@"Please run enrollment first");
        return;
    }
    
    NSError *error = nil;
    if ([daemon startMonitoringWithError:&error]) {
        NSLog(@"✅ Daemon started successfully");
        
        // Simulate screen lock in debug mode
        if (daemon.options & IroncliwVoiceUnlockOptionEnableDebugLogging) {
            NSLog(@"Simulating screen lock...");
            [daemon simulateScreenLock];
            
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:1.0]];
            
            NSLog(@"Simulating voice unlock...");
            [daemon simulateVoiceUnlock:@"Hello Ironcliw, unlock my Mac"];
            
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:1.0]];
        }
        
        [daemon stopMonitoring];
        NSLog(@"Daemon stopped");
    } else {
        NSLog(@"❌ Failed to start daemon: %@", error);
    }
}

// Main test function
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Ironcliw Voice Unlock Test Suite");
        NSLog(@"==============================");
        NSLog(@"Running on macOS %@", [[NSProcessInfo processInfo] operatingSystemVersionString]);
        
        // Parse arguments
        BOOL runAll = YES;
        NSString *specificTest = nil;
        
        if (argc > 1) {
            specificTest = [NSString stringWithUTF8String:argv[1]];
            runAll = [specificTest isEqualToString:@"all"];
        }
        
        // Run tests
        if (runAll || [specificTest isEqualToString:@"permissions"]) {
            testPermissions();
        }
        
        if (runAll || [specificTest isEqualToString:@"authenticator"]) {
            testVoiceAuthenticator();
        }
        
        if (runAll || [specificTest isEqualToString:@"screen"]) {
            testScreenUnlock();
        }
        
        if (runAll || [specificTest isEqualToString:@"websocket"]) {
            testWebSocketBridge();
        }
        
        if (runAll || [specificTest isEqualToString:@"python"]) {
            testPythonBridge();
        }
        
        if (runAll || [specificTest isEqualToString:@"monitor"]) {
            testVoiceMonitor();
        }
        
        if (runAll || [specificTest isEqualToString:@"daemon"]) {
            testDaemonStatus();
        }
        
        if ([specificTest isEqualToString:@"integration"]) {
            runIntegrationTest();
        }
        
        NSLog(@"\nTest suite completed");
        NSLog(@"\nAvailable test targets:");
        NSLog(@"  ./IroncliwVoiceUnlockTest all         - Run all tests");
        NSLog(@"  ./IroncliwVoiceUnlockTest permissions - Test permissions");
        NSLog(@"  ./IroncliwVoiceUnlockTest authenticator- Test voice authenticator");
        NSLog(@"  ./IroncliwVoiceUnlockTest screen      - Test screen unlock");
        NSLog(@"  ./IroncliwVoiceUnlockTest websocket   - Test WebSocket bridge");
        NSLog(@"  ./IroncliwVoiceUnlockTest python      - Test Python bridge");
        NSLog(@"  ./IroncliwVoiceUnlockTest monitor     - Test voice monitor");
        NSLog(@"  ./IroncliwVoiceUnlockTest daemon      - Test daemon status");
        NSLog(@"  ./IroncliwVoiceUnlockTest integration - Run integration test");
    }
    
    return 0;
}