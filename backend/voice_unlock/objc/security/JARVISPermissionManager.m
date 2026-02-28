/**
 * IroncliwPermissionManager.m
 * Ironcliw Voice Unlock System
 *
 * Implementation of macOS permission management.
 */

#import "IroncliwPermissionManager.h"
#import <AVFoundation/AVFoundation.h>
#import <Carbon/Carbon.h>
#import <IOKit/IOKitLib.h>
#import <AppKit/AppKit.h>
#import <os/log.h>

// Permission info implementation
@implementation IroncliwPermissionInfo
@end

// Main implementation
@interface IroncliwPermissionManager ()

@property (nonatomic, strong) NSMutableDictionary<NSNumber *, IroncliwPermissionInfo *> *permissionInfoCache;
@property (nonatomic, strong) NSTimer *monitoringTimer;
@property (nonatomic, strong) os_log_t logger;
@property (nonatomic, assign) BOOL isMonitoring;

@end

@implementation IroncliwPermissionManager

- (instancetype)init {
    self = [super init];
    if (self) {
        _permissionInfoCache = [NSMutableDictionary dictionary];
        _logger = os_log_create("com.jarvis.voiceunlock", "permissions");
        
        [self setupPermissionInfo];
        [self checkInitialPermissions];
    }
    return self;
}

- (void)dealloc {
    [self stopMonitoringPermissions];
}

#pragma mark - Setup

- (void)setupPermissionInfo {
    // Microphone
    IroncliwPermissionInfo *mic = [[IroncliwPermissionInfo alloc] init];
    mic.type = IroncliwPermissionTypeMicrophone;
    mic.displayName = @"Microphone";
    mic.explanation = @"Required to capture voice commands for unlocking your Mac";
    mic.isRequired = YES;
    mic.canRequestInApp = YES;
    self.permissionInfoCache[@(IroncliwPermissionTypeMicrophone)] = mic;
    
    // Accessibility
    IroncliwPermissionInfo *accessibility = [[IroncliwPermissionInfo alloc] init];
    accessibility.type = IroncliwPermissionTypeAccessibility;
    accessibility.displayName = @"Accessibility";
    accessibility.explanation = @"Required to simulate keyboard input for unlocking the screen";
    accessibility.isRequired = YES;
    accessibility.canRequestInApp = YES;
    self.permissionInfoCache[@(IroncliwPermissionTypeAccessibility)] = accessibility;
    
    // Screen Recording
    IroncliwPermissionInfo *screenRecording = [[IroncliwPermissionInfo alloc] init];
    screenRecording.type = IroncliwPermissionTypeScreenRecording;
    screenRecording.displayName = @"Screen Recording";
    screenRecording.explanation = @"Optional - Allows detection of screen lock state";
    screenRecording.isRequired = NO;
    screenRecording.canRequestInApp = NO;
    self.permissionInfoCache[@(IroncliwPermissionTypeScreenRecording)] = screenRecording;
    
    // Full Disk Access
    IroncliwPermissionInfo *fullDisk = [[IroncliwPermissionInfo alloc] init];
    fullDisk.type = IroncliwPermissionTypeFullDiskAccess;
    fullDisk.displayName = @"Full Disk Access";
    fullDisk.explanation = @"Optional - Allows access to voice training data";
    fullDisk.isRequired = NO;
    fullDisk.canRequestInApp = NO;
    self.permissionInfoCache[@(IroncliwPermissionTypeFullDiskAccess)] = fullDisk;
    
    // Input Monitoring
    IroncliwPermissionInfo *inputMonitoring = [[IroncliwPermissionInfo alloc] init];
    inputMonitoring.type = IroncliwPermissionTypeInputMonitoring;
    inputMonitoring.displayName = @"Input Monitoring";
    inputMonitoring.explanation = @"Optional - Enhances security by detecting unauthorized access attempts";
    inputMonitoring.isRequired = NO;
    inputMonitoring.canRequestInApp = NO;
    self.permissionInfoCache[@(IroncliwPermissionTypeInputMonitoring)] = inputMonitoring;
    
    // System Events
    IroncliwPermissionInfo *systemEvents = [[IroncliwPermissionInfo alloc] init];
    systemEvents.type = IroncliwPermissionTypeSystemEvents;
    systemEvents.displayName = @"System Events";
    systemEvents.explanation = @"Optional - Enhanced interaction with the lock screen";
    systemEvents.isRequired = NO;  // Make optional for now
    systemEvents.canRequestInApp = NO;
    self.permissionInfoCache[@(IroncliwPermissionTypeSystemEvents)] = systemEvents;
    
    // Keychain
    IroncliwPermissionInfo *keychain = [[IroncliwPermissionInfo alloc] init];
    keychain.type = IroncliwPermissionTypeKeychain;
    keychain.displayName = @"Keychain Access";
    keychain.explanation = @"Required to securely store authentication tokens";
    keychain.isRequired = YES;
    keychain.canRequestInApp = YES;
    self.permissionInfoCache[@(IroncliwPermissionTypeKeychain)] = keychain;
}

- (void)checkInitialPermissions {
    for (IroncliwPermissionInfo *info in self.permissionInfoCache.allValues) {
        info.status = [self checkPermissionStatus:info.type];
    }
}

#pragma mark - Permission Checking

- (IroncliwPermissionStatus)statusForPermission:(IroncliwPermissionType)permission {
    IroncliwPermissionInfo *info = self.permissionInfoCache[@(permission)];
    if (info) {
        info.status = [self checkPermissionStatus:permission];
        return info.status;
    }
    return IroncliwPermissionStatusNotDetermined;
}

- (IroncliwPermissionStatus)checkPermissionStatus:(IroncliwPermissionType)permission {
    switch (permission) {
        case IroncliwPermissionTypeMicrophone:
            return [self checkMicrophonePermission];
            
        case IroncliwPermissionTypeAccessibility:
            return [self checkAccessibilityPermission];
            
        case IroncliwPermissionTypeScreenRecording:
            return [self checkScreenRecordingPermission];
            
        case IroncliwPermissionTypeFullDiskAccess:
            return [self checkFullDiskAccessPermission];
            
        case IroncliwPermissionTypeInputMonitoring:
            return [self checkInputMonitoringPermission];
            
        case IroncliwPermissionTypeSystemEvents:
            return [self checkSystemEventsPermission];
            
        case IroncliwPermissionTypeKeychain:
            return [self checkKeychainPermission];
    }
}

- (IroncliwPermissionStatus)checkMicrophonePermission {
    if (@available(macOS 10.14, *)) {
        AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
        switch (status) {
            case AVAuthorizationStatusAuthorized:
                return IroncliwPermissionStatusAuthorized;
            case AVAuthorizationStatusDenied:
                return IroncliwPermissionStatusDenied;
            case AVAuthorizationStatusRestricted:
                return IroncliwPermissionStatusRestricted;
            case AVAuthorizationStatusNotDetermined:
                return IroncliwPermissionStatusNotDetermined;
        }
    }
    return IroncliwPermissionStatusAuthorized; // Pre-10.14 doesn't require permission
}

- (IroncliwPermissionStatus)checkAccessibilityPermission {
    if (AXIsProcessTrusted()) {
        return IroncliwPermissionStatusAuthorized;
    }
    return IroncliwPermissionStatusDenied;
}

- (IroncliwPermissionStatus)checkScreenRecordingPermission {
    if (@available(macOS 10.15, *)) {
        // Check if we can access screen content
        // This is a simple check - actual screen recording would need ScreenCaptureKit
        CFArrayRef windowList = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
        if (windowList) {
            CFIndex count = CFArrayGetCount(windowList);
            CFRelease(windowList);
            
            // If we can get window info with real content, we have permission
            // Without permission, the count would be very limited
            if (count > 0) {
                return IroncliwPermissionStatusAuthorized;
            }
        }
        return IroncliwPermissionStatusDenied;
    }
    return IroncliwPermissionStatusAuthorized;
}

- (IroncliwPermissionStatus)checkFullDiskAccessPermission {
    // Check if we can access a protected location
    NSString *testPath = [@"~/Library/Safari/Bookmarks.plist" stringByExpandingTildeInPath];
    if ([[NSFileManager defaultManager] isReadableFileAtPath:testPath]) {
        return IroncliwPermissionStatusAuthorized;
    }
    
    // Try to read TCC database
    NSString *tccPath = @"/Library/Application Support/com.apple.TCC/TCC.db";
    if ([[NSFileManager defaultManager] isReadableFileAtPath:tccPath]) {
        return IroncliwPermissionStatusAuthorized;
    }
    
    return IroncliwPermissionStatusDenied;
}

- (IroncliwPermissionStatus)checkInputMonitoringPermission {
    // Check if we can monitor global events
    if (@available(macOS 10.15, *)) {
        // Try to create an event tap
        CFMachPortRef eventTap = CGEventTapCreate(kCGSessionEventTap,
                                                  kCGHeadInsertEventTap,
                                                  kCGEventTapOptionListenOnly,
                                                  kCGEventMaskForAllEvents,
                                                  NULL, NULL);
        if (eventTap) {
            CFRelease(eventTap);
            return IroncliwPermissionStatusAuthorized;
        }
        return IroncliwPermissionStatusDenied;
    }
    return IroncliwPermissionStatusAuthorized;
}

- (IroncliwPermissionStatus)checkSystemEventsPermission {
    // Check if we can send Apple Events to System Events
    NSAppleEventDescriptor *targetDescriptor = [NSAppleEventDescriptor descriptorWithBundleIdentifier:@"com.apple.systemevents"];
    
    NSAppleEventDescriptor *appleEvent = [NSAppleEventDescriptor appleEventWithEventClass:'ascr'
                                                                                  eventID:'gdte'
                                                                         targetDescriptor:targetDescriptor
                                                                                 returnID:kAutoGenerateReturnID
                                                                            transactionID:kAnyTransactionID];
    
    AEDesc reply = {typeNull, NULL};
    OSStatus status = AESendMessage([appleEvent aeDesc], &reply, kAENoReply | kAECanInteract, kAEDefaultTimeout);
    
    if (reply.descriptorType != typeNull) {
        AEDisposeDesc(&reply);
    }
    
    if (status == noErr) {
        return IroncliwPermissionStatusAuthorized;
    } else if (status == -1743) { // errAEEventNotPermitted
        return IroncliwPermissionStatusDenied;
    }
    
    return IroncliwPermissionStatusNotDetermined;
}

- (IroncliwPermissionStatus)checkKeychainPermission {
    // Keychain access is generally available on macOS
    // Try a simple keychain operation to verify access
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: @"com.jarvis.test",
        (__bridge id)kSecReturnData: @NO,
        (__bridge id)kSecMatchLimit: (__bridge id)kSecMatchLimitOne
    };
    
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, NULL);
    
    // errSecItemNotFound is fine - it means we can access keychain but item doesn't exist
    // errSecInteractionNotAllowed or other errors mean we don't have access
    if (status == errSecSuccess || status == errSecItemNotFound) {
        return IroncliwPermissionStatusAuthorized;
    }
    
    return IroncliwPermissionStatusDenied;
}

#pragma mark - Permission Requests

- (BOOL)checkAndRequestPermissions:(NSError **)error {
    NSMutableArray<IroncliwPermissionInfo *> *missingRequired = [NSMutableArray array];
    
    // Check all permissions
    for (IroncliwPermissionInfo *info in self.permissionInfoCache.allValues) {
        info.status = [self checkPermissionStatus:info.type];
        
        if (info.isRequired && info.status != IroncliwPermissionStatusAuthorized) {
            [missingRequired addObject:info];
        }
    }
    
    // If all required permissions are granted
    if (missingRequired.count == 0) {
        if ([self.delegate respondsToSelector:@selector(allRequiredPermissionsGranted)]) {
            [self.delegate allRequiredPermissionsGranted];
        }
        return YES;
    }
    
    // Request permissions that can be requested in-app
    for (IroncliwPermissionInfo *info in missingRequired) {
        if (info.canRequestInApp && info.status == IroncliwPermissionStatusNotDetermined) {
            switch (info.type) {
                case IroncliwPermissionTypeMicrophone:
                    [self requestMicrophonePermission:nil];
                    break;
                    
                case IroncliwPermissionTypeAccessibility:
                    [self requestAccessibilityPermission];
                    break;
                    
                default:
                    break;
            }
        }
    }
    
    // Notify about missing permissions
    if ([self.delegate respondsToSelector:@selector(missingRequiredPermissions:)]) {
        [self.delegate missingRequiredPermissions:missingRequired];
    }
    
    if (error) {
        NSString *missingList = [[missingRequired valueForKey:@"displayName"] componentsJoinedByString:@", "];
        *error = [NSError errorWithDomain:@"IroncliwPermissions"
                                     code:1001
                                 userInfo:@{
            NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Missing required permissions: %@", missingList]
        }];
    }
    
    return NO;
}

- (void)checkAllPermissionsWithCompletion:(void (^)(BOOL, NSArray<IroncliwPermissionInfo *> *))completion {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSMutableArray<IroncliwPermissionInfo *> *missing = [NSMutableArray array];
        BOOL allGranted = YES;
        
        for (IroncliwPermissionInfo *info in self.permissionInfoCache.allValues) {
            info.status = [self checkPermissionStatus:info.type];
            
            if (info.isRequired && info.status != IroncliwPermissionStatusAuthorized) {
                [missing addObject:info];
                allGranted = NO;
            }
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) {
                completion(allGranted, missing);
            }
        });
    });
}

#pragma mark - Individual Permission Requests

- (void)requestMicrophonePermission:(void (^)(BOOL))completion {
    if (@available(macOS 10.14, *)) {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio 
                                 completionHandler:^(BOOL granted) {
            IroncliwPermissionInfo *info = self.permissionInfoCache[@(IroncliwPermissionTypeMicrophone)];
            info.status = granted ? IroncliwPermissionStatusAuthorized : IroncliwPermissionStatusDenied;
            
            dispatch_async(dispatch_get_main_queue(), ^{
                if ([self.delegate respondsToSelector:@selector(permissionStatusChanged:status:)]) {
                    [self.delegate permissionStatusChanged:IroncliwPermissionTypeMicrophone 
                                                    status:info.status];
                }
                
                if (completion) {
                    completion(granted);
                }
            });
        }];
    } else {
        if (completion) {
            completion(YES);
        }
    }
}

- (BOOL)requestAccessibilityPermission {
    NSDictionary *options = @{(__bridge id)kAXTrustedCheckOptionPrompt: @YES};
    BOOL trusted = AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
    
    IroncliwPermissionInfo *info = self.permissionInfoCache[@(IroncliwPermissionTypeAccessibility)];
    info.status = trusted ? IroncliwPermissionStatusAuthorized : IroncliwPermissionStatusDenied;
    
    if ([self.delegate respondsToSelector:@selector(permissionStatusChanged:status:)]) {
        [self.delegate permissionStatusChanged:IroncliwPermissionTypeAccessibility 
                                        status:info.status];
    }
    
    return trusted;
}

- (BOOL)requestScreenRecordingPermission {
    // Screen recording permission cannot be requested programmatically
    // Open system preferences instead
    [self openSystemPreferencesForPermission:IroncliwPermissionTypeScreenRecording];
    return NO;
}

- (BOOL)requestFullDiskAccessPermission {
    // Full disk access cannot be requested programmatically
    [self openSystemPreferencesForPermission:IroncliwPermissionTypeFullDiskAccess];
    return NO;
}

- (BOOL)requestInputMonitoringPermission {
    // Input monitoring permission cannot be requested programmatically
    [self openSystemPreferencesForPermission:IroncliwPermissionTypeInputMonitoring];
    return NO;
}

#pragma mark - System Preferences

- (void)openSystemPreferencesForPermission:(IroncliwPermissionType)permission {
    NSString *prefPane = nil;
    
    switch (permission) {
        case IroncliwPermissionTypeMicrophone:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone";
            break;
            
        case IroncliwPermissionTypeAccessibility:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";
            break;
            
        case IroncliwPermissionTypeScreenRecording:
            if (@available(macOS 10.15, *)) {
                prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture";
            }
            break;
            
        case IroncliwPermissionTypeFullDiskAccess:
            if (@available(macOS 10.14, *)) {
                prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles";
            }
            break;
            
        case IroncliwPermissionTypeInputMonitoring:
            if (@available(macOS 10.15, *)) {
                prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent";
            }
            break;
            
        case IroncliwPermissionTypeSystemEvents:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_Automation";
            break;
            
        default:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy";
            break;
    }
    
    if (prefPane) {
        [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:prefPane]];
    }
}

- (void)openPrivacyPreferences {
    NSString *urlString = @"x-apple.systempreferences:com.apple.preference.security?Privacy";
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:urlString]];
}

- (void)openAccessibilityPreferences {
    NSString *urlString = @"x-apple.systempreferences:com.apple.preference.universalaccess";
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:urlString]];
}

- (void)openSecurityPreferences {
    NSString *urlString = @"x-apple.systempreferences:com.apple.preference.security";
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:urlString]];
}

#pragma mark - Explanations

- (NSString *)explanationForPermission:(IroncliwPermissionType)permission {
    IroncliwPermissionInfo *info = self.permissionInfoCache[@(permission)];
    return info ? info.explanation : @"This permission is required for Ironcliw Voice Unlock to function properly.";
}

- (NSString *)instructionsForPermission:(IroncliwPermissionType)permission {
    switch (permission) {
        case IroncliwPermissionTypeMicrophone:
            return @"Click 'OK' when prompted to grant microphone access.";
            
        case IroncliwPermissionTypeAccessibility:
            return @"1. Click 'Open System Preferences' when prompted\n"
                   @"2. Click the lock icon to make changes\n"
                   @"3. Check the box next to Ironcliw Voice Unlock\n"
                   @"4. Close System Preferences";
            
        case IroncliwPermissionTypeScreenRecording:
            return @"1. Open System Preferences > Security & Privacy > Privacy\n"
                   @"2. Select 'Screen Recording' from the list\n"
                   @"3. Click the lock icon to make changes\n"
                   @"4. Check the box next to Ironcliw Voice Unlock\n"
                   @"5. Restart Ironcliw Voice Unlock";
            
        case IroncliwPermissionTypeFullDiskAccess:
            return @"1. Open System Preferences > Security & Privacy > Privacy\n"
                   @"2. Select 'Full Disk Access' from the list\n"
                   @"3. Click the lock icon to make changes\n"
                   @"4. Click '+' and add Ironcliw Voice Unlock\n"
                   @"5. Restart Ironcliw Voice Unlock";
            
        default:
            return @"Please grant the requested permission in System Preferences.";
    }
}

#pragma mark - Monitoring

- (void)startMonitoringPermissions {
    if (self.isMonitoring) {
        return;
    }
    
    self.isMonitoring = YES;
    
    // Monitor permission changes every 2 seconds
    self.monitoringTimer = [NSTimer scheduledTimerWithTimeInterval:2.0
                                                            target:self
                                                          selector:@selector(checkPermissionChanges)
                                                          userInfo:nil
                                                           repeats:YES];
}

- (void)stopMonitoringPermissions {
    self.isMonitoring = NO;
    [self.monitoringTimer invalidate];
    self.monitoringTimer = nil;
}

- (void)checkPermissionChanges {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        BOOL changesDetected = NO;
        
        for (IroncliwPermissionInfo *info in self.permissionInfoCache.allValues) {
            IroncliwPermissionStatus oldStatus = info.status;
            IroncliwPermissionStatus newStatus = [self checkPermissionStatus:info.type];
            
            if (oldStatus != newStatus) {
                info.status = newStatus;
                changesDetected = YES;
                
                dispatch_async(dispatch_get_main_queue(), ^{
                    if ([self.delegate respondsToSelector:@selector(permissionStatusChanged:status:)]) {
                        [self.delegate permissionStatusChanged:info.type status:newStatus];
                    }
                });
                
                os_log_info(self.logger, "Permission %@ changed from %ld to %ld",
                           info.displayName, (long)oldStatus, (long)newStatus);
            }
        }
        
        if (changesDetected) {
            [self checkAndRequestPermissions:nil];
        }
    });
}

#pragma mark - Properties

- (BOOL)hasAllRequiredPermissions {
    for (IroncliwPermissionInfo *info in self.permissionInfoCache.allValues) {
        if (info.isRequired && info.status != IroncliwPermissionStatusAuthorized) {
            return NO;
        }
    }
    return YES;
}

- (NSArray<IroncliwPermissionInfo *> *)allPermissions {
    return [self.permissionInfoCache.allValues sortedArrayUsingComparator:^NSComparisonResult(IroncliwPermissionInfo *a, IroncliwPermissionInfo *b) {
        if (a.isRequired != b.isRequired) {
            return a.isRequired ? NSOrderedAscending : NSOrderedDescending;
        }
        return [a.displayName compare:b.displayName];
    }];
}

- (NSArray<IroncliwPermissionInfo *> *)missingPermissions {
    NSMutableArray *missing = [NSMutableArray array];
    
    for (IroncliwPermissionInfo *info in self.permissionInfoCache.allValues) {
        if (info.status != IroncliwPermissionStatusAuthorized) {
            [missing addObject:info];
        }
    }
    
    return [missing sortedArrayUsingComparator:^NSComparisonResult(IroncliwPermissionInfo *a, IroncliwPermissionInfo *b) {
        if (a.isRequired != b.isRequired) {
            return a.isRequired ? NSOrderedAscending : NSOrderedDescending;
        }
        return [a.displayName compare:b.displayName];
    }];
}

#pragma mark - Utility

- (BOOL)isRunningInSandbox {
    // Check if app is sandboxed
    NSString *homeDir = NSHomeDirectory();
    return [homeDir containsString:@"/Library/Containers/"];
}

- (BOOL)hasEntitlement:(NSString *)entitlement {
    SecTaskRef task = SecTaskCreateFromSelf(kCFAllocatorDefault);
    if (task == NULL) {
        return NO;
    }
    
    CFTypeRef value = SecTaskCopyValueForEntitlement(task, (__bridge CFStringRef)entitlement, NULL);
    CFRelease(task);
    
    BOOL hasEntitlement = NO;
    if (value != NULL) {
        if (CFGetTypeID(value) == CFBooleanGetTypeID()) {
            hasEntitlement = CFBooleanGetValue(value);
        }
        CFRelease(value);
    }
    
    return hasEntitlement;
}

@end