/**
 * IroncliwVoiceUnlockDaemon.h
 * Ironcliw Voice Unlock System
 *
 * Main daemon that runs in the background to monitor for voice unlock phrases
 * when the screen is locked. Integrates with macOS Security Framework.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <Security/Security.h>
#import <LocalAuthentication/LocalAuthentication.h>

NS_ASSUME_NONNULL_BEGIN

// Forward declarations
@class IroncliwVoiceAuthenticator;
@class IroncliwScreenUnlockManager;
@class IroncliwVoiceMonitor;
@class IroncliwPythonBridge;
@class IroncliwPermissionManager;
@class IroncliwWebSocketBridge;

// Notification constants
extern NSString *const IroncliwVoiceUnlockStatusChangedNotification;
extern NSString *const IroncliwVoiceUnlockAuthenticationFailedNotification;
extern NSString *const IroncliwVoiceUnlockAuthenticationSucceededNotification;

// Error domain
extern NSString *const IroncliwVoiceUnlockErrorDomain;

// Voice unlock state
typedef NS_ENUM(NSInteger, IroncliwVoiceUnlockState) {
    IroncliwVoiceUnlockStateInactive = 0,
    IroncliwVoiceUnlockStateMonitoring,
    IroncliwVoiceUnlockStateProcessing,
    IroncliwVoiceUnlockStateUnlocking,
    IroncliwVoiceUnlockStateError
};

// Configuration options
typedef NS_OPTIONS(NSUInteger, IroncliwVoiceUnlockOptions) {
    IroncliwVoiceUnlockOptionNone = 0,
    IroncliwVoiceUnlockOptionEnableLivenessDetection = 1 << 0,
    IroncliwVoiceUnlockOptionEnableAntiSpoofing = 1 << 1,
    IroncliwVoiceUnlockOptionEnableAdaptiveThresholds = 1 << 2,
    IroncliwVoiceUnlockOptionEnableContinuousAuthentication = 1 << 3,
    IroncliwVoiceUnlockOptionEnableDebugLogging = 1 << 4
};

/**
 * Main daemon interface
 */
@interface IroncliwVoiceUnlockDaemon : NSObject

// Singleton instance
+ (instancetype)sharedDaemon;

// Core properties
@property (nonatomic, readonly) IroncliwVoiceUnlockState state;
@property (nonatomic, readonly) BOOL isMonitoring;
@property (nonatomic, readonly) BOOL isScreenLocked;
@property (nonatomic, readonly) NSString *enrolledUserIdentifier;
@property (nonatomic, readonly) NSDate *lastUnlockAttempt;
@property (nonatomic, readonly) NSUInteger failedAttemptCount;

// Configuration
@property (nonatomic, assign) IroncliwVoiceUnlockOptions options;
@property (nonatomic, assign) NSTimeInterval authenticationTimeout;
@property (nonatomic, assign) NSUInteger maxFailedAttempts;
@property (nonatomic, assign) NSTimeInterval lockoutDuration;

// Unlock phrases (dynamically loaded from configuration)
@property (nonatomic, readonly) NSArray<NSString *> *unlockPhrases;

// Core methods
- (BOOL)startMonitoringWithError:(NSError **)error;
- (void)stopMonitoring;
- (BOOL)isUserEnrolled;
- (NSDictionary *)getStatus;
- (void)resetFailedAttempts;

// Configuration methods
- (void)loadConfigurationFromFile:(NSString *)path;
- (void)updateConfiguration:(NSDictionary *)config;
- (NSDictionary *)currentConfiguration;

// Test methods (for development)
- (void)simulateVoiceUnlock:(NSString *)phrase;
- (void)simulateScreenLock;
- (void)simulateScreenUnlock;

@end

NS_ASSUME_NONNULL_END