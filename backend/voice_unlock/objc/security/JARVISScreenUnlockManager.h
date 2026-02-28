/**
 * IroncliwScreenUnlockManager.h
 * Ironcliw Voice Unlock System
 *
 * Manages screen lock detection and unlock operations using
 * macOS Security Framework and private APIs.
 */

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <LocalAuthentication/LocalAuthentication.h>

NS_ASSUME_NONNULL_BEGIN

// Screen state
typedef NS_ENUM(NSInteger, IroncliwScreenState) {
    IroncliwScreenStateUnknown = 0,
    IroncliwScreenStateUnlocked,
    IroncliwScreenStateLocked,
    IroncliwScreenStateScreensaver,
    IroncliwScreenStateSleeping
};

// Unlock method
typedef NS_ENUM(NSInteger, IroncliwUnlockMethod) {
    IroncliwUnlockMethodPassword = 0,
    IroncliwUnlockMethodBiometric,
    IroncliwUnlockMethodVoice,
    IroncliwUnlockMethodEmergency
};

// Unlock result
@interface IroncliwUnlockResult : NSObject
@property (nonatomic, readonly) BOOL success;
@property (nonatomic, readonly) IroncliwUnlockMethod method;
@property (nonatomic, readonly) NSTimeInterval duration;
@property (nonatomic, readonly, nullable) NSError *error;
@end

// Screen unlock delegate
@protocol IroncliwScreenUnlockDelegate <NSObject>
@optional
- (void)screenStateDidChange:(IroncliwScreenState)newState;
- (void)screenUnlockDidBegin;
- (void)screenUnlockDidComplete:(IroncliwUnlockResult *)result;
- (void)screenUnlockDidFail:(NSError *)error;
@end

// Main screen unlock manager
@interface IroncliwScreenUnlockManager : NSObject

@property (nonatomic, weak, nullable) id<IroncliwScreenUnlockDelegate> delegate;
@property (nonatomic, readonly) IroncliwScreenState currentScreenState;
@property (nonatomic, readonly) BOOL canUnlockScreen;
@property (nonatomic, readonly) BOOL hasSecureToken;

// Screen state detection
- (BOOL)isScreenLocked;
- (BOOL)isScreensaverActive;
- (BOOL)isSystemSleeping;
- (IroncliwScreenState)detectScreenState;

// Lock operations
- (BOOL)lockScreen;
- (BOOL)lockScreenWithError:(NSError **)error;

// Unlock operations
- (BOOL)unlockScreenWithError:(NSError **)error;
- (void)unlockScreenAsync:(void (^)(IroncliwUnlockResult *result))completion;
- (BOOL)unlockScreenWithPassword:(NSString *)password error:(NSError **)error;

// Authentication
- (BOOL)authenticateWithVoice:(NSData *)voiceData error:(NSError **)error;
- (BOOL)verifyUserPassword:(NSString *)password error:(NSError **)error;

// Keychain integration for secure token
- (BOOL)storeSecureTokenForUnlock:(NSString *)password error:(NSError **)error;
- (BOOL)hasStoredSecureToken;
- (void)clearSecureToken;

// System integration
- (BOOL)requestScreenUnlockPermission;
- (BOOL)hasScreenUnlockPermission;
- (void)simulateUserPresence;

// Wake system
- (void)wakeDisplayIfNeeded;
- (void)preventSystemSleep:(BOOL)prevent;

@end

NS_ASSUME_NONNULL_END