/**
 * IroncliwPermissionManager.h
 * Ironcliw Voice Unlock System
 *
 * Manages macOS permissions required for voice unlock functionality
 * including microphone, accessibility, and security permissions.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

// Permission types
typedef NS_ENUM(NSInteger, IroncliwPermissionType) {
    IroncliwPermissionTypeMicrophone = 0,
    IroncliwPermissionTypeAccessibility,
    IroncliwPermissionTypeScreenRecording,
    IroncliwPermissionTypeFullDiskAccess,
    IroncliwPermissionTypeInputMonitoring,
    IroncliwPermissionTypeSystemEvents,
    IroncliwPermissionTypeKeychain
};

// Permission status
typedef NS_ENUM(NSInteger, IroncliwPermissionStatus) {
    IroncliwPermissionStatusNotDetermined = 0,
    IroncliwPermissionStatusDenied,
    IroncliwPermissionStatusAuthorized,
    IroncliwPermissionStatusRestricted
};

// Permission info
@interface IroncliwPermissionInfo : NSObject
@property (nonatomic, assign) IroncliwPermissionType type;
@property (nonatomic, assign) IroncliwPermissionStatus status;
@property (nonatomic, strong) NSString *displayName;
@property (nonatomic, strong) NSString *explanation;
@property (nonatomic, assign) BOOL isRequired;
@property (nonatomic, assign) BOOL canRequestInApp;
@end

// Permission delegate
@protocol IroncliwPermissionManagerDelegate <NSObject>
@optional
- (void)permissionStatusChanged:(IroncliwPermissionType)permission status:(IroncliwPermissionStatus)status;
- (void)allRequiredPermissionsGranted;
- (void)missingRequiredPermissions:(NSArray<IroncliwPermissionInfo *> *)permissions;
@end

// Main permission manager interface
@interface IroncliwPermissionManager : NSObject

@property (nonatomic, weak, nullable) id<IroncliwPermissionManagerDelegate> delegate;
@property (nonatomic, readonly) BOOL hasAllRequiredPermissions;
@property (nonatomic, readonly) NSArray<IroncliwPermissionInfo *> *allPermissions;
@property (nonatomic, readonly) NSArray<IroncliwPermissionInfo *> *missingPermissions;

// Permission checking
- (IroncliwPermissionStatus)statusForPermission:(IroncliwPermissionType)permission;
- (BOOL)checkAndRequestPermissions:(NSError **)error;
- (void)checkAllPermissionsWithCompletion:(void (^)(BOOL allGranted, NSArray<IroncliwPermissionInfo *> *missing))completion;

// Individual permission requests
- (void)requestMicrophonePermission:(void (^)(BOOL granted))completion;
- (BOOL)requestAccessibilityPermission;
- (BOOL)requestScreenRecordingPermission;
- (BOOL)requestFullDiskAccessPermission;
- (BOOL)requestInputMonitoringPermission;

// System preferences
- (void)openSystemPreferencesForPermission:(IroncliwPermissionType)permission;
- (void)openPrivacyPreferences;
- (void)openAccessibilityPreferences;
- (void)openSecurityPreferences;

// Permission explanations
- (NSString *)explanationForPermission:(IroncliwPermissionType)permission;
- (NSString *)instructionsForPermission:(IroncliwPermissionType)permission;

// Monitoring
- (void)startMonitoringPermissions;
- (void)stopMonitoringPermissions;

// Utility
- (BOOL)isRunningInSandbox;
- (BOOL)hasEntitlement:(NSString *)entitlement;

@end

NS_ASSUME_NONNULL_END