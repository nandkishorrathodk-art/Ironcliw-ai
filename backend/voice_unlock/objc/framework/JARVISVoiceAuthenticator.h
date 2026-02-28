/**
 * IroncliwVoiceAuthenticator.h
 * Ironcliw Voice Unlock System
 *
 * Core voice authentication engine that performs voiceprint matching
 * and anti-spoofing detection.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

// Authentication result keys
extern NSString *const IroncliwAuthResultSuccessKey;
extern NSString *const IroncliwAuthResultConfidenceKey;
extern NSString *const IroncliwAuthResultUserIDKey;
extern NSString *const IroncliwAuthResultReasonKey;
extern NSString *const IroncliwAuthResultLivenessScoreKey;
extern NSString *const IroncliwAuthResultAntispoofingScoreKey;

// Authentication failure reasons
typedef NS_ENUM(NSInteger, IroncliwAuthFailureReason) {
    IroncliwAuthFailureReasonUnknown = 0,
    IroncliwAuthFailureReasonNoMatch,
    IroncliwAuthFailureReasonLowConfidence,
    IroncliwAuthFailureReasonLivenessFailed,
    IroncliwAuthFailureReasonSpoofingDetected,
    IroncliwAuthFailureReasonNoiseLevel,
    IroncliwAuthFailureReasonAudioQuality,
    IroncliwAuthFailureReasonTimeout
};

// Voiceprint model
@interface IroncliwVoiceprint : NSObject <NSSecureCoding>

@property (nonatomic, readonly) NSString *userID;
@property (nonatomic, readonly) NSString *userName;
@property (nonatomic, readonly) NSDate *createdDate;
@property (nonatomic, readonly) NSDate *lastUpdated;
@property (nonatomic, readonly) NSArray<NSNumber *> *features;
@property (nonatomic, readonly) NSUInteger sampleCount;
@property (nonatomic, readonly) float qualityScore;

- (instancetype)initWithUserID:(NSString *)userID
                      userName:(NSString *)userName
                      features:(NSArray<NSNumber *> *)features;

- (BOOL)updateWithNewFeatures:(NSArray<NSNumber *> *)features;

@end

// Main authenticator interface
@interface IroncliwVoiceAuthenticator : NSObject

// Configuration
@property (nonatomic, assign) float confidenceThreshold;
@property (nonatomic, assign) float livenessThreshold;
@property (nonatomic, assign) float antispoofingThreshold;
@property (nonatomic, assign) BOOL enableAdaptiveLearning;
@property (nonatomic, assign) BOOL enableLivenessDetection;
@property (nonatomic, assign) BOOL enableAntispoofing;

// Voiceprint management
- (BOOL)loadVoiceprintForUser:(NSString *)userID error:(NSError **)error;
- (BOOL)hasVoiceprintForUser:(NSString *)userID;
- (NSArray<NSString *> *)enrolledUsers;
- (BOOL)deleteVoiceprintForUser:(NSString *)userID;

// Authentication
- (NSDictionary *)authenticateVoice:(NSData *)audioData forUser:(nullable NSString *)userID;
- (NSDictionary *)authenticateAudioBuffer:(AVAudioPCMBuffer *)buffer forUser:(nullable NSString *)userID;

// Feature extraction (for enrollment)
- (NSArray<NSNumber *> *)extractFeaturesFromAudio:(NSData *)audioData;
- (NSArray<NSNumber *> *)extractFeaturesFromBuffer:(AVAudioPCMBuffer *)buffer;

// Audio quality checks
- (NSDictionary *)analyzeAudioQuality:(NSData *)audioData;
- (BOOL)isAudioSuitableForAuthentication:(NSData *)audioData;

// Anti-spoofing
- (float)calculateLivenessScore:(NSData *)audioData;
- (float)calculateAntispoofingScore:(NSData *)audioData;

// Adaptive learning
- (void)updateVoiceprintWithSuccessfulAuth:(NSData *)audioData forUser:(NSString *)userID;

@end

NS_ASSUME_NONNULL_END