/**
 * IroncliwWebSocketServer.h
 * Ironcliw Voice Unlock System
 *
 * WebSocket server for daemon communication with Ironcliw API
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class IroncliwWebSocketServer;
@class PSWebSocketServer;
@class PSWebSocket;

@protocol IroncliwWebSocketServerDelegate <NSObject>
@optional
- (void)webSocketServer:(IroncliwWebSocketServer *)server didStart:(NSUInteger)port;
- (void)webSocketServer:(IroncliwWebSocketServer *)server didReceiveMessage:(NSDictionary *)message fromClient:(PSWebSocket *)client;
- (void)webSocketServer:(IroncliwWebSocketServer *)server didConnectClient:(PSWebSocket *)client;
- (void)webSocketServer:(IroncliwWebSocketServer *)server didDisconnectClient:(PSWebSocket *)client;
- (void)webSocketServer:(IroncliwWebSocketServer *)server didFailWithError:(NSError *)error;
@end

@interface IroncliwWebSocketServer : NSObject

@property (nonatomic, weak) id<IroncliwWebSocketServerDelegate> delegate;
@property (nonatomic, readonly) NSUInteger port;
@property (nonatomic, readonly) BOOL isRunning;
@property (nonatomic, readonly) NSArray<PSWebSocket *> *connectedClients;

- (instancetype)initWithPort:(NSUInteger)port;
- (BOOL)startWithError:(NSError **)error;
- (void)stop;

// Send messages to clients
- (void)sendMessage:(NSDictionary *)message toClient:(PSWebSocket *)client;
- (void)broadcastMessage:(NSDictionary *)message;

// Handle daemon commands
- (void)handleCommand:(NSString *)command parameters:(NSDictionary *)parameters completion:(void (^)(NSDictionary *response))completion;

@end

NS_ASSUME_NONNULL_END