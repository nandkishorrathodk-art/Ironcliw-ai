/**
 * Unified WebSocket Service v3.0
 * ==============================
 * Uses DynamicWebSocketClient for all WebSocket operations
 * Provides a simplified API for components that need WebSocket connectivity
 */

import React from 'react';
import DynamicWebSocketClient, { ConnectionState as WSConnectionState } from './DynamicWebSocketClient';
import configService from './DynamicConfigService';

class UnifiedWebSocketService {
  constructor() {
    this.client = new DynamicWebSocketClient({
      autoDiscover: false,
      reconnectStrategy: 'exponential',
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      connectionTimeout: 10000
    });

    this.subscriptions = new Map();
    this.connectionState = 'disconnected';
    this.isInitialized = false;

    // Maintenance mode state for tracking system updates/restarts
    this.maintenanceMode = false;
    this.maintenanceReason = null; // 'updating' | 'restarting' | 'rollback' | null

    // Wait for config and then connect
    this._initializeWhenReady();
  }

  async _initializeWhenReady() {
    try {
      // Wait for config service to discover backend
      const config = await configService.waitForConfig(30000);

      if (config?.WS_BASE_URL) {
        // Configure endpoints
        this.client.endpoints = [
          {
            path: `${config.WS_BASE_URL}/ws`,
            capabilities: ['general', 'voice', 'command'],
            priority: 10
          },
          {
            // Backend mounts vision WS at /vision/ws/vision
            path: `${config.WS_BASE_URL}/vision/ws/vision`,
            capabilities: ['vision', 'monitoring'],
            priority: 8
          }
        ];

        this.isInitialized = true;
        this._setupClientHandlers();
      }
    } catch (error) {
      console.error('[UnifiedWebSocket] Initialization failed:', error.message);
    }
  }

  _setupClientHandlers() {
    this.client.on('connected', ({ endpoint }) => {
      this.connectionState = 'connected';
      // Clear maintenance mode on successful reconnection
      if (this.maintenanceMode) {
        this.maintenanceMode = false;
        this.maintenanceReason = null;
        console.log('✅ Reconnected after maintenance');
        this._notifySubscribers('maintenance_mode', {
          active: false,
          reason: null,
          message: 'System is back online'
        });
      }
      console.log('✅ UnifiedWebSocket connected:', endpoint);
      this._notifySubscribers('connection', { state: 'connected', endpoint });
    });

    this.client.on('disconnected', ({ endpoint }) => {
      this.connectionState = 'disconnected';
      console.log('🔌 UnifiedWebSocket disconnected:', endpoint);
      // Only show "disconnected" if not in maintenance mode
      if (!this.maintenanceMode) {
        this._notifySubscribers('connection', { state: 'disconnected', endpoint });
      }
    });

    // ═══════════════════════════════════════════════════════════════════
    // MAINTENANCE MODE - System Update/Restart/Rollback Events
    // ═══════════════════════════════════════════════════════════════════

    // Handle system updating event from supervisor
    this.client.on('system_updating', (data) => {
      console.log('🔄 System entering maintenance mode: updating');
      this.maintenanceMode = true;
      this.maintenanceReason = 'updating';
      this._notifySubscribers('maintenance_mode', {
        active: true,
        reason: 'updating',
        message: data?.message || 'Downloading updates...',
        estimatedTime: data?.estimated_time || 30,
      });
    });

    // Handle system restarting event
    this.client.on('system_restarting', (data) => {
      console.log('🔄 System entering maintenance mode: restarting');
      this.maintenanceMode = true;
      this.maintenanceReason = 'restarting';
      this._notifySubscribers('maintenance_mode', {
        active: true,
        reason: 'restarting',
        message: data?.message || 'Restarting JARVIS core...',
        estimatedTime: data?.estimated_time || 15,
      });
    });

    // Handle system rollback event
    this.client.on('system_rollback', (data) => {
      console.log('🔄 System entering maintenance mode: rollback');
      this.maintenanceMode = true;
      this.maintenanceReason = 'rollback';
      this._notifySubscribers('maintenance_mode', {
        active: true,
        reason: 'rollback',
        message: data?.message || 'Rolling back to previous version...',
        estimatedTime: data?.estimated_time || 20,
      });
    });

    // Handle system back online
    this.client.on('system_online', (data) => {
      console.log('✅ System back online');
      this.maintenanceMode = false;
      this.maintenanceReason = null;
      this._notifySubscribers('maintenance_mode', {
        active: false,
        reason: null,
        message: data?.message || 'JARVIS is back online',
      });
    });
  }

  /**
   * Connect to a specific capability endpoint
   */
  async connect(capability = 'general') {
    if (!this.isInitialized) {
      await this._initializeWhenReady();
    }

    try {
      await this.client.connect(capability);
      return true;
    } catch (error) {
      console.error('[UnifiedWebSocket] Connect failed:', error.message);
      return false;
    }
  }

  /**
   * Subscribe to a message type
   */
  subscribe(messageType, handler) {
    this.client.on(messageType, handler);

    if (!this.subscriptions.has(messageType)) {
      this.subscriptions.set(messageType, new Set());
    }
    this.subscriptions.get(messageType).add(handler);

    return () => this.unsubscribe(messageType, handler);
  }

  /**
   * Unsubscribe from a message type
   */
  unsubscribe(messageType, handler) {
    this.client.off(messageType, handler);

    const handlers = this.subscriptions.get(messageType);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.subscriptions.delete(messageType);
      }
    }
  }

  /**
   * Send a message
   */
  async send(message, capability = null) {
    return this.client.send(message, capability);
  }

  /**
   * Send a reliable message (waits for ACK)
   */
  async sendReliable(message, capability = null, timeout = 5000) {
    return this.client.sendReliable(message, capability, timeout);
  }

  /**
   * Request workspace analysis
   */
  async requestWorkspaceAnalysis() {
    return this.send({
      type: 'request_workspace_analysis',
      timestamp: new Date().toISOString()
    }, 'vision');
  }

  /**
   * Set monitoring interval
   */
  async setMonitoringInterval(interval) {
    return this.send({
      type: 'set_monitoring_interval',
      interval
    }, 'vision');
  }

  /**
   * Execute an action
   */
  async executeAction(action) {
    return this.send({
      type: 'execute_action',
      action
    }, 'vision');
  }

  /**
   * Send vision command
   */
  async sendVisionCommand(command) {
    return this.send({
      type: 'vision_command',
      command
    }, 'vision');
  }

  /**
   * Send Claude vision query
   */
  async sendClaudeVision(query) {
    return this.send({
      type: 'claude_vision',
      query
    }, 'vision');
  }

  /**
   * Enable/disable autonomous mode
   */
  async setAutonomousMode(enabled) {
    return this.send({
      type: 'set_autonomous_mode',
      enabled
    }, 'vision');
  }

  /**
   * Get connection statistics
   */
  getStats() {
    return this.client.getStats();
  }

  /**
   * Check if connected
   */
  isConnected() {
    return this.connectionState === 'connected';
  }

  /**
   * Disconnect from all endpoints
   */
  disconnect() {
    this.client.destroy();
    this.connectionState = 'disconnected';
  }

  /**
   * Check if system is in maintenance mode (updating/restarting/rollback)
   */
  isInMaintenanceMode() {
    return this.maintenanceMode;
  }

  /**
   * Get maintenance mode details
   */
  getMaintenanceStatus() {
    return {
      active: this.maintenanceMode,
      reason: this.maintenanceReason,
    };
  }

  /**
   * Notify all subscribers of an event
   */
  _notifySubscribers(eventType, data) {
    const handlers = this.subscriptions.get(eventType);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('[UnifiedWebSocket] Subscriber error:', error);
        }
      });
    }
  }
}

// Singleton instance
let serviceInstance = null;

export function getUnifiedWebSocketService() {
  if (!serviceInstance) {
    serviceInstance = new UnifiedWebSocketService();
  }
  return serviceInstance;
}

// React hook for using the service
export function useUnifiedWebSocket() {
  const [connected, setConnected] = React.useState(false);
  const [stats, setStats] = React.useState(null);
  const [maintenanceMode, setMaintenanceMode] = React.useState(false);
  const [maintenanceReason, setMaintenanceReason] = React.useState(null);
  const [maintenanceMessage, setMaintenanceMessage] = React.useState(null);
  const service = React.useMemo(() => getUnifiedWebSocketService(), []);

  React.useEffect(() => {
    // Subscribe to connection changes
    const unsubscribeConnection = service.subscribe('connection', (data) => {
      setConnected(data.state === 'connected');
    });

    // Subscribe to maintenance mode changes
    const unsubscribeMaintenance = service.subscribe('maintenance_mode', (data) => {
      setMaintenanceMode(data.active);
      setMaintenanceReason(data.reason);
      setMaintenanceMessage(data.message);
    });

    // Initial connection state
    setConnected(service.isConnected());
    setMaintenanceMode(service.isInMaintenanceMode());

    // Update stats periodically
    const interval = setInterval(() => {
      setStats(service.getStats());
    }, 5000);

    return () => {
      unsubscribeConnection();
      unsubscribeMaintenance();
      clearInterval(interval);
    };
  }, [service]);

  return {
    service,
    connected,
    stats,
    maintenanceMode,
    maintenanceReason,
    maintenanceMessage,
    connect: (capability) => service.connect(capability),
    disconnect: () => service.disconnect(),
    send: (message, capability) => service.send(message, capability),
    sendReliable: (message, capability, timeout) => service.sendReliable(message, capability, timeout),
    subscribe: (messageType, handler) => service.subscribe(messageType, handler)
  };
}

export default UnifiedWebSocketService;
