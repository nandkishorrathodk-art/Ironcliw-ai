/**
 * UpdateNotificationBadge Component
 * ==================================
 * 
 * Displays an "Update Available" notification badge/modal when the
 * supervisor detects a new version is available on the remote repository.
 * 
 * Features:
 * - Animated badge that appears when updates are available
 * - Priority-based styling (normal, security, breaking changes)
 * - Rich information display (commits behind, summary, highlights)
 * - "Update Now" button to trigger immediate update
 * - "Later" button to dismiss temporarily
 * - Voice command hint
 * 
 * Usage:
 *   <UpdateNotificationBadge />
 * 
 * Place alongside MaintenanceOverlay in your app root.
 */

import React, { useState, useCallback } from 'react';
import { useUnifiedWebSocket } from '../services/UnifiedWebSocketService';
import './UpdateNotificationBadge.css';

const UpdateNotificationBadge = () => {
    const { 
        updateAvailable, 
        updateInfo, 
        dismissUpdate,
        sendReliable 
    } = useUnifiedWebSocket();
    
    const [showModal, setShowModal] = useState(false);
    const [updating, setUpdating] = useState(false);

    // Handle "Update Now" button click
    const handleUpdateNow = useCallback(async () => {
        setUpdating(true);
        try {
            // Send update command via WebSocket
            await sendReliable({
                type: 'command',
                command: 'update_system',
                source: 'ui_button',
            }, 'general', 5000);
            
            setShowModal(false);
            // MaintenanceOverlay will take over from here
        } catch (error) {
            console.error('Failed to trigger update:', error);
            setUpdating(false);
        }
    }, [sendReliable]);

    // Handle "Later" button click
    const handleLater = useCallback(() => {
        dismissUpdate();
        setShowModal(false);
    }, [dismissUpdate]);

    // Toggle modal visibility
    const toggleModal = useCallback(() => {
        setShowModal(prev => !prev);
    }, []);

    // Don't render if no update available
    if (!updateAvailable || !updateInfo) {
        return null;
    }

    // Determine badge style based on priority
    const getBadgeClass = () => {
        if (updateInfo.security_update) return 'badge-security';
        if (updateInfo.breaking_changes) return 'badge-breaking';
        if (updateInfo.priority === 'high') return 'badge-high';
        return 'badge-normal';
    };

    // Get badge icon
    const getBadgeIcon = () => {
        if (updateInfo.security_update) return '🔒';
        if (updateInfo.breaking_changes) return '⚠️';
        return '📦';
    };

    return (
        <>
            {/* Floating Badge */}
            <button 
                className={`update-notification-badge ${getBadgeClass()}`}
                onClick={toggleModal}
                title="Click for update details"
            >
                <span className="badge-icon">{getBadgeIcon()}</span>
                <span className="badge-text">Update Available</span>
                {updateInfo.commits_behind > 0 && (
                    <span className="badge-count">{updateInfo.commits_behind}</span>
                )}
                <span className="badge-pulse" />
            </button>

            {/* Modal Overlay */}
            {showModal && (
                <div className="update-modal-overlay" onClick={handleLater}>
                    <div 
                        className={`update-modal ${getBadgeClass()}`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="update-modal-header">
                            <span className="modal-icon">{getBadgeIcon()}</span>
                            <h2>System Update Available</h2>
                            <button 
                                className="modal-close"
                                onClick={handleLater}
                                title="Close"
                            >
                                ×
                            </button>
                        </div>

                        {/* Content */}
                        <div className="update-modal-content">
                            {/* Summary */}
                            <p className="update-summary">{updateInfo.summary}</p>

                            {/* Commits behind */}
                            <div className="update-stats">
                                <span className="stat-label">Commits behind:</span>
                                <span className="stat-value">{updateInfo.commits_behind}</span>
                            </div>

                            {/* Priority indicator */}
                            {updateInfo.security_update && (
                                <div className="update-alert security-alert">
                                    🔒 This update includes security fixes
                                </div>
                            )}
                            {updateInfo.breaking_changes && (
                                <div className="update-alert breaking-alert">
                                    ⚠️ This update includes breaking changes
                                </div>
                            )}

                            {/* Highlights */}
                            {updateInfo.highlights && updateInfo.highlights.length > 0 && (
                                <div className="update-highlights">
                                    <h3>What's new:</h3>
                                    <ul>
                                        {updateInfo.highlights.map((highlight, index) => (
                                            <li key={index}>{highlight}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Voice command hint */}
                            <p className="voice-hint">
                                💬 You can also say: "JARVIS, update to the latest version"
                            </p>
                        </div>

                        {/* Actions */}
                        <div className="update-modal-actions">
                            <button 
                                className="btn-later"
                                onClick={handleLater}
                                disabled={updating}
                            >
                                Later
                            </button>
                            <button 
                                className="btn-update"
                                onClick={handleUpdateNow}
                                disabled={updating}
                            >
                                {updating ? 'Starting Update...' : 'Update Now'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default UpdateNotificationBadge;

