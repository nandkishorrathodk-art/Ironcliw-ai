/**
 * MaintenanceOverlay Component
 * =============================
 * 
 * Displays a premium overlay when JARVIS is in maintenance mode
 * (updating, restarting, or rolling back).
 * 
 * Features:
 * - Matrix rain effect (canvas-based)
 * - Falling particles animation
 * - Arc reactor spinner
 * - Matches JARVIS green theme (#00ff41)
 * 
 * Shows status message instead of a "Connection Error" banner.
 * Automatically hides when the system comes back online.
 * 
 * Usage:
 *   <MaintenanceOverlay />
 * 
 * Place at the root of your app (e.g., in App.js).
 */

import React, { useEffect, useRef } from 'react';
import { useUnifiedWebSocket } from '../services/UnifiedWebSocketService';
import './MaintenanceOverlay.css';

// Matrix rain characters
const MATRIX_CHARS = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

const MaintenanceOverlay = () => {
    const { maintenanceMode, maintenanceReason, maintenanceMessage } = useUnifiedWebSocket();
    const canvasRef = useRef(null);
    const animationRef = useRef(null);

    // Matrix rain effect
    useEffect(() => {
        if (!maintenanceMode || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas size
        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Matrix columns
        const fontSize = 14;
        const columns = Math.floor(canvas.width / fontSize);
        const drops = Array(columns).fill(1);

        // Animation loop
        const draw = () => {
            // Semi-transparent black to create trail effect
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Green text
            ctx.fillStyle = '#00ff41';
            ctx.font = `${fontSize}px monospace`;

            for (let i = 0; i < drops.length; i++) {
                // Random character
                const char = MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)];

                // Draw character
                ctx.fillText(char, i * fontSize, drops[i] * fontSize);

                // Reset drop to top randomly after reaching bottom
                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }

                drops[i]++;
            }

            animationRef.current = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [maintenanceMode]);

    if (!maintenanceMode) {
        return null;
    }

    // Determine icon based on reason
    const getIcon = () => {
        switch (maintenanceReason) {
            case 'updating':
                return '⬇️';
            case 'restarting':
                return '🔄';
            case 'rollback':
                return '⏪';
            default:
                return '⚙️';
        }
    };

    // Determine title based on reason
    const getTitle = () => {
        switch (maintenanceReason) {
            case 'updating':
                return 'Updating JARVIS';
            case 'restarting':
                return 'Restarting';
            case 'rollback':
                return 'Rolling Back';
            default:
                return 'Maintenance';
        }
    };

    // Generate particles for CSS animation
    const renderParticles = () => {
        const particles = [];
        for (let i = 1; i <= 20; i++) {
            particles.push(<div key={i} className="particle" />);
        }
        return particles;
    };

    return (
        <div className="maintenance-overlay">
            {/* Matrix rain canvas */}
            <canvas ref={canvasRef} className="matrix-rain-canvas" />

            {/* Dark overlay for readability */}
            <div className="matrix-overlay-dark" />

            {/* Falling particles */}
            <div className="maintenance-particles">
                {renderParticles()}
            </div>

            {/* Main content */}
            <div className="maintenance-content">
                {/* Arc Reactor Style Spinner */}
                <div className="arc-reactor-spinner">
                    <div className="reactor-core"></div>
                    <div className="reactor-ring reactor-ring-1"></div>
                    <div className="reactor-ring reactor-ring-2"></div>
                    <div className="reactor-ring reactor-ring-3"></div>
                </div>

                {/* Status Icon */}
                <div className="maintenance-icon">{getIcon()}</div>

                {/* Title */}
                <h2 className="maintenance-title">{getTitle()}</h2>

                {/* Message */}
                <p className="maintenance-message">
                    {maintenanceMessage || 'Please wait...'}
                </p>

                {/* Subtitle */}
                <p className="maintenance-subtitle">
                    I'll be back shortly, sir.
                </p>

                {/* Animated progress bar */}
                <div className="maintenance-progress">
                    <div className="maintenance-progress-bar" />
                </div>
            </div>

            {/* Status strip at bottom */}
            <div className="maintenance-status-strip" />
        </div>
    );
};

export default MaintenanceOverlay;
