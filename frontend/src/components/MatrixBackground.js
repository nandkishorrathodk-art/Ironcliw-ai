/**
 * MatrixBackground.js - Advanced Matrix Rain Animation
 * =====================================================
 *
 * A highly optimized, dynamic Matrix-style background animation with:
 * - Multi-layer parallax depth effect
 * - Smooth fade in/out pulses
 * - Dynamic character sets (including Japanese katakana)
 * - Adaptive performance based on device capabilities
 * - requestAnimationFrame for smooth 60fps rendering
 * - Canvas-based for optimal performance
 *
 * @author JARVIS System
 * @version 2.0.0
 */

import React, { useEffect, useRef, useCallback, useMemo } from 'react';

// Character sets for Matrix rain
const CHAR_SETS = {
  katakana: 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン',
  latin: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
  numbers: '0123456789',
  symbols: '!@#$%^&*()_+-=[]{}|;:,.<>?',
  binary: '01',
  hex: '0123456789ABCDEF',
};

// Dynamic configuration based on performance
const getConfig = () => {
  const isMobile = window.innerWidth < 768;
  const isLowPower = navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4;

  return {
    // Adaptive column density
    columnDensity: isMobile ? 25 : isLowPower ? 20 : 15,
    // Number of parallax layers
    layers: isMobile ? 2 : 3,
    // Animation speeds for each layer (pixels per frame)
    layerSpeeds: [0.3, 0.5, 0.8],
    // Opacity for each layer (creates depth)
    layerOpacities: [0.15, 0.25, 0.4],
    // Font sizes for each layer
    layerFontSizes: [10, 14, 18],
    // Character change probability per frame
    charChangeProb: 0.02,
    // Fade pulse configuration
    fadePulse: {
      enabled: true,
      minOpacity: 0.3,
      maxOpacity: 1.0,
      cycleDuration: 8000, // ms for full pulse cycle
    },
    // Glow effect intensity
    glowIntensity: isMobile ? 0.3 : 0.5,
    // Target FPS
    targetFPS: 60,
  };
};

// Generate combined character set
const generateCharSet = () => {
  return CHAR_SETS.katakana + CHAR_SETS.latin + CHAR_SETS.numbers + CHAR_SETS.symbols;
};

/**
 * MatrixBackground Component
 *
 * @param {Object} props
 * @param {number} props.opacity - Base opacity (0-1)
 * @param {string} props.color - Primary color (hex or rgb)
 * @param {boolean} props.enabled - Enable/disable animation
 * @param {string} props.intensity - 'low', 'medium', 'high'
 */
const MatrixBackground = ({
  opacity = 0.6,
  color = '#00ff41',
  enabled = true,
  intensity = 'medium'
}) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const layersRef = useRef([]);
  const configRef = useRef(getConfig());
  const startTimeRef = useRef(Date.now());
  const lastFrameTimeRef = useRef(0);

  // Memoize character set
  const charSet = useMemo(() => generateCharSet(), []);

  // Get random character from set
  const getRandomChar = useCallback(() => {
    return charSet[Math.floor(Math.random() * charSet.length)];
  }, [charSet]);

  // Parse color to RGB components
  const parseColor = useCallback((colorStr) => {
    if (colorStr.startsWith('#')) {
      const hex = colorStr.slice(1);
      return {
        r: parseInt(hex.slice(0, 2), 16),
        g: parseInt(hex.slice(2, 4), 16),
        b: parseInt(hex.slice(4, 6), 16),
      };
    }
    return { r: 0, g: 255, b: 65 }; // Default green
  }, []);

  // Initialize a single column
  const initColumn = useCallback((x, layerIndex, canvasHeight) => {
    const config = configRef.current;
    const fontSize = config.layerFontSizes[layerIndex] || 14;
    const maxChars = Math.ceil(canvasHeight / fontSize) + 5;

    return {
      x,
      y: Math.random() * canvasHeight * -1, // Start above screen
      speed: config.layerSpeeds[layerIndex] * (0.8 + Math.random() * 0.4),
      chars: Array.from({ length: maxChars }, () => ({
        char: getRandomChar(),
        brightness: Math.random(),
      })),
      fontSize,
      trailLength: 15 + Math.floor(Math.random() * 10),
      layerIndex,
    };
  }, [getRandomChar]);

  // Initialize all layers
  const initLayers = useCallback((canvas) => {
    const config = configRef.current;
    const layers = [];

    for (let layerIndex = 0; layerIndex < config.layers; layerIndex++) {
      const columns = [];
      const fontSize = config.layerFontSizes[layerIndex] || 14;
      const columnWidth = config.columnDensity + (layerIndex * 5);
      const numColumns = Math.ceil(canvas.width / columnWidth);

      for (let i = 0; i < numColumns; i++) {
        const x = i * columnWidth + (Math.random() * 10 - 5);
        columns.push(initColumn(x, layerIndex, canvas.height));
      }

      layers.push({
        columns,
        opacity: config.layerOpacities[layerIndex],
        fontSize,
      });
    }

    return layers;
  }, [initColumn]);

  // Calculate fade pulse multiplier
  const getFadePulse = useCallback(() => {
    const config = configRef.current;
    if (!config.fadePulse.enabled) return 1;

    const elapsed = Date.now() - startTimeRef.current;
    const cycleProgress = (elapsed % config.fadePulse.cycleDuration) / config.fadePulse.cycleDuration;

    // Smooth sine wave pulse
    const pulse = Math.sin(cycleProgress * Math.PI * 2) * 0.5 + 0.5;

    return config.fadePulse.minOpacity +
           (config.fadePulse.maxOpacity - config.fadePulse.minOpacity) * pulse;
  }, []);

  // Main render function
  const render = useCallback((ctx, canvas, timestamp) => {
    const config = configRef.current;
    const frameInterval = 1000 / config.targetFPS;

    // Frame rate limiting
    if (timestamp - lastFrameTimeRef.current < frameInterval) {
      return;
    }
    lastFrameTimeRef.current = timestamp;

    // Clear canvas with fade effect (creates trails)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const pulseMultiplier = getFadePulse();
    const rgb = parseColor(color);

    // Render each layer (back to front)
    layersRef.current.forEach((layer, layerIndex) => {
      const layerOpacity = layer.opacity * opacity * pulseMultiplier;

      layer.columns.forEach((column) => {
        const fontSize = column.fontSize;
        ctx.font = `${fontSize}px "Courier New", monospace`;

        // Update column position
        column.y += column.speed;

        // Reset column when it goes off screen
        if (column.y > canvas.height + fontSize * column.trailLength) {
          column.y = -fontSize * column.trailLength;
          column.speed = config.layerSpeeds[layerIndex] * (0.8 + Math.random() * 0.4);
        }

        // Render characters in column
        column.chars.forEach((charData, charIndex) => {
          const charY = column.y + charIndex * fontSize;

          // Skip if off screen
          if (charY < -fontSize || charY > canvas.height + fontSize) return;

          // Calculate brightness based on position in trail
          const trailPosition = charIndex / column.trailLength;
          let brightness = 1 - trailPosition;

          // Head character is brightest
          if (charIndex === 0) {
            brightness = 1.2;
            // Add glow effect to head
            ctx.shadowBlur = 15 * config.glowIntensity;
            ctx.shadowColor = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${layerOpacity})`;
          } else {
            ctx.shadowBlur = 0;
          }

          // Random character changes
          if (Math.random() < config.charChangeProb) {
            charData.char = getRandomChar();
          }

          // Calculate final opacity
          const charOpacity = Math.max(0, brightness * layerOpacity * charData.brightness);

          // Set color with calculated opacity
          if (charIndex === 0) {
            // Head character is white/bright green
            ctx.fillStyle = `rgba(255, 255, 255, ${charOpacity})`;
          } else {
            ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${charOpacity})`;
          }

          // Draw character
          ctx.fillText(charData.char, column.x, charY);
        });
      });
    });
  }, [color, opacity, getFadePulse, parseColor, getRandomChar]);

  // Animation loop
  const animate = useCallback((timestamp) => {
    const canvas = canvasRef.current;
    if (!canvas || !enabled) return;

    const ctx = canvas.getContext('2d');
    render(ctx, canvas, timestamp);

    animationRef.current = requestAnimationFrame(animate);
  }, [enabled, render]);

  // Handle resize
  const handleResize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size to window size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Update config for new size
    configRef.current = getConfig();

    // Reinitialize layers
    layersRef.current = initLayers(canvas);
  }, [initLayers]);

  // Setup effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Initial setup
    handleResize();

    // Set initial canvas styles
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Start animation if enabled
    if (enabled) {
      startTimeRef.current = Date.now();
      animationRef.current = requestAnimationFrame(animate);
    }

    // Add resize listener
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [enabled, animate, handleResize]);

  // Handle enabled state changes
  useEffect(() => {
    if (enabled && !animationRef.current) {
      startTimeRef.current = Date.now();
      animationRef.current = requestAnimationFrame(animate);
    } else if (!enabled && animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, [enabled, animate]);

  return (
    <canvas
      ref={canvasRef}
      className="matrix-background"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  );
};

export default MatrixBackground;
