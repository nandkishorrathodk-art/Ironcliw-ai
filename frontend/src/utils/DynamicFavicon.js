/**
 * Dynamic Favicon Manager for Ironcliw
 * 
 * Changes the browser favicon based on Ironcliw state:
 * - idle: Green arc reactor (default)
 * - processing: Blue spinning rings (thinking)
 * - listening: Orange pulsing with microphone (voice input)
 * - error: Red (optional future state)
 */

const FAVICON_STATES = {
  idle: '/favicon.svg',
  processing: '/favicon-processing.svg',
  listening: '/favicon-listening.svg',
};

let currentState = 'idle';
let linkElement = null;

/**
 * Initialize the dynamic favicon system
 * Call this once when the app starts
 */
export function initDynamicFavicon() {
  // Find or create the favicon link element
  linkElement = document.querySelector('link[rel="icon"][type="image/svg+xml"]');
  
  if (!linkElement) {
    linkElement = document.createElement('link');
    linkElement.rel = 'icon';
    linkElement.type = 'image/svg+xml';
    document.head.appendChild(linkElement);
  }
  
  // Set initial favicon
  setFaviconState('idle');
  
  console.log('🎨 Dynamic favicon system initialized');
}

/**
 * Set the favicon to a specific state
 * @param {'idle' | 'processing' | 'listening'} state - The state to set
 */
export function setFaviconState(state) {
  if (!FAVICON_STATES[state]) {
    console.warn(`Unknown favicon state: ${state}`);
    return;
  }
  
  if (state === currentState) {
    return; // No change needed
  }
  
  currentState = state;
  
  if (linkElement) {
    linkElement.href = FAVICON_STATES[state];
  } else {
    // Fallback: find and update the element directly
    const favicon = document.querySelector('link[rel="icon"]');
    if (favicon) {
      favicon.href = FAVICON_STATES[state];
    }
  }
  
  // Also update document title with state indicator (optional)
  const baseTitle = 'Ironcliw-AI Interface';
  switch (state) {
    case 'processing':
      document.title = '⚙️ ' + baseTitle;
      break;
    case 'listening':
      document.title = '🎤 ' + baseTitle;
      break;
    default:
      document.title = baseTitle;
  }
}

/**
 * Get the current favicon state
 * @returns {string} The current state
 */
export function getFaviconState() {
  return currentState;
}

/**
 * Set favicon to idle (green arc reactor)
 */
export function setFaviconIdle() {
  setFaviconState('idle');
}

/**
 * Set favicon to processing (blue spinning)
 */
export function setFaviconProcessing() {
  setFaviconState('processing');
}

/**
 * Set favicon to listening (orange pulsing)
 */
export function setFaviconListening() {
  setFaviconState('listening');
}

/**
 * React hook for managing favicon state
 * @param {boolean} isProcessing - Whether Ironcliw is processing
 * @param {boolean} isListening - Whether Ironcliw is listening
 */
export function useDynamicFavicon(isProcessing, isListening) {
  // Initialize on first render
  if (!linkElement) {
    initDynamicFavicon();
  }
  
  // Update based on state (processing takes precedence over listening)
  if (isProcessing) {
    setFaviconState('processing');
  } else if (isListening) {
    setFaviconState('listening');
  } else {
    setFaviconState('idle');
  }
}

const DynamicFavicon = {
  init: initDynamicFavicon,
  setState: setFaviconState,
  getState: getFaviconState,
  setIdle: setFaviconIdle,
  setProcessing: setFaviconProcessing,
  setListening: setFaviconListening,
  STATES: FAVICON_STATES,
};

export default DynamicFavicon;
