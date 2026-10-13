/**
 * SignVision – Sign Language Recognition System
 * Frontend Application Controller
 * Handles SSE / Status polling for continuous subtitle and FPS updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    const subtitleEl = document.getElementById('subtitleText');
    const statusLabelEl = document.getElementById('statusLabel');
    const fpsCounterEl = document.getElementById('fpsCounter');

    let isSSEConnected = false;

    // 1. Initialize Server-Sent Events (SSE) Stream
    function initSSE() {
        const evtSource = new EventSource('/api/stream');

        evtSource.onopen = () => {
            isSSEConnected = true;
            statusLabelEl.textContent = 'Listening...';
        };

        evtSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateUI(data);
            } catch (err) {
                console.error('Failed to parse SSE data:', err);
            }
        };

        evtSource.onerror = () => {
            isSSEConnected = false;
            evtSource.close();
            // Fallback to HTTP polling if SSE disconnects
            setTimeout(pollStatus, 500);
        };
    }

    // 2. HTTP Polling Fallback
    async function pollStatus() {
        if (isSSEConnected) return;

        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                updateUI(data);
            }
        } catch (err) {
            console.warn('Polling error:', err);
            statusLabelEl.textContent = 'Connecting...';
        }

        setTimeout(pollStatus, 300);
    }

    // 3. UI Update Helper
    function updateUI(data) {
        if (data.subtitle && subtitleEl) {
            subtitleEl.textContent = data.subtitle;
        }

        if (data.fps !== undefined && fpsCounterEl) {
            fpsCounterEl.textContent = `FPS: ${data.fps.toFixed(1)}`;
        }

        if (data.status && statusLabelEl) {
            statusLabelEl.textContent = data.status;
        }
    }

    // Start stream listening
    initSSE();
});
