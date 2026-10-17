/**
 * SignVision – Sign Language Recognition System
 * Frontend Application Controller
 * Handles SSE / Status polling for continuous subtitle and FPS updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    const subtitleContainerEl = document.getElementById('subtitleContainer');
    const subtitleBadgeRowEl = document.getElementById('subtitleBadgeRow');
    const badgeStatusEl = document.getElementById('badgeStatus');
    const badgeGlossesEl = document.getElementById('badgeGlosses');
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

    // 3. UI Update Helper with Auto-Disappear & Smooth Transitions
    function updateUI(data) {
        if (!subtitleEl || !subtitleContainerEl) return;

        const hasText = data.subtitle && data.subtitle.trim().length > 0;

        if (data.fade_out) {
            // Smoothly fade out the subtitle container
            subtitleContainerEl.classList.add('fade-out');
        } else if (!hasText) {
            // Reset to ready state after full disappearance
            subtitleContainerEl.classList.remove('fade-out', 'is-final');
            if (subtitleBadgeRowEl) subtitleBadgeRowEl.style.display = 'none';
            subtitleEl.textContent = 'Ready. Start signing...';
            subtitleEl.classList.add('empty');
        } else {
            // Text is actively being signed or has been completed
            subtitleContainerEl.classList.remove('fade-out');
            subtitleEl.classList.remove('empty');
            subtitleEl.textContent = data.subtitle;

            // Render glosses pills if available
            if (subtitleBadgeRowEl && badgeStatusEl && badgeGlossesEl) {
                subtitleBadgeRowEl.style.display = 'flex';

                if (data.is_final) {
                    subtitleContainerEl.classList.add('is-final');
                    badgeStatusEl.className = 'badge-status';
                    badgeStatusEl.textContent = '✓ Sentence Generated';
                } else {
                    subtitleContainerEl.classList.remove('is-final');
                    badgeStatusEl.className = 'badge-status active';
                    badgeStatusEl.textContent = 'Translating...';
                }

                if (Array.isArray(data.glosses) && data.glosses.length > 0) {
                    badgeGlossesEl.innerHTML = data.glosses
                        .map(g => `<span class="gloss-pill">${escapeHtml(g)}</span>`)
                        .join('');
                } else {
                    badgeGlossesEl.innerHTML = '';
                }
            }
        }

        if (data.fps !== undefined && fpsCounterEl) {
            fpsCounterEl.textContent = `FPS: ${data.fps.toFixed(1)}`;
        }

        if (data.status && statusLabelEl) {
            statusLabelEl.textContent = data.status;
        }
    }

    function escapeHtml(str) {
        return String(str).replace(/[&<>"']/g, s => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[s]));
    }

    // Start stream listening
    initSSE();
});
