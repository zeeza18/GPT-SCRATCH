// ZEEPT Web Application - Client-Side JavaScript

const API_BASE = window.location.origin;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    loadModelInfo();
});

// Initialize application
function initializeApp() {
    console.log('ZEEPT Application Initialized');
    updateSliderValues();
}

// Setup event listeners
function setupEventListeners() {
    // Slider updates
    document.getElementById('max-tokens').addEventListener('input', (e) => {
        document.getElementById('max-tokens-value').textContent = e.target.value;
    });

    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('temperature-value').textContent = e.target.value;
    });

    document.getElementById('top-k').addEventListener('input', (e) => {
        document.getElementById('top-k-value').textContent = e.target.value;
    });

    // Buttons
    document.getElementById('generate-btn').addEventListener('click', generateText);
    document.getElementById('example-btn').addEventListener('click', loadExample);
    document.getElementById('evaluate-btn').addEventListener('click', evaluateText);
}

// Update slider value displays
function updateSliderValues() {
    document.getElementById('max-tokens-value').textContent =
        document.getElementById('max-tokens').value;
    document.getElementById('temperature-value').textContent =
        document.getElementById('temperature').value;
    document.getElementById('top-k-value').textContent =
        document.getElementById('top-k').value;
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        if (data.model_loaded && data.model_info) {
            document.getElementById('layers').textContent = data.model_info.layers || '-';
            document.getElementById('heads').textContent = data.model_info.heads || '-';
            document.getElementById('embed-dim').textContent = data.model_info.embedding_dim || '-';
        }

        // Update status badge
        const statusBadge = document.getElementById('status-badge');
        if (data.status === 'online') {
            statusBadge.innerHTML = '<span class="status-dot"></span><span>Online</span>';
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('status-badge').innerHTML =
            '<span class="status-dot" style="background: var(--danger-color);"></span><span>Offline</span>';
    }
}

// Load example prompt
async function loadExample() {
    try {
        const response = await fetch(`${API_BASE}/api/examples`);
        const data = await response.json();

        if (data.examples && data.examples.length > 0) {
            const randomExample = data.examples[Math.floor(Math.random() * data.examples.length)];
            document.getElementById('prompt').value = randomExample;
        }
    } catch (error) {
        console.error('Error loading example:', error);
        document.getElementById('prompt').value = 'Artificial intelligence is';
    }
}

// Generate text
async function generateText() {
    const prompt = document.getElementById('prompt').value.trim();

    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    const maxTokens = parseInt(document.getElementById('max-tokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const topK = parseInt(document.getElementById('top-k').value);

    const button = document.getElementById('generate-btn');
    const btnText = button.querySelector('.btn-text');
    const btnLoader = button.querySelector('.btn-loader');

    // Show loading state
    button.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    try {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temperature,
                top_k: topK
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Display results
            document.getElementById('generated-text').textContent = data.generated_text;
            document.getElementById('gen-tokens').textContent = data.num_tokens;
            document.getElementById('gen-temp').textContent = temperature.toFixed(1);
            document.getElementById('generation-output').style.display = 'block';

            // Smooth scroll to results
            document.getElementById('generation-output').scrollIntoView({
                behavior: 'smooth',
                block: 'nearest'
            });
        } else {
            alert('Error generating text: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate text. Please try again.');
    } finally {
        // Reset button state
        button.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Evaluate text
async function evaluateText() {
    const text = document.getElementById('eval-text').value.trim();

    if (!text) {
        alert('Please enter text to evaluate');
        return;
    }

    const button = document.getElementById('evaluate-btn');
    const btnText = button.querySelector('.btn-text');
    const btnLoader = button.querySelector('.btn-loader');

    // Show loading state
    button.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    try {
        const response = await fetch(`${API_BASE}/api/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Display results
            document.getElementById('perplexity-value').textContent =
                data.perplexity.toFixed(2);
            document.getElementById('eval-tokens').textContent = data.num_tokens;
            document.getElementById('evaluation-output').style.display = 'block';

            // Smooth scroll to results
            document.getElementById('evaluation-output').scrollIntoView({
                behavior: 'smooth',
                block: 'nearest'
            });
        } else {
            alert('Error evaluating text: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to evaluate text. Please try again.');
    } finally {
        // Reset button state
        button.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Utility: Format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}
