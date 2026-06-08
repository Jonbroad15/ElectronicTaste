document.addEventListener('DOMContentLoaded', () => {
    
    // --- Security Gate Logic ---
    // In a real production app, client-side passwords are not secure. 
    // This provides a simple gate to keep casual visitors out as requested.
    const CORRECT_PASSWORD = 'raver'; 
    const overlay = document.getElementById('auth-overlay');
    const mainContent = document.getElementById('main-content');
    const passwordInput = document.getElementById('password-input');
    const authSubmit = document.getElementById('auth-submit');
    const authError = document.getElementById('auth-error');

    function attemptLogin() {
        if (passwordInput.value === CORRECT_PASSWORD) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
                mainContent.style.display = 'block';
                // Trigger markdown load after login so it's fresh
                loadMarkdownFiles();
            }, 300);
        } else {
            authError.textContent = 'Incorrect password. Try again.';
            passwordInput.value = '';
            passwordInput.focus();
        }
    }

    authSubmit.addEventListener('click', attemptLogin);
    passwordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') attemptLogin();
    });


    // --- Navigation Logic ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.content-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            navBtns.forEach(b => b.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });


    // --- Markdown Loading Logic ---
    async function loadMarkdownFiles() {
        const files = {
            'mission': 'specs/mission.md',
            'roadmap': 'specs/roadmap.md',
            'tech': 'specs/tech.md'
        };

        for (const [id, path] of Object.entries(files)) {
            try {
                // Fetch the markdown file with a cache-busting query parameter
                const cacheBuster = new Date().getTime();
                const response = await fetch(`${path}?t=${cacheBuster}`, { cache: "no-store" });
                if (!response.ok) throw new Error('Network response was not ok');
                const text = await response.text();
                
                // Parse and inject
                const contentDiv = document.getElementById(`${id}-content`);
                contentDiv.innerHTML = marked.parse(text);
            } catch (error) {
                console.error(`Error loading ${path}:`, error);
                document.getElementById(`${id}-content`).innerHTML = `<p style="color:red;">Error loading document. Ensure you are running a local server.</p>`;
            }
        }
    }

    // --- UI Demo: Uploader & Classifier Simulation ---
    
    // Selectors
    const uploadZone = document.getElementById('upload-zone');
    const audioFileInput = document.getElementById('audio-file-input');
    const uploadError = document.getElementById('upload-error');
    const errorText = document.getElementById('error-text');
    const processingPanel = document.getElementById('processing-panel');
    const resultsPanel = document.getElementById('results-panel');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const processingTitle = document.getElementById('processing-title');
    const processingFileInfo = document.getElementById('processing-file-info');
    const resetButton = document.getElementById('reset-button');
    
    // Results DOM elements
    const resultTrackName = document.getElementById('result-track-name');
    const resultTrackMeta = document.getElementById('result-track-meta');
    const resultSubgenre = document.getElementById('result-subgenre');
    const resultConfidence = document.getElementById('result-confidence');
    const resultConfidenceBar = document.getElementById('result-confidence-bar');
    const resultBpm = document.getElementById('result-bpm');
    const resultKey = document.getElementById('result-key');
    const resultEnergyText = document.getElementById('result-energy-text');
    const resultEnergyDots = document.getElementById('result-energy-dots');
    const probabilityDistributionList = document.getElementById('probability-distribution-list');

    // List of electronic music subgenres with associated tempos, energy levels, and display colors
    const SUBGENRES = [
        { name: 'Progressive House', bpmMin: 124, bpmMax: 128, energy: 'High', vocalProb: 'Vocal (35%)', color: '#ba55d3' },
        { name: 'Melodic Techno', bpmMin: 122, bpmMax: 126, energy: 'Medium-High', vocalProb: 'Instrumental (90%)', color: '#00e676' },
        { name: 'Drum & Bass', bpmMin: 170, bpmMax: 176, energy: 'Intense', vocalProb: 'Instrumental (78%)', color: '#ff9100' },
        { name: 'Deep House', bpmMin: 118, bpmMax: 122, energy: 'Medium', vocalProb: 'Vocal (62%)', color: '#2979ff' },
        { name: 'Psytrance', bpmMin: 138, bpmMax: 145, energy: 'Intense', vocalProb: 'Instrumental (95%)', color: '#d500f9' },
        { name: 'Dubstep', bpmMin: 140, bpmMax: 142, energy: 'Intense', vocalProb: 'Instrumental (82%)', color: '#ffd600' },
        { name: 'Ambient Chill', bpmMin: 80, bpmMax: 100, energy: 'Low', vocalProb: 'Instrumental (98%)', color: '#00e5ff' }
    ];

    const KEYS = [
        '8A (A Minor)', '9A (E Minor)', '10A (B Minor)', '11A (F# Minor)', '12A (C# Minor)',
        '1A (G# Minor)', '2A (D# Minor)', '3A (A# Minor)', '4A (F Minor)', '5A (C Minor)',
        '6A (G Minor)', '7A (D Minor)', '8B (C Major)', '9B (G Major)', '10B (D Major)'
    ];

    // Drag and Drop listeners
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragging');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragging');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragging');
        if (e.dataTransfer.files.length > 0) {
            handleAudioFile(e.dataTransfer.files[0]);
        }
    });

    // Custom mouse positioning for uploader glow effect (premium detail!)
    uploadZone.addEventListener('mousemove', (e) => {
        const rect = uploadZone.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        uploadZone.style.setProperty('--x', `${x}px`);
        uploadZone.style.setProperty('--y', `${y}px`);
    });

    // Click trigger uploader
    uploadZone.addEventListener('click', () => {
        audioFileInput.click();
    });

    audioFileInput.addEventListener('change', () => {
        if (audioFileInput.files.length > 0) {
            handleAudioFile(audioFileInput.files[0]);
        }
    });

    // Main Audio processing handler
    function handleAudioFile(file) {
        // Validate MIME type or file extension
        const isAudio = file.type.startsWith('audio/') || 
                        /\.(mp3|wav|m4a|flac|ogg|aac)$/i.test(file.name);
        
        if (!isAudio) {
            showError('Invalid file format. Please upload an audio file (MP3, WAV, FLAC, M4A, OGG).');
            return;
        }

        // Hide errors, hide uploader
        uploadError.style.display = 'none';
        uploadZone.style.display = 'none';
        processingPanel.style.display = 'block';
        resultsPanel.style.display = 'none';

        // Format metadata size
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        processingFileInfo.textContent = `${file.name} (${fileSizeMB} MB)`;

        // Reset step classes
        document.querySelectorAll('.step-item').forEach(step => {
            step.className = 'step-item';
        });
        document.getElementById('step-decode').classList.add('active');
        progressBarFill.style.width = '0%';

        // Start Web Audio API decoding check (calculates actual duration)
        let durationSeconds = 0;
        let sampleRate = 44100;
        const startTime = Date.now();

        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const audioCtx = new AudioContextClass();
            const reader = new FileReader();

            reader.onload = function(event) {
                const arrayBuffer = event.target.result;
                audioCtx.decodeAudioData(arrayBuffer)
                    .then((audioBuffer) => {
                        durationSeconds = audioBuffer.duration;
                        sampleRate = audioBuffer.sampleRate;
                        // Minimum simulation delay to feel premium
                        const elapsed = Date.now() - startTime;
                        const delay = Math.max(500, 1500 - elapsed);
                        setTimeout(() => {
                            runSimulation(file, durationSeconds, sampleRate);
                        }, delay);
                    })
                    .catch((err) => {
                        console.warn('Browser failed decoding audio buffer. Falling back to mockup.', err);
                        fallbackSimulation(file);
                    });
            };

            reader.readAsArrayBuffer(file);
        } catch (err) {
            console.warn('AudioContext not supported. Falling back to mockup.', err);
            fallbackSimulation(file);
        }
    }

    function fallbackSimulation(file) {
        // Use a mock duration (between 120 and 320 seconds) and standard sample rate
        const mockDuration = Math.floor(120 + Math.random() * 200);
        setTimeout(() => {
            runSimulation(file, mockDuration, 44100);
        }, 1200);
    }

    function showError(message) {
        errorText.textContent = message;
        uploadError.style.display = 'block';
    }

    // Processing animation and mock calculations
    function runSimulation(file, durationSeconds, sampleRate) {
        let progress = 0;
        const stepDecode = document.getElementById('step-decode');
        const stepEmbed = document.getElementById('step-embed');
        const stepClassify = document.getElementById('step-classify');

        stepDecode.className = 'step-item completed';
        stepEmbed.className = 'step-item active';
        processingTitle.textContent = 'Computing spectrogram embeddings...';

        const interval = setInterval(() => {
            progress += Math.floor(Math.random() * 8) + 2;
            
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                progressBarFill.style.width = '100%';
                
                stepClassify.className = 'step-item completed';
                processingTitle.textContent = 'Analysis complete!';
                
                setTimeout(() => {
                    renderResults(file, durationSeconds, sampleRate);
                }, 400);
            } else {
                progressBarFill.style.width = `${progress}%`;
                
                if (progress > 35 && progress < 70 && stepEmbed.className !== 'step-item active') {
                    stepDecode.className = 'step-item completed';
                    stepEmbed.className = 'step-item active';
                }
                
                if (progress >= 70 && stepClassify.className !== 'step-item active') {
                    stepEmbed.className = 'step-item completed';
                    stepClassify.className = 'step-item active';
                    processingTitle.textContent = 'Running zero-shot MERT classification...';
                }
            }
        }, 80);
    }

    // Render results view
    function renderResults(file, durationSeconds, sampleRate) {
        // Deterministic seeding based on file name character sum
        const fileSeed = file.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
        
        // Pick primary subgenre
        const genreIndex = fileSeed % SUBGENRES.length;
        const primaryGenre = SUBGENRES[genreIndex];

        // Format duration (MM:SS)
        const mins = Math.floor(durationSeconds / 60);
        const secs = Math.floor(durationSeconds % 60);
        const durationStr = `${mins}:${secs < 10 ? '0' : ''}${secs}`;

        // Generate metadata formats
        const fileExt = file.name.split('.').pop().toUpperCase() || 'MP3';
        const sampleRateKHz = (sampleRate / 1000).toFixed(1);

        // Compute metrics
        const confidence = 75 + (fileSeed % 22); // 75% to 96%
        const bpm = primaryGenre.bpmMin + (fileSeed % (primaryGenre.bpmMax - primaryGenre.bpmMin + 1));
        const keyIndex = fileSeed % KEYS.length;
        const keyVal = KEYS[keyIndex];
        const energyRating = 6.5 + ((fileSeed % 30) / 10); // 6.5 to 9.5

        // Update DOM
        resultTrackName.textContent = file.name;
        resultTrackMeta.textContent = `${fileExt} • ${sampleRateKHz} kHz • ${durationStr} duration`;
        resultSubgenre.textContent = primaryGenre.name;
        resultSubgenre.style.textShadow = `0 0 15px ${primaryGenre.color}40`;
        
        // Handle gradients on text
        resultSubgenre.style.background = `linear-gradient(90deg, #ffffff 0%, ${primaryGenre.color} 100%)`;
        resultSubgenre.style.webkitBackgroundClip = 'text';

        resultConfidence.textContent = `${confidence}%`;
        resultConfidenceBar.style.width = `${confidence}%`;
        resultConfidenceBar.style.background = `linear-gradient(90deg, var(--accent) 0%, ${primaryGenre.color} 100%)`;

        resultBpm.textContent = `${bpm} BPM`;
        resultKey.textContent = keyVal;
        resultEnergyText.textContent = `${energyRating.toFixed(1)} / 10`;
        
        // Render gold scale dots
        if (resultEnergyDots) {
            resultEnergyDots.innerHTML = '';
            const filledDotsCount = Math.round(energyRating);
            for (let i = 1; i <= 10; i++) {
                const dot = document.createElement('div');
                dot.className = `energy-dot ${i <= filledDotsCount ? 'filled' : ''}`;
                resultEnergyDots.appendChild(dot);
            }
        }

        // Generate Probabilities list (Deterministic mix of other genres)
        probabilityDistributionList.innerHTML = '';
        
        // Clone and sort other genres
        let probs = SUBGENRES.map((g, idx) => {
            let val = 0;
            if (idx === genreIndex) {
                val = confidence;
            } else {
                // Distribute remaining probability
                const offset = (fileSeed + idx) % 15 + 2;
                val = Math.floor(((100 - confidence) * offset) / 38);
            }
            return { name: g.name, val: val, color: g.color, isTop: idx === genreIndex };
        });

        // Sort descending
        probs.sort((a, b) => b.val - a.val);

        // Normalize to ensure total is exactly 100%
        const sum = probs.reduce((acc, p) => acc + p.val, 0);
        if (sum !== 100) {
            const diff = 100 - sum;
            probs[0].val += diff;
        }

        probs.forEach(p => {
            const row = document.createElement('div');
            row.className = `prob-row ${p.isTop ? 'top-probability' : ''}`;
            row.innerHTML = `
                <span class="prob-name">${p.name}</span>
                <div class="prob-bar-wrapper">
                    <div class="prob-bar-fill" style="width: ${p.val}%; background-color: ${p.color}; box-shadow: 0 0 8px ${p.color}40;"></div>
                </div>
                <span class="prob-val">${p.val}%</span>
            `;
            probabilityDistributionList.appendChild(row);
        });

        // Hide processing, show results
        processingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
    }

    // Reset button handler
    resetButton.addEventListener('click', () => {
        audioFileInput.value = '';
        resultsPanel.style.display = 'none';
        uploadZone.style.display = 'block';
    });

    // Simulation button handler
    const simulateUploadBtn = document.getElementById('simulate-upload-btn');
    if (simulateUploadBtn) {
        simulateUploadBtn.addEventListener('click', () => {
            const file = new File(["dummy content"], "neon_vibrations.mp3", {type: "audio/mp3"});
            handleAudioFile(file);
        });
    }

    // Feedback modal logic
    const feedbackTrigger = document.getElementById('feedback-trigger');
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackClose = document.getElementById('feedback-close');
    const feedbackSubmit = document.getElementById('feedback-submit');
    const feedbackSuccessMsg = document.getElementById('feedback-success-msg');
    
    const feedbackName = document.getElementById('feedback-name');
    const feedbackText = document.getElementById('feedback-text');

    if (feedbackTrigger && feedbackModal) {
        feedbackTrigger.addEventListener('click', () => {
            feedbackModal.style.display = 'flex';
            feedbackSuccessMsg.style.display = 'none';
            if (feedbackText) feedbackText.value = '';
            if (feedbackName) feedbackName.value = '';
        });
    }

    if (feedbackClose && feedbackModal) {
        feedbackClose.addEventListener('click', () => {
            feedbackModal.style.display = 'none';
        });
    }

    if (feedbackModal) {
        feedbackModal.addEventListener('click', (e) => {
            if (e.target === feedbackModal) {
                feedbackModal.style.display = 'none';
            }
        });
    }

    if (feedbackSubmit) {
        feedbackSubmit.addEventListener('click', () => {
            if (feedbackText && !feedbackText.value.trim()) {
                alert('Please enter your suggestions before submitting.');
                return;
            }
            if (feedbackSuccessMsg) {
                feedbackSuccessMsg.style.display = 'block';
            }
            setTimeout(() => {
                feedbackModal.style.display = 'none';
            }, 1500);
        });
    }
});
