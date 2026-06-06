# Phase 4: Execution Plan

## Step 1: DOM Structure Setup
- Modify [index.html](file:///Users/mbroadbent/Code/ElectronicTaste/index.html) to incorporate the new "UI Demo" tab under the navigation system.
- Create a container section (`#ui-demo`) containing:
  - An interactive uploader box with icons, a drag-and-drop boundary, and a hidden native file input.
  - A loading overlay with a progress bar and status text (e.g. "Decoding audio buffer...", "Extracting spectrogram embeddings...").
  - An analysis results panel with rounded card blocks, metadata details, and interactive sliders showing simulated genre probabilities, BPM, key, energy, and vocal ratios.
- **Output**: Unstyled HTML DOM components added to the dashboard.

## Step 2: Stylistic Theme & Rounded Layouts
- Modify [style.css](file:///Users/mbroadbent/Code/ElectronicTaste/style.css) to replace the light blue/cyan secondary glow with a vibrant lime/neon green (`#00e676`), creating the requested purple and green color theme.
- Add general custom variables for green highlights (`--accent-green`).
- Design modern Google-style cards using clean glassmorphism styling, soft border gradients, and distinct rounded corners (`border-radius: 24px` on main blocks, `16px` on sub-items).
- Implement interactive hover, focus, active, and dragging states for the drag-and-drop zone.
- **Output**: Visually stunning dashboard consistent with a premium raver theme.

## Step 3: File Input & Ingestion Handlers
- Update [script.js](file:///Users/mbroadbent/Code/ElectronicTaste/script.js) to register the new `ui-demo` section in the tab navigation switcher.
- Implement event listeners for drag-and-drop operations (`dragover`, `dragleave`, `drop`) on the dropzone, adding visual highlight classes dynamically.
- Implement manual file input change listeners.
- Validate file types to filter for standard audio containers.
- **Output**: Functional uploader capturing local user audio files.

## Step 4: Web Audio API Decoding & Simulated Analysis
- Build an audio parser function using `window.AudioContext` or `window.webkitAudioContext`.
- Read files as `ArrayBuffer` and decode the audio data.
- Retrieve the actual sample rate and precise length (duration) of the uploaded track.
- Implement a multi-step simulated classification pipeline:
  1. *Step 1*: Ingest & Parse (0-500ms)
  2. *Step 2*: Audio decoding & duration extraction (runs asynchronously, up to 1500ms depending on file size)
  3. *Step 3*: Feature extraction simulation (MERT embedding model inference - 1000ms)
  4. *Step 4*: Subgenre decision & metadata compile (500ms)
- Randomize classification outputs based on the file name or properties to simulate organic subgenre predictions, BPMs, keys, and voice metrics.
- Display the completed analysis dashboard with beautiful animations.
- **Output**: A fully working mock interactive analysis flow.
