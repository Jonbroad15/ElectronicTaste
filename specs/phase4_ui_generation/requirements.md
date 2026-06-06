# Phase 4: Requirements (UI Generation)

## Objectives
The primary goal of Phase 4 is to design and implement a high-fidelity browser user interface for the Electric Taste project. This interface will allow users to upload audio files (individual songs or mixes) and view a mockup of the subgenre analysis, showcasing the future capability of the core machine learning models in a modern, interactive web application.

## Specific Requirements

### 1. Aesthetic and Visual Identity
- **Theme**: Follow a dark mode aesthetic with purple and green ambient glows.
- **Styling**: Google-like layout featuring modern rounded corner edges (`border-radius: 24px` for main containers, `16px` for cards).
- **Interactions**: Include smooth transitions, hover states, and dynamic visual cues (e.g., active drop states for drag-and-drop).
- **Responsive Layout**: Ensure the interface works seamlessly on desktop, tablet, and mobile screens.

### 2. Tab Integration
- Add a new "UI Demo" tab to the existing navigation system of the Project Constitution app.
- Protect this section under the existing password authorization gate (`raver`).

### 3. Audio File Upload & Ingestion
- **Drag-and-Drop Area**: A visually distinct zone where users can drag and drop audio files.
- **Fallback File Input**: A clickable selector to choose files via the OS file explorer.
- **Supported Formats**: Target standard audio containers (e.g., `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`).
- **File Validation**: Validate that the uploaded file is indeed a supported audio file and provide clear feedback if not.

### 4. Client-side Analysis Simulation
- **File Metadata Extraction**: Read file name, format, and size.
- **Audio Context Decoding**: Utilize the browser's Web Audio API to parse and decode the file into an `AudioBuffer` to retrieve its exact duration.
- **Processing Queue**: Show an active processing spinner or progress bar during decoding.
- **Interactive Results Dashboard**: Once processed, display an analysis card with:
  - File Information (Name, Size, Duration).
  - Classification Metrics:
    - Predicted Subgenre (e.g., Progressive House, Melodic Techno, Drum & Bass, Dubstep, Trance).
    - BPM (Beats Per Minute) estimation.
    - Key detection (e.g., A Minor, 8A, F# Major).
    - Energy Level (high, medium, low) and Vocal/Instrumental ratio.
  - Interactive Action to reset the uploader and upload another song.
