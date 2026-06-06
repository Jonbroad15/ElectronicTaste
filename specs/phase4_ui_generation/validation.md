# Phase 4: Validation & Success Criteria

To successfully complete the UI Generation phase, the following verification checklist must be satisfied:

## 1. Visual Design Validation
- [ ] **Purple & Green Theme**: Confirm that the background glows feature a combination of deep purple and vibrant green ambient lights rather than the previous cyan/blue combination.
- [ ] **Rounded Corner Styling**: Verify that main container boxes have Google-like heavily rounded corners (`border-radius: 24px` or `16px`) and interactive elements (buttons, inputs) are similarly styled.
- [ ] **Responsive Flow**: Check that the navigation and sections wrap correctly on smaller layouts (mobile and tablet viewports) without horizontal page scrollbars.

## 2. Ingestion Interface Validation
- [ ] **Access Security**: Verify the "UI Demo" tab is hidden until a user enters the passcode `raver` on the security gate page.
- [ ] **Upload Interactions**:
  - [ ] **Dragover State**: Dragging a file over the uploader area triggers a visual change (e.g., green dashed border highlight, hover glow).
  - [ ] **Dragleave State**: Moving the file away restores the uploader's default visual appearance.
  - [ ] **Manual Select**: Clicking the uploader opens the native OS file selection prompt.

## 3. Input Validation & Error Handling
- [ ] **Invalid File Upload**: Attempting to upload a non-audio file (e.g., `test.txt`, `image.png`) fails, cancels the flow, and displays an appropriate warning message (e.g., "Invalid file format. Please upload an audio file (MP3, WAV, etc.)").
- [ ] **Success Flow**: Ingesting a valid audio file (e.g., a `.mp3` or `.wav` track) initiates the decoding progress screen.

## 4. Processing & Analysis Verification
- [ ] **Web Audio Decoding**: The app decodes the file using the browser's `AudioContext`. Check that the actual duration displayed matches the loaded file's audio runtime.
- [ ] **Dynamic Result Display**: Once the progress bar fills, verify the results card presents:
  - Estimated BPM, key signature, subgenre name, and energy levels.
  - Re-upload button to repeat the process.
