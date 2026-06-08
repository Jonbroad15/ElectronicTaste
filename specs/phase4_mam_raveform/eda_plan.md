# Raveform Dataset Exploratory Data Analysis (EDA) Specification

## 1. Objective
Produce a comprehensive, visually engaging Exploratory Data Analysis (EDA) of the expanded Raveform dataset. The final EDA will be published to the project's GitHub Pages website to showcase the dataset's scale, hierarchy, and diversity.

## 2. Data Sources & Metrics
- **Data Sources:** `data/labels.json`, `data/pulseroots_taxonomy.json`, and `data/djmix_manifest_raw.json`.
- **Sample Size Definition:** 1 sample = 30 seconds of audio.
- **Scale:** All class/node statistics will compute both:
  - Total duration (in seconds/hours)
  - Total number of 30s samples

## 3. Required Visualizations

### 3.1. Class Distributions (L1, L2, L3)
- **Format:** Bar charts or horizontal bar plots.
- **Purpose:** Show the absolute volume of data available for each genre at different hierarchy levels (L1, L2, L3).
- **Details:** Bars will be labeled with both sample counts and total durations.

### 3.2. Multi-Label Co-occurrences
- **Format:** Heatmap or Chord Diagram.
- **Purpose:** Illustrate how often genres co-occur within the same mix (e.g., mixes tagged as both *House* and *Techno*).
- **Details:** Matrix showing the intersection frequency of top-level genres to highlight cross-genre representation.

### 3.3. Taxonomy Coverage Tree (L1-L3)
- **Format:** Hierarchical Tree Diagram (e.g., Sunburst chart, Dendrogram, or D3.js tree).
- **Purpose:** Map out the `pulseroots` taxonomy and visually demonstrate our dataset's coverage of it.
- **Details:** 
  - Branches and nodes will be color-coded by coverage intensity.
  - Nodes with high sample counts will use bright/intense colors.
  - Nodes with zero or very low coverage will be greyed out.

### 3.4. Additional Proposed Figures
1. **Mix Duration Distribution:** Histogram showing the distribution of mix lengths (in minutes/hours) across the dataset.
2. **Labels per Mix Distribution:** Histogram showing the density of multi-labeling (i.e., how many distinct genres a typical mix is assigned).
3. **Temporal Coverage (Optional):** If release years or dates are available in the manifest, a timeline showing the temporal distribution of the mixes.

## 4. Implementation & Publishing Plan
1. **Data Extraction Script:** Write a Python script to cross-reference `labels.json` with `djmix_manifest_raw.json` to calculate exact durations and sample counts per mix.
2. **Visualization Generation:** Generate all plots as interactive Plotly/D3.js visualizations (HTML widgets) to provide an engaging experience on the project website.
3. **Web Integration:** Compile the figures and statistical summaries into a web-ready format (Markdown/HTML) and integrate it into the existing GitHub Pages repository structure.

## 5. Validation
Before publishing the EDA to the project website, the following validation steps must be completed:

1. **Data Consistency Check:** 
   - Verify that the total duration across all mixes matches the sum of durations calculated per class (accounting for multi-labels appropriately without double-counting the total dataset length).
   - Ensure that the computed number of 30s samples aligns mathematically with the extracted durations.
2. **Taxonomy Alignment:**
   - Confirm that the nodes rendered in the taxonomy tree correctly map to the canonical `pulseroots` hierarchy and that parent-child relationships are strictly preserved.
3. **Interactive Widget Testing:**
   - Test all generated Plotly/D3.js interactive HTML widgets in a local browser environment to ensure hover states, zooming, and tooltips function without JavaScript errors.
4. **Visual Readability:**
   - Review color scaling on heatmaps and taxonomy trees to ensure that low-coverage vs. high-coverage areas are visually distinct and accessible.
   - Ensure labels and legends on the interactive plots are legible and descriptive.
