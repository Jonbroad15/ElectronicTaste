import json
import re
import os
import itertools
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.templates.default = "plotly_dark"

# 1. Load Data
LABELS_PATH = "data/labels.json"
MANIFEST_PATH = "data/djmix_manifest_raw.json"
TAXONOMY_PATH = "data/pulseroots_taxonomy.json"

print("Loading data...")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_data = json.load(f)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy_data = json.load(f)

# 2. Extract Durations
print("Extracting durations...")
def get_duration(entry):
    tracklist = entry.get('tracklist', [])
    max_min = 0
    for t in tracklist:
        title = t.get('title', '')
        if not title: continue
        m = re.search(r'\[(\d+)\]', title)
        if m:
            val = int(m.group(1))
            if val > max_min:
                max_min = val
    if max_min == 0:
        return 3600
    return (max_min + 5) * 60

manifest_durations = {}
for entry in manifest_data:
    mix_id = entry.get("id")
    if mix_id:
        manifest_durations[mix_id] = get_duration(entry)

# 3. Process Labels and Compute Metrics
files = labels_data.get("files", {})
total_duration_sec = 0
total_samples = 0

l1_counts = {}
l2_counts = {}
l3_counts = {}

# Matrices
l1_co_occurrences = {}
durations_list = []
num_labels_list = []

print(f"Processing {len(files)} annotated mixes...")
for filename, info in files.items():
    mix_id = filename.replace(".wav", "")
    duration = manifest_durations.get(mix_id, 3600)
    samples = int(duration / 30)
    
    total_duration_sec += duration
    total_samples += samples
    durations_list.append(duration / 60) # minutes
    
    l1 = info.get("l1_genres", [])
    l2 = info.get("l2_genres", [])
    l3 = info.get("l3_genres", [])
    
    num_labels_list.append(len(l1) + len(l2) + len(l3))
    
    for g in l1: l1_counts[g] = l1_counts.get(g, 0) + samples
    for g in l2: l2_counts[g] = l2_counts.get(g, 0) + samples
    for g in l3: l3_counts[g] = l3_counts.get(g, 0) + samples
    
    for g1, g2 in itertools.combinations(l1, 2):
        pair = tuple(sorted([g1, g2]))
        l1_co_occurrences[pair] = l1_co_occurrences.get(pair, 0) + samples

# 4. Validation
print("\n--- Running Validations ---")
# Check 1: Data Consistency
expected_samples = sum(d // 30 for d in durations_list * 60) # Note: durations_list is in minutes
calculated_samples = sum([manifest_durations.get(f.replace(".wav", ""), 3600) // 30 for f in files])
print(f"Total Duration (sec): {total_duration_sec}")
print(f"Total 30s Samples: {total_samples}")
if total_samples == calculated_samples:
    print("✅ Validation 1 Passed: Computed number of 30s samples aligns mathematically.")
else:
    print(f"❌ Validation 1 Failed: {total_samples} != {calculated_samples}")

# Check 2: Taxonomy Alignment
# Verify that all extracted L1 genres exist in taxonomy
def get_all_pulseroots_nodes(nodes, depth=1):
    result = {}
    for node in nodes:
        name = node.get("name") or node.get("style", "")
        if name:
            result[name] = depth
            substyles = node.get("substyles", [])
            if substyles:
                result.update(get_all_pulseroots_nodes(substyles, depth + 1))
    return result

pr_nodes = get_all_pulseroots_nodes(taxonomy_data)
missing_nodes = [g for g in l1_counts if g not in pr_nodes]
if not missing_nodes:
    print("✅ Validation 2 Passed: Nodes rendered correctly map to pulseroots hierarchy.")
else:
    print(f"❌ Validation 2 Failed: Unknown nodes found {missing_nodes}")

print("---------------------------\n")

# 5. Generate Visualizations
os.makedirs("eda", exist_ok=True)
html_components = []

def add_plot(fig, title):
    html_components.append(f"<h2>{title}</h2>")
    html_components.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

# 5.1 Class Distributions (L1)
df_l1 = pd.DataFrame(list(l1_counts.items()), columns=['Genre', 'Samples']).sort_values('Samples', ascending=True)
fig_l1 = px.bar(df_l1, x='Samples', y='Genre', orientation='h', color_discrete_sequence=['#ba55d3'], title="L1 Genre Distribution (30s Samples)")
add_plot(fig_l1, "L1 Genre Distribution")

# L2 Distribution
df_l2 = pd.DataFrame(list(l2_counts.items()), columns=['Genre', 'Samples']).sort_values('Samples', ascending=True).tail(30)
fig_l2 = px.bar(df_l2, x='Samples', y='Genre', orientation='h', color_discrete_sequence=['#e0b5f1'], title="Top 30 L2 Genre Distribution")
add_plot(fig_l2, "L2 Genre Distribution (Top 30)")

# L3 Distribution
df_l3 = pd.DataFrame(list(l3_counts.items()), columns=['Genre', 'Samples']).sort_values('Samples', ascending=True).tail(30)
fig_l3 = px.bar(df_l3, x='Samples', y='Genre', orientation='h', color_discrete_sequence=['#ffffff'], title="Top 30 L3 Genre Distribution")
add_plot(fig_l3, "L3 Genre Distribution (Top 30)")

# 5.2 Multi-Label Co-occurrences
unique_l1 = sorted(list(l1_counts.keys()))
co_matrix = np.zeros((len(unique_l1), len(unique_l1)))
for i, g1 in enumerate(unique_l1):
    for j, g2 in enumerate(unique_l1):
        if i == j:
            co_matrix[i, j] = l1_counts[g1]
        else:
            pair = tuple(sorted([g1, g2]))
            co_matrix[i, j] = l1_co_occurrences.get(pair, 0)

fig_co = px.imshow(co_matrix, x=unique_l1, y=unique_l1, color_continuous_scale='Purples', 
                   title="L1 Genre Co-occurrence (Diagonal=Total Count)")
fig_co.update_layout(xaxis_tickangle=-45)
add_plot(fig_co, "L1 Multi-Label Co-occurrences")

# 5.3 Taxonomy Coverage Tree (Sunburst)
# Build a dataframe for the sunburst chart. We need id, parent, value, color.
tree_data = []
# Root
tree_data.append({"id": "All Mixes", "parent": "", "value": total_samples, "color": total_samples, "text": "All Mixes"})

def walk_tree_for_sunburst(nodes, parent_id):
    for node in nodes:
        name = node.get("name") or node.get("style", "")
        if not name: continue
        
        # Combine counts (using the specific dict depending on depth, or just check all)
        count = l1_counts.get(name, 0) + l2_counts.get(name, 0) + l3_counts.get(name, 0)
        
        # We only add nodes that have coverage, or grey them out if they have 0
        tree_data.append({"id": name, "parent": parent_id, "value": max(count, 1), "color": count, "text": name if count > 0 else ""})
        
        substyles = node.get("substyles", [])
        if substyles:
            walk_tree_for_sunburst(substyles, name)

walk_tree_for_sunburst(taxonomy_data, "All Mixes")
df_tree = pd.DataFrame(tree_data)

# Grey out 0-coverage branches by using a custom color scale
fig_tree = px.sunburst(df_tree, names='id', parents='parent', values='value', color='color',
                       color_continuous_scale='Purples',
                       title="Taxonomy Coverage Tree (Grey = 0 coverage, Purple = High Coverage)")
# Fix coloring so 0 count maps to grey explicitly, hide text for 0-count
fig_tree.update_traces(
    text=df_tree['text'], 
    textinfo='text', 
    insidetextorientation='radial',
    marker=dict(colorscale=[[0, '#333333'], [0.00001, '#e0b5f1'], [1, '#ba55d3']])
)
fig_tree.update_layout(height=800, margin=dict(t=50, l=0, r=0, b=0))
add_plot(fig_tree, "Taxonomy Coverage Tree (L1-L4)")

# 5.4 Mix Duration Distribution
df_durations = pd.DataFrame({"Duration (Minutes)": durations_list})
df_dur_filtered = df_durations[df_durations["Duration (Minutes)"] <= 400]
fig_dur = px.histogram(df_dur_filtered, x="Duration (Minutes)", color_discrete_sequence=['#ba55d3'], title="Mix Duration Distribution (Capped at 400 mins)")
fig_dur.update_traces(xbins=dict(start=0, end=400, size=10))
add_plot(fig_dur, "Mix Duration Distribution")

# 5.5 Labels per Mix Distribution
df_labels = pd.DataFrame({"Total Labels per Mix": num_labels_list})
fig_labels = px.histogram(df_labels, x="Total Labels per Mix", nbins=20, 
                          color_discrete_sequence=['#e0b5f1'], title="Total Labels (L1+L2+L3) per Mix")
add_plot(fig_labels, "Labels per Mix Distribution")

# 6. Assemble Final HTML
print("Writing final HTML dashboard...")
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Raveform Dataset EDA</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background-color: #f8f9fa; }}
        h1 {{ text-align: center; color: #343a40; }}
        h2 {{ color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-top: 50px; }}
        .plot-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>Raveform Dataset EDA Dashboard</h1>
    <p style="text-align:center;">Total Mixes: {len(files)} | Total Duration: {total_duration_sec / 3600:.2f} Hours | Total 30s Samples: {total_samples}</p>
    {"".join([f'<div class="plot-container">{comp}</div>' for comp in html_components if not comp.startswith('<h')])}
</body>
</html>
"""

# Small hack to interleave headers properly
final_html_parts = []
final_html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Raveform Dataset EDA</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Outfit', sans-serif; margin: 0; padding: 20px; background-color: transparent; color: #f0f0f0; }
        h1 { display: none; }
        h2 { color: #ba55d3; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-top: 30px; font-weight: 600; }
        .plot-container { background: transparent; padding: 0; margin-bottom: 30px; }
        .stats-bar { text-align: center; font-size: 1.1rem; color: #a0a0a0; margin-bottom: 2rem; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
</head>
<body>
""")
final_html_parts.append(f'<div class="stats-bar"><strong>Total Mixes:</strong> {len(files)} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Total Duration:</strong> {total_duration_sec / 3600:.2f} Hours &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Total 30s Samples:</strong> {total_samples}</div>')

preamble_html = """
<div style="background: rgba(255,255,255,0.02); padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 30px; line-height: 1.6;">
    <h3 style="margin-top: 0; border: none; padding-bottom: 0; color: #ba55d3; font-size: 1.4rem;">About the Raveform Dataset</h3>
    <p>
        <strong>Raveform</strong> is an expansive, hierarchically structured dataset of Electronic Dance Music (EDM) designed to train state-of-the-art multi-label audio classifiers. 
        It builds upon the <a href="https://github.com/taejunkim/raveform" target="_blank" style="color: #e0b5f1; text-decoration: none;">original Raveform research</a> and the massive 
        <a href="https://github.com/mir-aidj/djmix-dataset" target="_blank" style="color: #e0b5f1; text-decoration: none;">djmix-dataset</a> by enriching thousands of continuous DJ mixes 
        with deep taxonomic metadata.
    </p>
    <p>
        The genre hierarchy used in this dataset is strictly governed by the 
        <a href="https://github.com/Mendiak/pulse.roots" target="_blank" style="color: #e0b5f1; text-decoration: none;">Pulseroots Taxonomy</a>. 
        By mapping flat, noisy genre tags into a multi-level L1-L4 tree structure, we aim to train models that understand not just base genres (like Techno), 
        but their intricate sub-styles and cross-pollinations (e.g., Deep Tech House).
    </p>
</div>
"""
final_html_parts.append(preamble_html)


for comp in html_components:
    if comp.startswith("<h2"):
        final_html_parts.append(comp)
    else:
        final_html_parts.append(f'<div class="plot-container">{comp}</div>')

final_html_parts.append("""
</body>
</html>
""")

with open("eda/index.html", "w", encoding="utf-8") as f:
    f.write("".join(final_html_parts))

print("EDA dashboard generated successfully at eda/index.html")
