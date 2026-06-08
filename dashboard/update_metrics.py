import subprocess
import json
import re
import os

def run():
    print("Fetching logs from VM...")
    cmd = [
        "gcloud", "compute", "ssh", "electronic-taste-train",
        "--project=project-58e658a7-9bc6-41eb-876",
        "--zone=us-central1-a",
        "--command=grep 'Loss:' /mnt/data/logs/mam_pretrain.log"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching logs: {result.stderr}")
        return

    lines = result.stdout.strip().split('\n')
    
    metrics = {
        "steps": [],
        "total_loss": [],
        "acoustic_loss": [],
        "cqt_loss": [],
        "lr": [],
        "speed": []
    }
    
    # Example line:
    # Step     50/ 50000 | Loss: 7.4231 (Acoustic: 6.5432, CQT: 0.1234) | LR: 1.00e-04 | Speed: 0.38 steps/s
    pattern = re.compile(
        r"Step\s+(?P<step>\d+)/\s*\d+\s*\|\s*"
        r"Loss:\s*(?P<loss>[\d.]+)\s*"
        r"\(Acoustic:\s*(?P<acoustic>[\d.]+),\s*CQT:\s*(?P<cqt>[\d.]+)\)\s*\|\s*"
        r"LR:\s*(?P<lr>[\d.e+-]+)\s*\|\s*"
        r"Speed:\s*(?P<speed>[\d.]+)\s*steps/s"
    )
    
    for line in lines:
        match = pattern.search(line)
        if match:
            metrics["steps"].append(int(match.group("step")))
            metrics["total_loss"].append(float(match.group("loss")))
            metrics["acoustic_loss"].append(float(match.group("acoustic")))
            metrics["cqt_loss"].append(float(match.group("cqt")))
            metrics["lr"].append(float(match.group("lr")))
            metrics["speed"].append(float(match.group("speed")))
            
    out_path = os.path.join(os.path.dirname(__file__), "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f)
        
    print(f"Updated metrics.json with {len(metrics['steps'])} data points.")
    
    # Also push to the metrics-data worktree
    worktree_dir = os.path.join(os.path.dirname(__file__), "../../metrics-data/dashboard")
    os.makedirs(worktree_dir, exist_ok=True)
    worktree_path = os.path.join(worktree_dir, "metrics.json")
    
    with open(worktree_path, "w") as f:
        json.dump(metrics, f)
        
    subprocess.run(["git", "add", "dashboard/metrics.json"], cwd=os.path.dirname(worktree_dir))
    res = subprocess.run(["git", "commit", "-m", f"Auto-update metrics step {metrics['steps'][-1] if metrics['steps'] else 0}"], cwd=os.path.dirname(worktree_dir), capture_output=True)
    if res.returncode == 0:
        subprocess.run(["git", "push"], cwd=os.path.dirname(worktree_dir))
        print("Successfully pushed metrics.json to origin/metrics-data")
    else:
        print("No changes to commit.")

if __name__ == "__main__":
    run()
