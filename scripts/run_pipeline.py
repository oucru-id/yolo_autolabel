#!/usr/bin/env python3

import subprocess
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path
import time
import base64

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import BASE, VERIFICATION, TEST_RESULTS, TRAIN_RUN, OUTPUT_DIR
except ImportError:
    BASE = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE / "results"
    VERIFICATION = OUTPUT_DIR / "verification"
    TEST_RESULTS = OUTPUT_DIR / "test_results"
    TRAIN_RUN = OUTPUT_DIR / "runs" / "train"

TRAIN_RESULTS_DIR = TRAIN_RUN
TRAIN_PLOT = TRAIN_RESULTS_DIR / "training_analysis.png"
TEST_DASHBOARD = TEST_RESULTS / "test_dashboard.png"
TRAINING_HISTORY_PLOT = VERIFICATION / "history_chart.png"
RETRAIN_DIR = VERIFICATION / "retrain_ready"

def run_step(command, step_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {step_name}")
    print(f"CMD: {command}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='') 
                output_lines.append(line)
                
        rc = process.poll()
        duration = time.time() - start_time
        
        status = "Success" if rc == 0 else "FAILED"
        print(f"\n>>> {step_name} DONE ({status}) - {duration:.1f}s")
        return rc == 0, "".join(output_lines), duration
    except Exception as e:
        print(f"\n>>> {step_name} ERROR: {e}")
        return False, str(e), 0.0

def img_to_base64(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        return None

def generate_html_report(steps_data, output_dir):
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    nice_date = datetime.now().strftime("%d %B %Y, %H:%M")
    report_file = output_dir / f"Report Detail_{now_str}.html"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_duration = sum(d['duration'] for d in steps_data)
    
    css = """
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f4f6f8; color: #333; }
    .container { max-width: 1000px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-radius: 8px; }
    h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
    .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 30px; }
    .summary-box { background: #f8f9fa; padding: 20px; border-radius: 5px; border: 1px solid #e9ecef; }
    .status-pass { color: #27ae60; font-weight: bold; }
    .status-fail { color: #c0392b; font-weight: bold; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
    th { background-color: #f8f9fa; font-weight: 600; }
    .toc { background: #eef2f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
    .toc ul { list-style: none; padding: 0; }
    .toc li { margin: 8px 0; }
    .toc a { text-decoration: none; color: #2980b9; font-weight: 500; }
    .toc a:hover { text-decoration: underline; }
    .img-container { text-align: center; margin: 20px 0; border: 1px solid #eee; padding: 10px; }
    .img-container img { max-width: 100%; height: auto; border-radius: 4px; }
    .log-box { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: monospace; font-size: 0.85em; max-height: 300px; }
    """

    html = [f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Report Detail - {nice_date}</title>
        <style>{css}</style>
    </head>
    <body>
    <div class="container">
        <h1>End-to-End Analysis Report</h1>
        <div class="meta">
            Generated: {nice_date} | Duration: {total_duration:.1f}s | Project: {BASE.name}
        </div>
    """]

    html.append("""
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#summary">1. Execution Summary</a></li>
                <li><a href="#training">2. Training Analysis</a></li>
                <li><a href="#testing">3. Model Testing</a></li>
                <li><a href="#verification">4. Verification</a></li>
                <li><a href="#retrain">5. Prepare Retrain</a></li>
            </ul>
        </div>
    """)

    html.append("<h2 id='summary'>1. Execution Summary</h2>")
    html.append("<table><thead><tr><th>Step</th><th>Status</th><th>Duration</th></tr><tbody>")
    
    for step in steps_data:
        status_class = "status-pass" if step['success'] else "status-fail"
        status_text = "PASSED" if step['success'] else "FAILED"
        html.append(f"<tr><td>{step['name']}</td><td class='{status_class}'>{status_text}</td><td>{step['duration']:.1f}s</td></tr>")
    html.append("</tbody></table>")

    html.append("<h2 id='progress'>1.5 Incremental Progress</h2>")
    
    b64_history = img_to_base64(TRAIN_RESULTS_DIR.parent.parent / "verification" / "history_chart.png")
    if b64_history:
        html.append(f"<div class='img-container'><img src='data:image/png;base64,{b64_history}' alt='Incremental Progress'></div>")
    else:
        html.append("<p><i>No history chart available.</i></p>")

    html.append("<h2 id='training'>2. Training Analysis</h2>")
    
    b64_train = img_to_base64(TRAIN_PLOT)
    if b64_train:
        html.append(f"<div class='img-container'><img src='data:image/png;base64,{b64_train}' alt='Training Analysis'></div>")
    else:
        html.append("<p><i>No visualization available (runs/train/training_analysis.png not found).</i></p>")
    
    train_step = next((s for s in steps_data if "Training" in s['name']), None)
    if train_step:
        html.append("<div class='summary-box'><h3>Analysis Output</h3><ul>")
        for line in train_step['output'].splitlines():
            clean = line.strip()
            if any(k in clean for k in ["Plateau", "Early Stopping", "Diagnosis", "Recommendation"]):
                 html.append(f"<li>{clean}</li>")
        html.append("</ul></div>")

    html.append("<h2 id='testing'>3. Model Testing</h2>")
    
    b64_test = img_to_base64(TEST_DASHBOARD)
    if b64_test:
        html.append(f"<div class='img-container'><img src='data:image/png;base64,{b64_test}' alt='Test Dashboard'></div>")
    else:
        html.append("<p><i>No dashboard available.</i></p>")

    html.append("<h2 id='verification'>4. Verification Data</h2>")
    html.append("<p>Comparing Auto-Labels (YOLO) vs Ground Truth.</p>")
    verify_step = next((s for s in steps_data if "Verification" in s['name']), None)
    if verify_step:
         html.append("<div class='log-box'><pre>")
         capture = False
         for line in verify_step['output'].splitlines():
             if "SUMMARY" in line: capture = True
             if capture and line.strip():
                 html.append(line + "\n")
         html.append("</pre></div>")

    html.append("<h2 id='retrain'>5. Prepare Retrain</h2>")
    retrain_step = next((s for s in steps_data if "Prepare" in s['name']), None)
    if retrain_step:
        html.append("<div class='summary-box'>")
        for line in retrain_step['output'].splitlines():
             if "Trusted" in line or "Review" in line or "Retrain" in line or "Total pairs" in line:
                 html.append(f"<p>{line}</p>")
        html.append("</div>")

    html.append("""
    </div>
    </body>
    </html>
    """)
    
    report_file.write_text("\n".join(html), encoding="utf-8")
    print(f"\n\U0001f4dd HTML Report created: {report_file}")
    return report_file

def main():
    print("STARTING END-TO-END PIPELINE (HTML Report)...")
    
    python_exe = sys.executable
    scripts_dir = BASE / "scripts"
    
    steps = []
    
    cmd_analyze = f'"{python_exe}" "{scripts_dir / "analyze.py"}"'
    success, output, duration = run_step(cmd_analyze, "1. Analysis")
    steps.append({'name': "Training Analysis", 'success': success, 'output': output, 'duration': duration})

    cmd_test = f'"{python_exe}" "{scripts_dir / "test_model.py"}"'
    success, output, duration = run_step(cmd_test, "2. Testing")
    steps.append({'name': "Model Testing", 'success': success, 'output': output, 'duration': duration})

    if "--fine-tune" in sys.argv:
        ft_args = ""
        if "--trial-epochs" in sys.argv:
            idx = sys.argv.index("--trial-epochs")
            ft_args += f' --trial-epochs {sys.argv[idx + 1]}'
        if "--steps" in sys.argv:
            idx = sys.argv.index("--steps")
            ft_args += f' --steps {sys.argv[idx + 1]}'
        cmd_ft = f'"{python_exe}" "{scripts_dir / "fine_tune.py"}"{ft_args}'
        success, output, duration = run_step(cmd_ft, "3. Fine-Tune")
        steps.append({'name': "Hyperparameter Fine-Tune", 'success': success, 'output': output, 'duration': duration})

    try:
        from analyze import plot_training_history
        plot_training_history()
    except Exception as e:
        print(f"Failed to plot history: {e}")

    report_path = generate_html_report(steps, VERIFICATION)
    
    if os.name == 'nt':
        try:
            os.startfile(report_path)
        except Exception:
            pass
    elif sys.platform == "darwin":
        subprocess.call(["open", str(report_path)])
    elif sys.platform == "linux":
        try:
            subprocess.call(["xdg-open", str(report_path)])
        except Exception:
            print(f"Open the report manually: {report_path}")

if __name__ == "__main__":
    main()