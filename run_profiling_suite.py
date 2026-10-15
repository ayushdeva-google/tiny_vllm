"""
End-to-End LLM Profiling Pipeline and Results Publisher for tiny_vllm.

Orchestrates:
1. PyTorch Fine-Grained Sampled Profiler (`llama_inference_with_profiling.py`):
   - Generates per-step Chrome trace JSONs
   - Computes the Three Physical Metrics: CPU Launch Gaps, VRAM Data Wait, Math Compute
   - Emits terminal report and interactive HTML dashboard (`profile_dashboard.html`)
2. Nsight Systems Hardware Profiling (`profile_micro_nsys.py`):
   - Targets isolated single-token decode with NVTX and cudaProfilerApi
   - Emits hardware trace (`nsys_hardware_profile.nsys-rep`) and exports SQLite database
3. Report Bundler & Publisher:
   - Packages all traces, dashboards, and databases into `profile_reports_bundle.tar.gz` and `.zip`
"""

import os
import sys
import shutil
import tarfile
import zipfile
import subprocess
from pathlib import Path

WORKSPACE = Path("/home/ayushdeva_google_com/tiny_vllm")
PYTHON_EXE = WORKSPACE / ".venv/bin/python"
NSYS_EXE = WORKSPACE / "tools/nsys_pkg/opt/nvidia/nsight-systems/2023.1.2/bin/nsys"
OUTPUT_DIR = WORKSPACE / "profile_results"


def run_command(cmd, desc):
    print(f"\n{'='*70}\n[*] {desc}\n[*] Command: {' '.join(str(c) for c in cmd)}\n{'='*70}")
    res = subprocess.run(cmd, cwd=str(WORKSPACE))
    if res.returncode != 0:
        print(f"[!] Error: {desc} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✓] {desc} completed successfully.\n")


def main():
    print("=" * 70)
    print("  TINY_VLLM: END-TO-END PROFILING SUITE & PUBLISHER")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Run PyTorch Fine-Grained Sampled Profiler
    pytorch_cmd = [
        str(PYTHON_EXE),
        "llama_inference_with_profiling.py",
        "--prompt", "Explain the core architectural bottlenecks of autoregressive LLM inference.",
        "--max_new_tokens", "150",
        "--profile_interval", "100",
        "--profile_output_dir", str(OUTPUT_DIR),
        "--warmup",
    ]
    run_command(pytorch_cmd, "Stage 1: PyTorch Fine-Grained Sampled Profiling (150 tokens)")

    # 2. Run Nsight Systems Hardware Profiling
    nsys_rep_path = OUTPUT_DIR / "nsys_hardware_profile"
    nsys_cmd = [
        str(NSYS_EXE),
        "profile",
        "-c", "cudaProfilerApi",
        "--capture-range-end", "stop",
        "--sample=none",
        "--trace=cuda,nvtx,osrt",
        "-o", str(nsys_rep_path),
        "-f", "true",
        str(PYTHON_EXE),
        "profile_micro_nsys.py",
        "--warmup_tokens", "1",
        "--output_dir", str(OUTPUT_DIR / "nsys_micro"),
    ]
    run_command(nsys_cmd, "Stage 2: Nsight Systems Isolated Hardware Profiling (1 token decode)")

    # 3. Export Nsys SQLite database
    sqlite_cmd = [
        str(NSYS_EXE),
        "export",
        "--type", "sqlite",
        "-o", str(OUTPUT_DIR / "nsys_hardware_profile.sqlite"),
        "-f", "true",
        str(OUTPUT_DIR / "nsys_hardware_profile.nsys-rep"),
    ]
    run_command(sqlite_cmd, "Stage 3: Exporting Nsight Systems SQLite Database")

    # 4. Bundle artifacts for publishing
    print("\n" + "=" * 70)
    print("[*] Stage 4: Bundling Profiling Reports for Publication")
    print("=" * 70)

    tar_path = WORKSPACE / "profile_reports_bundle.tar.gz"
    zip_path = WORKSPACE / "profile_reports_bundle.zip"

    # Tar.gz bundle
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(OUTPUT_DIR), arcname="profile_results")
    print(f"[✓] Created tarball bundle: {tar_path} ({os.path.getsize(tar_path) / 1024 / 1024:.2f} MB)")

    # Zip bundle
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, WORKSPACE)
                zipf.write(file_path, arcname=rel_path)
    print(f"[✓] Created zip bundle: {zip_path} ({os.path.getsize(zip_path) / 1024 / 1024:.2f} MB)")

    # 5. Summary Verification
    print("\n" + "=" * 70)
    print("  FINAL PUBLISHING VERIFICATION SUMMARY")
    print("=" * 70)
    expected_files = [
        OUTPUT_DIR / "profile_dashboard.html",
        OUTPUT_DIR / "token_metrics.json",
        OUTPUT_DIR / "traces/step_0_trace.json",
        OUTPUT_DIR / "traces/step_100_trace.json",
        OUTPUT_DIR / "traces/step_149_trace.json",
        OUTPUT_DIR / "nsys_hardware_profile.nsys-rep",
        OUTPUT_DIR / "nsys_hardware_profile.sqlite",
        tar_path,
        zip_path,
    ]

    for p in expected_files:
        status = "[✓] FOUND" if p.exists() else "[!] MISSING"
        size_str = f"({p.stat().st_size / 1024:.1f} KB)" if p.exists() else ""
        print(f"  {status} {p.name} {size_str}")

    print("\n[✓] All profilers successfully executed and results packaged for publishing!\n")


if __name__ == "__main__":
    main()
