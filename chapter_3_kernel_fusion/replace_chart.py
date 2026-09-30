import re

with open("profile_visualizer.py", "r") as f:
    content = f.read()

# Replace the specific div containing the bar chart with a clean HTML table
start_marker = "<!-- COMPARATIVE TWO-TIER HARDWARE TIME ALLOCATION BARS -->"
end_marker = "<!-- KEY INSIGHTS CALLOUT -->"

replacement = """<!-- COMPARATIVE TWO-TIER HARDWARE TIME ALLOCATION TABLE -->
            <div style="background:#0f172a;border:1px solid var(--card-border);border-radius:0.75rem;padding:1.25rem;margin-bottom:1.5rem;overflow-x:auto;">
                <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;margin-bottom:1rem;text-align:center;">
                    📊 Two-Tier Hardware Time Allocation Breakdown (Head-to-Head)
                </div>
                <table style="width:100%; border-collapse: collapse; text-align: left; color:#cbd5e1; font-size:0.85rem;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <th style="padding: 0.75rem;">Metric</th>
                            <th style="padding: 0.75rem;">Chapter 2 (PyTorch)</th>
                            <th style="padding: 0.75rem;">Chapter 3 (Kernel Fusion)</th>
                            <th style="padding: 0.75rem;">Speedup / Reduction</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 0.75rem; font-weight: 700;">Total Wall Clock</td>
                            <td style="padding: 0.75rem;">{b_wall_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #38bdf8;">{total_wall_clock_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #34d399;">{b_wall_sec / total_wall_clock_sec if total_wall_clock_sec > 0 else 0:.1f}× Faster</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 0.75rem; font-weight: 700;">Active GPU Execution</td>
                            <td style="padding: 0.75rem;">{b_gpu_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #38bdf8;">{total_gpu_active_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #34d399;">{b_gpu_sec / total_gpu_active_sec if total_gpu_active_sec > 0 else 0:.1f}× Reduction</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 0.75rem; font-weight: 700;">Host CPU Idle Gaps</td>
                            <td style="padding: 0.75rem;">{b_cpu_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #38bdf8;">{total_host_cpu_gaps_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #f87171;">Host Bound</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 0.75rem; font-weight: 700;">GPU Memory Streaming</td>
                            <td style="padding: 0.75rem;">{b_mem_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #38bdf8;">{total_mem_sec:.1f}s</td>
                            <td style="padding: 0.75rem;">-</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 0.75rem; font-weight: 700;">GPU Tensor Compute</td>
                            <td style="padding: 0.75rem;">{b_comp_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #38bdf8;">{total_comp_sec:.1f}s</td>
                            <td style="padding: 0.75rem; color: #34d399;">{b_comp_sec / total_comp_sec if total_comp_sec > 0 else 0:.1f}× Reduction</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            """

parts = content.split(start_marker)
if len(parts) > 1:
    before = parts[0]
    after = parts[1].split(end_marker, 1)[1]
    new_content = before + replacement + end_marker + after
    with open("profile_visualizer.py", "w") as f:
        f.write(new_content)
    print("Replaced successfully")
else:
    print("Markers not found")
