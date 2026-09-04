"""
Profile Visualizer for tiny_vllm.

Extracts fine-grained token-wise latency and sub-operator time composition:
- Embedding
- RMSNorm_Attn (pre-attention norm)
- QKV_Linear (Q, K, V linear projections)
- RoPE (rotary position embeddings)
- Attn_Compute (GQA broadcast, attention matrix QK^T, mask, softmax, PV)
- O_Linear (attention output projection)
- RMSNorm_FFN (post-attention norm)
- FFN_Gate_Up_Linear (SwiGLU gate & up projections)
- FFN_SiLU_Mul (SiLU activation & elementwise multiplication)
- FFN_Down_Linear (SwiGLU down projection)
- RMSNorm_Final (final normalization)
- LM_Head (vocabulary projection)
- Sampling (argmax / multinomial + GPU-to-CPU sync)
- Tokenizer_Decode (string detokenization)
"""

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# High-level fine-grained categories tracked in the visualizer
FINE_GRAINED_CATEGORIES = [
    "Embedding",
    "RMSNorm_Attn",
    "QKV_Linear",
    "RoPE",
    "Attn_Compute",
    "O_Linear",
    "RMSNorm_FFN",
    "FFN_Gate_Up_Linear",
    "FFN_SiLU_Mul",
    "FFN_Down_Linear",
    "RMSNorm_Final",
    "LM_Head",
    "Sampling",
    "Tokenizer_Decode",
]

GPU_CATEGORIES = {
    "Embedding",
    "RMSNorm_Attn",
    "QKV_Linear",
    "RoPE",
    "Attn_Compute",
    "O_Linear",
    "RMSNorm_FFN",
    "FFN_Gate_Up_Linear",
    "FFN_SiLU_Mul",
    "FFN_Down_Linear",
    "RMSNorm_Final",
    "LM_Head",
    "Sampling",
}

CPU_CATEGORIES = {
    "Tokenizer_Decode",
    "Tokenizer_Encode",
}

CATEGORY_COLORS = {
    "Embedding": "#7f8c8d",          # Gray
    "RMSNorm_Attn": "#1abc9c",       # Light Teal
    "QKV_Linear": "#e74c3c",         # Red
    "RoPE": "#c0392b",               # Dark Red
    "Attn_Compute": "#ff7675",       # Salmon Red
    "O_Linear": "#d63031",           # Crimson
    "RMSNorm_FFN": "#2ecc71",        # Green
    "FFN_Gate_Up_Linear": "#f39c12", # Orange
    "FFN_SiLU_Mul": "#e67e22",       # Amber
    "FFN_Down_Linear": "#d35400",    # Rust Orange
    "RMSNorm_Final": "#27ae60",      # Dark Green
    "LM_Head": "#9b59b6",            # Purple
    "Sampling": "#3498db",           # Blue
    "Tokenizer_Decode": "#00cec9",   # Cyan
}

CATEGORY_TERMINAL_STYLES = {
    "Embedding": "dim white",
    "RMSNorm_Attn": "bold cyan",
    "QKV_Linear": "bold red",
    "RoPE": "red",
    "Attn_Compute": "bold bright_red",
    "O_Linear": "dark_red",
    "RMSNorm_FFN": "bold green",
    "FFN_Gate_Up_Linear": "bold yellow",
    "FFN_SiLU_Mul": "bold orange3",
    "FFN_Down_Linear": "bold orange_red1",
    "RMSNorm_Final": "green",
    "LM_Head": "bold magenta",
    "Sampling": "bold blue",
    "Tokenizer_Decode": "dim cyan",
}


def extract_single_token_metric(
    prof: torch.profiler.profile,
    step_idx: int,
    token_id: Optional[int] = None,
    token_text: str = "",
) -> Dict[str, Any]:
    """
    Extracts fine-grained timings from a single profiled step.
    """
    cat_times: Dict[str, float] = OrderedDict((cat, 0.0) for cat in FINE_GRAINED_CATEGORIES)

    def accumulate_child_times(event):
        if event.name in GPU_CATEGORIES:
            cat_times[event.name] += event.device_time_total / 1000.0  # us to ms
            return
        elif event.name in CPU_CATEGORIES:
            cat_times[event.name] += event.cpu_time_total / 1000.0  # us to ms
            return

        for child in event.cpu_children:
            accumulate_child_times(child)

    for event in prof.events():
        if event.name.startswith("token_") and len(event.cpu_children) > 0:
            accumulate_child_times(event)

    total_latency_ms = sum(cat_times.values())

    return {
        "step": step_idx,
        "token_id": token_id,
        "token_text": token_text,
        "total_latency_ms": total_latency_ms,
        "breakdown": cat_times,
    }


def render_terminal_dashboard(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    console: Optional[Console] = None,
):
    """
    Renders a comprehensive terminal dashboard with tables and stacked ASCII/Unicode bars.
    """
    if console is None:
        console = Console()

    if not token_records:
        console.print("[yellow][!] No token metrics found to display.[/yellow]")
        return

    sampled_count = len(token_records)
    total_time_ms = sum(r["total_latency_ms"] for r in token_records)
    avg_latency_ms = total_time_ms / sampled_count if sampled_count > 0 else 0.0

    # Aggregate total time per op across sampled steps
    total_by_op = OrderedDict((cat, sum(r["breakdown"][cat] for r in token_records)) for cat in FINE_GRAINED_CATEGORIES)

    # 1. Summary Header Panel
    header_text = Text()
    header_text.append(f"Sampled Steps: {sampled_count} checkpoints\n", style="bold white")
    header_text.append(f"Average Sampled Latency: {avg_latency_ms:.2f} ms/token\n", style="bold green")
    header_text.append("\nAggregate Op Breakdown across Sampled Steps:\n", style="bold underline")
    for op, op_ms in total_by_op.items():
        if op_ms > 0:
            pct = (op_ms / total_time_ms * 100.0) if total_time_ms > 0 else 0.0
            style = CATEGORY_TERMINAL_STYLES.get(op, "white")
            header_text.append(f"  • {op:20s}: {op_ms:7.2f} ms ({pct:5.1f}%)\n", style=style)

    console.print(Panel(header_text, title="[bold cyan]LLaMA-3.2 Fine-Grained Profiler Summary[/bold cyan]", expand=False))

    # 2. Detailed Breakdown Table
    table = Table(
        title="[bold green]Sampled Step-by-Step Fine-Grained Op Latency (ms)[/bold green]",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )

    table.add_column("Step", justify="right", style="cyan", width=6)
    table.add_column("Total", justify="right", style="bold white", width=8)
    table.add_column("QKV", justify="right", style="red", width=7)
    table.add_column("RoPE", justify="right", style="red", width=6)
    table.add_column("AttnCore", justify="right", style="bright_red", width=8)
    table.add_column("O_Proj", justify="right", style="dark_red", width=7)
    table.add_column("GateUp", justify="right", style="yellow", width=7)
    table.add_column("SiLU", justify="right", style="orange3", width=6)
    table.add_column("Down", justify="right", style="orange_red1", width=7)
    table.add_column("RMSNorms", justify="right", style="green", width=9)
    table.add_column("LM_Head", justify="right", style="magenta", width=8)
    table.add_column("Sampling", justify="right", style="blue", width=8)

    for r in token_records:
        b = r["breakdown"]
        tot = r["total_latency_ms"]
        # Sum all three RMSNorms for display brevity
        rmsnorms_total = b["RMSNorm_Attn"] + b["RMSNorm_FFN"] + b["RMSNorm_Final"]

        table.add_row(
            f"#{r['step']}",
            f"{tot:.2f}",
            f"{b['QKV_Linear']:.2f}",
            f"{b['RoPE']:.2f}",
            f"{b['Attn_Compute']:.2f}",
            f"{b['O_Linear']:.2f}",
            f"{b['FFN_Gate_Up_Linear']:.2f}",
            f"{b['FFN_SiLU_Mul']:.2f}",
            f"{b['FFN_Down_Linear']:.2f}",
            f"{rmsnorms_total:.2f}",
            f"{b['LM_Head']:.2f}",
            f"{b['Sampling']:.2f}",
        )

    console.print(table)


def generate_html_dashboard(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    output_file: str = "profile_dashboard.html",
):
    """
    Generates a zero-dependency, self-contained interactive HTML/SVG dashboard
    with 13-category fine-grained stacked bars and latency curves.
    """
    if not token_records:
        return

    steps = [r["step"] for r in token_records]
    total_latencies = [r["total_latency_ms"] for r in token_records]

    total_time = sum(total_latencies)
    avg_latency = total_time / len(token_records) if token_records else 0.0

    data_json = json.dumps({
        "prompt": prompt,
        "tokens": token_records,
        "categories": FINE_GRAINED_CATEGORIES,
        "colors": CATEGORY_COLORS,
    }, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tiny_vllm - Fine-Grained Profiler Dashboard</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --accent-color: #38bdf8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{ max-width: 1350px; margin: 0 auto; }}
        header {{ margin-bottom: 2rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1rem; }}
        h1 {{ font-size: 1.875rem; font-weight: 700; color: var(--accent-color); }}
        .subtitle {{ color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }}
        
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card-label {{ font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; }}
        .card-value {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.5rem; color: #38bdf8; }}
        
        .chart-section {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-title {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; }}
        .chart-desc {{ font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem; }}

        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem 1.25rem;
            margin-bottom: 1.5rem;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 0.5rem;
        }}
        .legend-item {{ display: flex; align-items: center; font-size: 0.8rem; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 3px; margin-right: 0.4rem; flex-shrink: 0; }}

        svg {{ width: 100%; height: auto; overflow: visible; }}
        .chart-svg text {{ font-family: monospace; font-size: 11px; fill: var(--text-muted); }}
        .grid-line {{ stroke: var(--border-color); stroke-dasharray: 4; stroke-width: 0.8; }}
        .bar-segment {{ transition: opacity 0.15s ease; cursor: pointer; }}
        .bar-segment:hover {{ opacity: 0.85; }}

        .tooltip {{
            position: fixed;
            display: none;
            background: #0f172a;
            border: 1px solid #38bdf8;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            color: #f8fafc;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }}
        th, td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border-color); }}
        th {{ background: rgba(15, 23, 42, 0.8); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; }}
        tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
    </style>
</head>
<body>
    <div id="tooltip" class="tooltip"></div>
    <div class="container">
        <header>
            <h1>⚡ tiny_vllm - Fine-Grained Latency Composition</h1>
            <div class="subtitle">Sampled Profiling (Step 0, every 100 steps, and last step) | LLaMA-3.2-1B Architecture</div>
        </header>

        <div class="grid-stats">
            <div class="card">
                <div class="card-label">Sampled Steps</div>
                <div class="card-value">{len(token_records)}</div>
            </div>
            <div class="card">
                <div class="card-label">Avg Sampled Latency</div>
                <div class="card-value">{avg_latency:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
            </div>
            <div class="card">
                <div class="card-label">Min Latency (Initial Step)</div>
                <div class="card-value">{min(total_latencies):.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
            </div>
            <div class="card">
                <div class="card-label">Max Latency (Final Step)</div>
                <div class="card-value">{max(total_latencies):.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
            </div>
        </div>

        <!-- 1. STACKED BAR CHART -->
        <div class="chart-section">
            <div class="chart-title">1. Fine-Grained Op-Time Composition per Sampled Step</div>
            <div class="chart-desc">
                Stacked breakdown of time spent in QKV Linear, RoPE, Attention Compute, O Linear, SwiGLU Gate/Up/Down, RMSNorms, LM Head, and Sampling.
            </div>
            <div class="legend" id="op-legend"></div>
            <div id="stacked-bar-container"></div>
        </div>

        <!-- 2. LATENCY TREND CURVE -->
        <div class="chart-section">
            <div class="chart-title">2. Latency Scaling Curve across Sequence Length</div>
            <div class="chart-desc">
                Visualizes how per-token latency scales without a KV cache as sequence length grows.
            </div>
            <div id="line-chart-container"></div>
        </div>

        <!-- 3. DATA TABLE -->
        <div class="chart-section">
            <div class="chart-title">3. Detailed Numerical Breakdown Table</div>
            <div style="overflow-x: auto;">
                <table id="metrics-table">
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Total (ms)</th>
                            <th>QKV Proj</th>
                            <th>RoPE</th>
                            <th>Attn Core</th>
                            <th>O Proj</th>
                            <th>Gate/Up</th>
                            <th>SiLU</th>
                            <th>Down</th>
                            <th>RMSNorms</th>
                            <th>LM Head</th>
                            <th>Sampling</th>
                        </tr>
                    </thead>
                    <tbody id="table-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const profileData = {data_json};

        // Populate Legend
        const legendContainer = document.getElementById('op-legend');
        profileData.categories.forEach(cat => {{
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<span class="legend-color" style="background:${{profileData.colors[cat]}};"></span><strong>${{cat}}</strong>`;
            legendContainer.appendChild(item);
        }});

        // Tooltip
        const tooltip = document.getElementById('tooltip');
        function showTooltip(e, html) {{
            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY + 15) + 'px';
        }}
        function hideTooltip() {{
            tooltip.style.display = 'none';
        }}

        // Render Stacked Bar Chart
        function renderStackedBarChart() {{
            const container = document.getElementById('stacked-bar-container');
            const data = profileData.tokens;
            const w = 1200, h = 380, padL = 70, padR = 20, padT = 20, padB = 40;
            const chartW = w - padL - padR;
            const chartH = h - padT - padB;

            const maxVal = Math.max(...data.map(d => d.total_latency_ms)) * 1.15 || 1;
            const barW = Math.max(12, Math.min(45, (chartW / data.length) * 0.65));
            const stepW = chartW / data.length;

            let svg = `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart-svg">`;

            // Grid lines
            for (let i = 0; i <= 5; i++) {{
                const yVal = (maxVal / 5) * i;
                const yPos = padT + chartH - (chartH / 5) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(1)}} ms</text>`;
            }}

            data.forEach((d, idx) => {{
                const x = padL + idx * stepW + (stepW - barW) / 2;
                let currentBottom = padT + chartH;

                profileData.categories.forEach(cat => {{
                    const val = d.breakdown[cat] || 0;
                    if (val <= 0.001) return;
                    const barH = (val / maxVal) * chartH;
                    const y = currentBottom - barH;
                    const color = profileData.colors[cat] || '#888';

                    svg += `<rect class="bar-segment" x="${{x}}" y="${{y}}" width="${{barW}}" height="${{barH}}" fill="${{color}}"
                        onmousemove="showTooltip(event, '<strong>Step ${{d.step}} (${{d.token_text || ''}})</strong><br/>${{cat}}: ${{val.toFixed(2)}} ms (${{((val/d.total_latency_ms)*100).toFixed(1)}}%)<br/>Total: ${{d.total_latency_ms.toFixed(2)}} ms')"
                        onmouseleave="hideTooltip()" />`;

                    currentBottom = y;
                }});

                svg += `<text x="${{x + barW/2}}" y="${{padT + chartH + 18}}" text-anchor="middle">#${{d.step}}</text>`;
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        // Render Latency Curve
        function renderLineChart() {{
            const container = document.getElementById('line-chart-container');
            const data = profileData.tokens;
            const w = 1200, h = 260, padL = 70, padR = 20, padT = 20, padB = 40;
            const chartW = w - padL - padR;
            const chartH = h - padT - padB;

            const maxVal = Math.max(...data.map(d => d.total_latency_ms)) * 1.15 || 1;
            const stepW = chartW / (data.length > 1 ? (data.length - 1) : 1);

            let points = [];
            data.forEach((d, idx) => {{
                const x = padL + idx * stepW;
                const y = padT + chartH - (d.total_latency_ms / maxVal) * chartH;
                points.push(`${{x}},${{y}}`);
            }});

            let svg = `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart-svg">`;

            for (let i = 0; i <= 4; i++) {{
                const yVal = (maxVal / 4) * i;
                const yPos = padT + chartH - (chartH / 4) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(1)}} ms</text>`;
            }}

            svg += `<polyline points="${{points.join(' ')}}" fill="none" stroke="#38bdf8" stroke-width="2.5" />`;

            data.forEach((d, idx) => {{
                const x = padL + idx * stepW;
                const y = padT + chartH - (d.total_latency_ms / maxVal) * chartH;
                svg += `<circle cx="${{x}}" cy="${{y}}" r="5" fill="#0f172a" stroke="#38bdf8" stroke-width="2" style="cursor:pointer;"
                    onmousemove="showTooltip(event, '<strong>Step ${{d.step}}</strong><br/>Latency: ${{d.total_latency_ms.toFixed(2)}} ms')"
                    onmouseleave="hideTooltip()" />`;
                svg += `<text x="${{x}}" y="${{padT + chartH + 18}}" text-anchor="middle">#${{d.step}}</text>`;
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        // Render Table Body
        function renderTable() {{
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            profileData.tokens.forEach(d => {{
                const b = d.breakdown;
                const rmsnorms = b.RMSNorm_Attn + b.RMSNorm_FFN + b.RMSNorm_Final;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${{d.step}}</strong></td>
                    <td><strong>${{d.total_latency_ms.toFixed(2)}}</strong></td>
                    <td style="color:${{profileData.colors.QKV_Linear}}">${{b.QKV_Linear.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.RoPE}}">${{b.RoPE.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.Attn_Compute}}">${{b.Attn_Compute.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.O_Linear}}">${{b.O_Linear.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.FFN_Gate_Up_Linear}}">${{b.FFN_Gate_Up_Linear.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.FFN_SiLU_Mul}}">${{b.FFN_SiLU_Mul.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.FFN_Down_Linear}}">${{b.FFN_Down_Linear.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.RMSNorm_Attn}}">${{rmsnorms.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.LM_Head}}">${{b.LM_Head.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.Sampling}}">${{b.Sampling.toFixed(2)}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        renderStackedBarChart();
        renderLineChart();
        renderTable();
    </script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[*] Standalone HTML profile dashboard generated at: {output_file}")


def save_json_metrics(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    output_file: str = "token_metrics.json",
):
    """
    Saves token metrics as a structured JSON file for downstream analysis.
    """
    data = {
        "prompt": prompt,
        "tokens": token_records,
        "summary": {
            "sampled_steps": len(token_records),
            "total_latency_ms": sum(r["total_latency_ms"] for r in token_records),
            "avg_latency_ms": (
                sum(r["total_latency_ms"] for r in token_records) / len(token_records)
                if token_records else 0.0
            ),
        }
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[*] Token metrics JSON saved at: {output_file}")
