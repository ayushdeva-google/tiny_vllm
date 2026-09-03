"""
Profile Visualizer for tiny_vllm.

Extracts token-wise latency and operator time composition (Tokenizer, RMSNorm, Attention,
FFN, LM Head, Sampling) from torch.profiler traces.

Capabilities:
1. Programmatic metric extraction from torch.profiler.profile events.
2. Rich terminal dashboard with formatted tables, statistics, and colored stacked bars.
3. Zero-dependency, standalone interactive HTML/SVG dashboard (100% offline).
4. Structured JSON metrics export.
5. Optional Matplotlib export if installed.
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


# High-level categories tracked in the visualizer
GPU_CATEGORIES = {"Embedding", "RMSNorm", "Attention", "FFN", "LM_Head", "Sampling"}
CPU_CATEGORIES = {"Tokenizer_Decode", "Tokenizer_Encode"}

CATEGORY_COLORS = {
    "Attention": "#e74c3c",       # Red
    "FFN": "#f39c12",             # Orange
    "RMSNorm": "#2ecc71",         # Green
    "LM_Head": "#9b59b6",         # Purple
    "Embedding": "#95a5a6",       # Gray
    "Sampling": "#3498db",        # Blue
    "Tokenizer_Decode": "#1abc9c",# Teal
    "Tokenizer_Encode": "#16a085",# Dark Teal
}

CATEGORY_TERMINAL_STYLES = {
    "Attention": "bold red",
    "FFN": "bold yellow",
    "RMSNorm": "bold green",
    "LM_Head": "bold magenta",
    "Embedding": "dim white",
    "Sampling": "bold cyan",
    "Tokenizer_Decode": "bold blue",
    "Tokenizer_Encode": "dim blue",
}


def extract_token_metrics(
    prof: torch.profiler.profile,
    token_details: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Traverses the profiler event tree and aggregates operator execution times
    under each top-level 'token_{step}' event.

    Args:
        prof: The active or completed PyTorch Profiler instance.
        token_details: Optional metadata list containing [{'step': int, 'id': int, 'text': str}, ...]

    Returns:
        A list of token metric dictionaries containing step index, total latency,
        and per-op time composition in milliseconds.
    """
    token_records = []

    def accumulate_child_times(event, cat_times: Dict[str, float]):
        # Match high-level categories
        if event.name in GPU_CATEGORIES:
            cat_times[event.name] += event.device_time_total / 1000.0  # microseconds to ms
            return  # Stop recursion: event.device_time_total already includes child CUDA kernels
        elif event.name in CPU_CATEGORIES:
            cat_times[event.name] += event.cpu_time_total / 1000.0  # microseconds to ms
            return

        for child in event.cpu_children:
            accumulate_child_times(child, cat_times)

    # Walk through top-level events with cpu_children
    for event in prof.events():
        if event.name.startswith("token_") and len(event.cpu_children) > 0:
            try:
                step_idx = int(event.name.split("_")[1])
            except (ValueError, IndexError):
                continue

            cat_times: Dict[str, float] = OrderedDict([
                ("Attention", 0.0),
                ("FFN", 0.0),
                ("RMSNorm", 0.0),
                ("LM_Head", 0.0),
                ("Sampling", 0.0),
                ("Embedding", 0.0),
                ("Tokenizer_Decode", 0.0),
            ])

            accumulate_child_times(event, cat_times)

            total_latency_ms = sum(cat_times.values())
            # Fallback if device_time on child was zero or missed
            if total_latency_ms == 0.0:
                total_latency_ms = max(event.device_time_total, event.cpu_time_total) / 1000.0

            # Attach token detail if available
            tok_text = ""
            tok_id = None
            if token_details and step_idx < len(token_details):
                tok_text = token_details[step_idx].get("text", "")
                tok_id = token_details[step_idx].get("id", None)

            token_records.append({
                "step": step_idx,
                "token_id": tok_id,
                "token_text": tok_text,
                "total_latency_ms": total_latency_ms,
                "breakdown": cat_times,
            })

    # Sort records by step index
    token_records.sort(key=lambda r: r["step"])
    return token_records


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

    total_tokens = len(token_records)
    total_time_ms = sum(r["total_latency_ms"] for r in token_records)
    avg_latency_ms = total_time_ms / total_tokens if total_tokens > 0 else 0.0
    throughput = (total_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    # Aggregate total time per op
    total_by_op = OrderedDict([
        ("Attention", sum(r["breakdown"]["Attention"] for r in token_records)),
        ("FFN", sum(r["breakdown"]["FFN"] for r in token_records)),
        ("RMSNorm", sum(r["breakdown"]["RMSNorm"] for r in token_records)),
        ("LM_Head", sum(r["breakdown"]["LM_Head"] for r in token_records)),
        ("Sampling", sum(r["breakdown"]["Sampling"] for r in token_records)),
        ("Embedding", sum(r["breakdown"]["Embedding"] for r in token_records)),
        ("Tokenizer_Decode", sum(r["breakdown"]["Tokenizer_Decode"] for r in token_records)),
    ])

    # 1. Summary Header Panel
    header_text = Text()
    header_text.append(f"Generated Tokens: {total_tokens} tokens\n", style="bold white")
    header_text.append(f"Total Decode Latency: {total_time_ms:.2f} ms\n", style="cyan")
    header_text.append(f"Mean Token Latency (TPOT): {avg_latency_ms:.2f} ms/token\n", style="bold green")
    header_text.append(f"Generation Throughput: {throughput:.1f} tokens/sec\n", style="bold yellow")
    header_text.append("\nGlobal Op Breakdown:\n", style="bold underline")
    for op, op_ms in total_by_op.items():
        pct = (op_ms / total_time_ms * 100.0) if total_time_ms > 0 else 0.0
        style = CATEGORY_TERMINAL_STYLES.get(op, "white")
        header_text.append(f"  • {op:16s}: {op_ms:7.2f} ms ({pct:5.1f}%)\n", style=style)

    console.print(Panel(header_text, title="[bold cyan]LLaMA-3.2 Profiler Summary[/bold cyan]", expand=False))

    # 2. Per-Token Breakdown Table
    table = Table(
        title="[bold green]Token-wise Latency & Op-Time Composition Table[/bold green]",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )

    table.add_column("Step", justify="right", style="cyan", width=5)
    table.add_column("Token", justify="left", style="white", max_width=15)
    table.add_column("Total (ms)", justify="right", style="bold white", width=10)
    table.add_column("Attention", justify="right", style="bold red", width=10)
    table.add_column("FFN", justify="right", style="bold yellow", width=10)
    table.add_column("RMSNorm", justify="right", style="bold green", width=9)
    table.add_column("LM Head", justify="right", style="bold magenta", width=9)
    table.add_column("Sampling", justify="right", style="bold cyan", width=9)
    table.add_column("Tokenizer", justify="right", style="blue", width=9)
    table.add_column("Op Stack Visual Bar", justify="left", min_width=25)

    bar_width = 30
    for r in token_records:
        b = r["breakdown"]
        tot = r["total_latency_ms"]
        safe_tot = tot if tot > 0 else 1.0

        # Construct visual mini stacked bar
        bar_text = Text()
        op_chars = [
            ("Attention", "█", "red"),
            ("FFN", "█", "yellow"),
            ("RMSNorm", "█", "green"),
            ("LM_Head", "█", "magenta"),
            ("Sampling", "█", "cyan"),
            ("Tokenizer_Decode", "█", "blue"),
        ]
        accumulated_chars = 0
        for op_name, char, style in op_chars:
            chars_for_op = int(round((b[op_name] / safe_tot) * bar_width))
            if chars_for_op > 0:
                bar_text.append(char * chars_for_op, style=style)
                accumulated_chars += chars_for_op

        remaining = bar_width - accumulated_chars
        if remaining > 0:
            bar_text.append(" " * remaining)

        tok_display = repr(r["token_text"])[1:-1] if r["token_text"] else f"id:{r['token_id']}"

        table.add_row(
            str(r["step"]),
            tok_display,
            f"{tot:.2f}",
            f"{b['Attention']:.2f}",
            f"{b['FFN']:.2f}",
            f"{b['RMSNorm']:.2f}",
            f"{b['LM_Head']:.2f}",
            f"{b['Sampling']:.2f}",
            f"{b['Tokenizer_Decode']:.2f}",
            bar_text,
        )

    console.print(table)


def generate_html_dashboard(
    token_records: List[Dict[str, Any]],
    prompt: str = "",
    output_file: str = "profile_dashboard.html",
):
    """
    Generates a zero-dependency, self-contained, interactive HTML/SVG dashboard.
    Works 100% offline with interactive hover tooltips, latency curves,
    and stacked bar charts.
    """
    if not token_records:
        return

    steps = [r["step"] for r in token_records]
    total_latencies = [r["total_latency_ms"] for r in token_records]
    categories = ["Attention", "FFN", "RMSNorm", "LM_Head", "Sampling", "Embedding", "Tokenizer_Decode"]

    num_tokens = len(steps)
    total_time = sum(total_latencies)
    avg_latency = total_time / num_tokens if num_tokens else 0.0

    # JSON-encoded data for client-side interactivity
    data_json = json.dumps({
        "prompt": prompt,
        "tokens": token_records,
        "categories": categories,
        "colors": CATEGORY_COLORS,
    }, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tiny_vllm - Profiler Dashboard</title>
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
        .container {{ max-width: 1300px; margin: 0 auto; }}
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
            gap: 1rem;
            margin-bottom: 1.5rem;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 0.5rem;
        }}
        .legend-item {{ display: flex; align-items: center; font-size: 0.85rem; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 3px; margin-right: 0.5rem; }}

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
            font-size: 0.9rem;
            text-align: left;
        }}
        th, td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); }}
        th {{ background: rgba(15, 23, 42, 0.8); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
        tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
        .token-tag {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div id="tooltip" class="tooltip"></div>
    <div class="container">
        <header>
            <h1>⚡ tiny_vllm - Profiler & Latency Composition</h1>
            <div class="subtitle">Architecture: Llama-3.2-1B-Instruct (16 Layers, GQA, SwiGLU, RMSNorm) | Autoregressive No-KV-Cache Loop</div>
        </header>

        <div class="grid-stats">
            <div class="card">
                <div class="card-label">Generated Tokens</div>
                <div class="card-value">{num_tokens}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Generation Time</div>
                <div class="card-value">{total_time:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
            </div>
            <div class="card">
                <div class="card-label">Avg Token Latency (TPOT)</div>
                <div class="card-value">{avg_latency:.2f} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
            </div>
            <div class="card">
                <div class="card-label">Throughput</div>
                <div class="card-value">{(num_tokens / (total_time / 1000.0) if total_time > 0 else 0):.1f} <span style="font-size:1rem;color:#94a3b8;">tok/s</span></div>
            </div>
        </div>

        <!-- 1. STACKED BAR CHART -->
        <div class="chart-section">
            <div class="chart-title">1. Per-Token Op-Time Composition (Stacked Bar Chart)</div>
            <div class="chart-desc">
                Shows exact time spent (ms) inside each component for each generated token step.
                Notice how without a KV cache, Attention and FFN times increase as sequence length grows!
            </div>
            <div class="legend" id="op-legend"></div>
            <div id="stacked-bar-container"></div>
        </div>

        <!-- 2. PER-TOKEN LATENCY CURVE -->
        <div class="chart-section">
            <div class="chart-title">2. Per-Token Latency Trend (ms vs Token Step)</div>
            <div class="chart-desc">
                Step-by-step latency curve illustrating autoregressive decode scaling.
            </div>
            <div id="line-chart-container"></div>
        </div>

        <!-- 3. DETAILED DATA TABLE -->
        <div class="chart-section">
            <div class="chart-title">3. Detailed Execution Metrics Table</div>
            <div style="overflow-x: auto;">
                <table id="metrics-table">
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Token ID</th>
                            <th>Decoded Text</th>
                            <th>Total (ms)</th>
                            <th>Attention</th>
                            <th>FFN</th>
                            <th>RMSNorm</th>
                            <th>LM Head</th>
                            <th>Sampling</th>
                            <th>Tokenizer</th>
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

        // Tooltip handling
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

        // Render Stacked Bar Chart (Inline SVG)
        function renderStackedBarChart() {{
            const container = document.getElementById('stacked-bar-container');
            const data = profileData.tokens;
            const w = 1150, h = 340, padL = 60, padR = 20, padT = 20, padB = 40;
            const chartW = w - padL - padR;
            const chartH = h - padT - padB;

            const maxVal = Math.max(...data.map(d => d.total_latency_ms)) * 1.15 || 1;
            const barW = Math.max(8, Math.min(40, (chartW / data.length) * 0.7));
            const stepW = chartW / data.length;

            let svg = `<svg viewBox="0 0 ${{w}} ${{h}}" class="chart-svg">`;

            // Grid lines & Y Axis
            for (let i = 0; i <= 5; i++) {{
                const yVal = (maxVal / 5) * i;
                const yPos = padT + chartH - (chartH / 5) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(1)}} ms</text>`;
            }}

            // Bars
            data.forEach((d, idx) => {{
                const x = padL + idx * stepW + (stepW - barW) / 2;
                let currentBottom = padT + chartH;

                profileData.categories.forEach(cat => {{
                    const val = d.breakdown[cat] || 0;
                    if (val <= 0) return;
                    const barH = (val / maxVal) * chartH;
                    const y = currentBottom - barH;
                    const color = profileData.colors[cat] || '#888';

                    svg += `<rect class="bar-segment" x="${{x}}" y="${{y}}" width="${{barW}}" height="${{barH}}" fill="${{color}}"
                        onmousemove="showTooltip(event, '<strong>Step ${{d.step}} (${{d.token_text || ''}})</strong><br/>${{cat}}: ${{val.toFixed(2)}} ms (${{((val/d.total_latency_ms)*100).toFixed(1)}}%)<br/>Total: ${{d.total_latency_ms.toFixed(2)}} ms')"
                        onmouseleave="hideTooltip()" />`;

                    currentBottom = y;
                }});

                // X-axis label
                svg += `<text x="${{x + barW/2}}" y="${{padT + chartH + 18}}" text-anchor="middle">#${{d.step}}</text>`;
            }});

            svg += `</svg>`;
            container.innerHTML = svg;
        }}

        // Render Latency Curve (Inline SVG)
        function renderLineChart() {{
            const container = document.getElementById('line-chart-container');
            const data = profileData.tokens;
            const w = 1150, h = 240, padL = 60, padR = 20, padT = 20, padB = 40;
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

            // Grid lines
            for (let i = 0; i <= 4; i++) {{
                const yVal = (maxVal / 4) * i;
                const yPos = padT + chartH - (chartH / 4) * i;
                svg += `<line x1="${{padL}}" y1="${{yPos}}" x2="${{w - padR}}" y2="${{yPos}}" class="grid-line" />`;
                svg += `<text x="${{padL - 10}}" y="${{yPos + 4}}" text-anchor="end">${{yVal.toFixed(1)}} ms</text>`;
            }}

            // Line
            svg += `<polyline points="${{points.join(' ')}}" fill="none" stroke="#38bdf8" stroke-width="2.5" />`;

            // Dots
            data.forEach((d, idx) => {{
                const x = padL + idx * stepW;
                const y = padT + chartH - (d.total_latency_ms / maxVal) * chartH;
                svg += `<circle cx="${{x}}" cy="${{y}}" r="5" fill="#0f172a" stroke="#38bdf8" stroke-width="2" style="cursor:pointer;"
                    onmousemove="showTooltip(event, '<strong>Token Step ${{d.step}}</strong><br/>Latency: ${{d.total_latency_ms.toFixed(2)}} ms<br/>Decoded: &quot;${{d.token_text || ''}}&quot;')"
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
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${{d.step}}</strong></td>
                    <td>${{d.token_id !== null ? d.token_id : '-'}}</td>
                    <td><span class="token-tag">${{d.token_text ? d.token_text : ''}}</span></td>
                    <td><strong>${{d.total_latency_ms.toFixed(2)}}</strong></td>
                    <td style="color:${{profileData.colors.Attention}}">${{b.Attention.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.FFN}}">${{b.FFN.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.RMSNorm}}">${{b.RMSNorm.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.LM_Head}}">${{b.LM_Head.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.Sampling}}">${{b.Sampling.toFixed(2)}}</td>
                    <td style="color:${{profileData.colors.Tokenizer_Decode}}">${{b.Tokenizer_Decode.toFixed(2)}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        // Initialize Charts
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
            "num_tokens": len(token_records),
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

