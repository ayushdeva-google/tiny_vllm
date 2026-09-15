import os
import json
import argparse

def parse_file(filename):
    if not os.path.exists(filename):
        return None
        
    with open(filename) as f:
        data = json.load(f)
    
    rmsnorm_attn_time = 0
    rmsnorm_ffn_time = 0
    rmsnorm_final_time = 0
    samples = 0
    
    for token in data.get('tokens', []):
        if 'breakdown' in token and token['breakdown']:
            b = token['breakdown']
            rmsnorm_attn_time += b.get('RMSNorm_Attn', 0)
            rmsnorm_ffn_time += b.get('RMSNorm_FFN', 0)
            rmsnorm_final_time += b.get('RMSNorm_Final', 0)
            samples += 1
            
    if samples > 0:
        return {
            "RMSNorm_Attn": rmsnorm_attn_time / samples,
            "RMSNorm_FFN": rmsnorm_ffn_time / samples,
            "RMSNorm_Final": rmsnorm_final_time / samples,
            "Total": (rmsnorm_attn_time + rmsnorm_ffn_time + rmsnorm_final_time) / samples
        }
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate RMSNorm Speedup HTML Dashboard")
    parser.add_argument("--json", type=str, default="profile_results/token_metrics.json", help="Path to chapter 3 token_metrics.json")
    args = parser.parse_args()

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chap2_json = os.path.join(os.path.dirname(script_dir), "chapter_2_kvcache", "profile_results", "token_metrics.json")
    chap3_json = args.json if os.path.isabs(args.json) else os.path.join(os.getcwd(), args.json)
    
    # Parse metrics
    c2_metrics = parse_file(chap2_json)
    c3_metrics = parse_file(chap3_json)
    
    if not c2_metrics or not c3_metrics:
        print("[!] Could not find token_metrics.json for both Chapter 2 and Chapter 3.")
        print(f"    Looked for: {chap2_json} and {chap3_json}")
        return

    # Calculate speedup
    speedup = c2_metrics["Total"] / c3_metrics["Total"] if c3_metrics["Total"] > 0 else 0
    c3_percentage = (c3_metrics["Total"] / c2_metrics["Total"] * 100) if c2_metrics["Total"] > 0 else 100

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @keyframes growWidth {{
      from {{ width: 0; }}
    }}
    .animate-bar {{
      animation: growWidth 1s ease-out forwards;
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-5">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-6 shadow-sm max-w-2xl mx-auto">
    
    <div class="flex items-center space-x-3 mb-4">
      <div class="p-2 bg-[var(--primary)] text-[var(--primary-foreground)] rounded-lg">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2v20"></path><path d="m17 5-5-3-5 3"></path><path d="m17 19-5 3-5-3"></path><path d="M2 12h20"></path><path d="m5 7-3 5 3 5"></path><path d="m19 7 3 5-3 5"></path>
        </svg>
      </div>
      <h2 class="text-xl font-bold">RMSNorm Speedup (Kernel Fusion)</h2>
    </div>
    
    <div class="grid grid-cols-2 gap-6 mb-6 items-stretch">
      <!-- Chapter 2 -->
      <div class="p-4 bg-[var(--background)] rounded-lg border border-[var(--border)] flex flex-col">
        <div class="min-h-[3.5rem] mb-2 flex items-start">
          <h3 class="font-semibold text-lg flex items-start gap-2"><span>🐢</span> <span>Native PyTorch Implementation</span></h3>
        </div>
        <ul class="text-sm font-mono space-y-1 text-[var(--muted-foreground)] flex-1">
          <li class="flex justify-between"><span>RMSNorm_Attn:</span> <span>{c2_metrics['RMSNorm_Attn']:.4f} ms</span></li>
          <li class="flex justify-between"><span>RMSNorm_FFN:</span> <span>{c2_metrics['RMSNorm_FFN']:.4f} ms</span></li>
          <li class="flex justify-between"><span>RMSNorm_Final:</span> <span>{c2_metrics['RMSNorm_Final']:.4f} ms</span></li>
        </ul>
        <div class="mt-3 pt-3 border-t border-[var(--border)] flex justify-between font-bold text-sm">
          <span>Total per step:</span>
          <span>~{c2_metrics['Total']:.4f} ms</span>
        </div>
      </div>

      <!-- Chapter 3 -->
      <div class="p-4 bg-[var(--background)] rounded-lg border border-[var(--border)] border-l-4 border-l-green-500 flex flex-col">
        <div class="min-h-[3.5rem] mb-2 flex items-start">
          <h3 class="font-semibold text-lg flex items-start gap-2"><span>🚀</span> <span>CUDA Fused Kernel</span></h3>
        </div>
        <ul class="text-sm font-mono space-y-1 text-[var(--muted-foreground)] flex-1">
          <li class="flex justify-between"><span>RMSNorm_Attn:</span> <span>{c3_metrics['RMSNorm_Attn']:.4f} ms</span></li>
          <li class="flex justify-between"><span>RMSNorm_FFN:</span> <span>{c3_metrics['RMSNorm_FFN']:.4f} ms</span></li>
          <li class="flex justify-between"><span>RMSNorm_Final:</span> <span>{"< 0.0001" if c3_metrics['RMSNorm_Final'] < 0.0001 else f"{c3_metrics['RMSNorm_Final']:.4f}"} ms</span></li>
        </ul>
        <div class="mt-3 pt-3 border-t border-[var(--border)] flex justify-between font-bold text-sm text-[var(--primary)]">
          <span>Total per step:</span>
          <span>~{c3_metrics['Total']:.4f} ms</span>
        </div>
      </div>
    </div>

    <!-- Bar Chart -->
    <div class="space-y-4 mb-6 bg-[var(--background)] p-4 rounded-lg border border-[var(--border)]">
      <div>
        <div class="flex justify-between items-end mb-1">
          <span class="font-semibold text-sm">Native PyTorch Implementation</span>
          <span class="font-mono text-sm text-[var(--muted-foreground)]">{c2_metrics['Total']:.4f} ms</span>
        </div>
        <div class="w-full bg-[var(--card)] rounded-full h-3 border border-[var(--border)] overflow-hidden">
          <div class="bg-red-400 h-full rounded-full animate-bar" style="width: 100%"></div>
        </div>
      </div>

      <div>
        <div class="flex justify-between items-end mb-1">
          <span class="font-semibold text-sm">CUDA Fused Kernel</span>
          <span class="font-mono text-sm text-[var(--primary)] font-bold">{c3_metrics['Total']:.4f} ms</span>
        </div>
        <div class="w-full bg-[var(--card)] rounded-full h-3 border border-[var(--border)] overflow-hidden">
          <div class="bg-green-500 h-full rounded-full animate-bar" style="width: {c3_percentage:.1f}%"></div>
        </div>
      </div>
      
      <div class="mt-2 text-center">
        <span class="text-lg font-black text-[var(--primary)]">{speedup:.1f}x Faster</span>
      </div>
    </div>

    <!-- Reason for Speedup -->
    <div class="p-4 bg-[var(--background)] rounded-lg border border-[var(--border)]">
      <h3 class="font-semibold mb-2">Why is it so much faster?</h3>
      <p class="text-sm text-[var(--muted-foreground)] leading-relaxed mb-3">
        The naive PyTorch implementation performs normalizations using multiple chained mathematical operations (<code class="bg-[var(--card)] px-1 rounded text-xs">pow</code>, <code class="bg-[var(--card)] px-1 rounded text-xs">mean</code>, <code class="bg-[var(--card)] px-1 rounded text-xs">add</code>, <code class="bg-[var(--card)] px-1 rounded text-xs">rsqrt</code>, <code class="bg-[var(--card)] px-1 rounded text-xs">mul</code>). Because standard PyTorch executes eagerly, it launches a completely separate CUDA kernel for each operation. This results in:
      </p>
      <ul class="text-sm text-[var(--muted-foreground)] list-disc pl-5 space-y-2 mb-3">
        <li><strong>Kernel Launch Overhead:</strong> The CPU has to instruct the GPU multiple times per token, wasting precious microseconds communicating across the PCIe bus.</li>
        <li><strong>The Memory Wall:</strong> Each intermediate step forces the GPU to write a temporary tensor back to slow VRAM, just for the next step to immediately read it back. Memory bandwidth becomes the primary bottleneck.</li>
      </ul>
      <p class="text-sm text-[var(--foreground)] font-medium leading-relaxed">
        By utilizing <span class="text-[var(--primary)]">Kernel Fusion</span>, the custom CUDA kernel reads the data from VRAM exactly once into ultra-fast on-chip SRAM (Registers and Shared Memory), computes the entire RMSNorm algorithm instantly, and writes the final result back to VRAM once.
      </p>
    </div>

  </div>
</body>
</html>
"""

    output_path = os.path.join(os.path.dirname(chap3_json), "rmsnorm_speedup_dashboard.html")
    with open(output_path, "w") as f:
        f.write(html_content)
        
    print(f"[✓] Successfully generated {output_path}")
    print(f"    - Native PyTorch RMSNorm: {c2_metrics['Total']:.4f} ms")
    print(f"    - Fused Kernel RMSNorm:   {c3_metrics['Total']:.4f} ms")
    print(f"    - Speedup:                {speedup:.1f}x")

if __name__ == "__main__":
    main()
