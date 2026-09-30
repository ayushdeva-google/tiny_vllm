import json

def get_norm(path):
    try:
        with open(path) as f:
            data = json.load(f)
        tokens = data.get('tokens', [])
        print(f"--- {path.split('/')[-2]} ---")
        for step in [250, 500, 2047]:
            for t in tokens:
                if t.get('step') == step and t.get('breakdown'):
                    b = t['breakdown']
                    if 'RMSNorm_Attn' in b:
                        print(f"Step {step} RMSNorm_Attn: {b['RMSNorm_Attn']:.4f} ms")
    except Exception as e:
        print(e)

get_norm("../chapter_2_kvcache/profile_results/token_metrics.json")
get_norm("profile_results_flash_decode/token_metrics.json")
