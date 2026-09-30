import json

def get_attn(path):
    try:
        with open(path) as f:
            data = json.load(f)
        tokens = data.get('tokens', [])
        print(f"--- {path.split('/')[-2]} ---")
        for step in [250, 500, 750, 1000, 2047]:
            for t in tokens:
                if t.get('step') == step and t.get('breakdown'):
                    b = t['breakdown']
                    if 'Attn_Compute' in b:
                        print(f"Step {step}: {b['Attn_Compute']:.4f} ms")
    except Exception as e:
        print(e)

get_attn("../chapter_2_kvcache/profile_results/token_metrics.json")
get_attn("profile_results_flash_decode/token_metrics.json")
