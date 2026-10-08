import json

def get_attn(path):
    with open(path) as f:
        data = json.load(f)
    print(f"File: {path}")
    for d in data.get('tokens', []):
        if d.get("breakdown"):
            for b in d["breakdown"]:
                if b["name"] == "Attn_Compute":
                    print(f"  Step {d['step']}: {b['dur_ms']} ms")
                    break

get_attn("../chapter_2_kvcache/profile_results/token_metrics.json")
get_attn("profile_results_flash_decode/token_metrics.json")
