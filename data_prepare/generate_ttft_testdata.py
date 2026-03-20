import json
import math
import os
import random
from typing import List, Dict, Any

#
# Prompt length distribution configuration (manual).
# - Adjust SHORT_RANGE/LONG_RANGE to control short/long bucket ranges.
# - We re-fit lognormal parameters (mu/sigma) from these ranges.
#
SHORT_RANGE = (512, 1024)
MIDDLE_RANGE = (1536, 2048)
LONG_RANGE = (4096-512, 4096)
SHORT_BUCKET_PROB = 0.65
MIDDLE_BUCKET_PROB = 0.20
LONG_BUCKET_PROB = 0.15

# z where Phi(z)=0.95 for standard normal N(0,1)
_Z_95 = 1.6448536269514722

def sample_truncated_lognormal(mu: float, sigma: float, lo: int, hi: int, rng: random.Random) -> int:
    # 简单重采样截断
    for _ in range(1000):
        x = rng.lognormvariate(mu, sigma)  # >0
        xi = int(round(x))
        if lo <= xi <= hi:
            return xi
    # 极端情况下兜底 clamp
    x = rng.lognormvariate(mu, sigma)
    xi = int(round(x))
    return max(lo, min(hi, xi))


def fit_lognormal_mu_sigma_from_range(lo: int, hi: int) -> tuple[float, float]:
    """
    Fit mu/sigma of lognormal such that ln(X) hits the symmetric range bounds:
      ln(lo)  = mu - z*sigma
      ln(hi)  = mu + z*sigma
    with z = Phi^{-1}(0.95) (approx 1.64485).
    """
    assert hi > lo > 0
    ln_lo = math.log(lo)
    ln_hi = math.log(hi)
    mu = (ln_lo + ln_hi) / 2.0
    sigma = (ln_hi - ln_lo) / (2.0 * _Z_95)
    return mu, sigma

def sample_prompt_len(rng: random.Random) -> int:
    # mixture: short SHORT_BUCKET_PROB, long LONG_BUCKET_PROB
    prob = rng.random()
    if prob < SHORT_BUCKET_PROB:
        lo, hi = SHORT_RANGE
        mu, sigma = fit_lognormal_mu_sigma_from_range(lo, hi)
        return sample_truncated_lognormal(mu, sigma, lo, hi, rng)
    elif prob < SHORT_BUCKET_PROB + MIDDLE_BUCKET_PROB:
        lo, hi = MIDDLE_RANGE
        mu, sigma = fit_lognormal_mu_sigma_from_range(lo, hi)
        return sample_truncated_lognormal(mu, sigma, lo, hi, rng)
    else:
        lo, hi = LONG_RANGE
        mu, sigma = fit_lognormal_mu_sigma_from_range(lo, hi)
        return sample_truncated_lognormal(mu, sigma, lo, hi, rng)
        
def sample_max_tokens(rng: random.Random) -> int:
    # 离散分布：主要短输出 + 少量长输出
    # 8:45%, 16:30%, 32:15%, 64:10%
    r = rng.random()
    if r < 0.45:
        return 8
    if r < 0.75:
        return 16
    if r < 0.90:
        return 32
    return 64

def main():
    seed = 42
    N = 1024
    prompt_token_id = 1
    # max_model_len = 1024
    rng = random.Random(seed)
    requests: List[Dict[str, Any]] = []
    out_path = "./data/ttft_testdata_long_prompt.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    short_mu, short_sigma = fit_lognormal_mu_sigma_from_range(SHORT_RANGE[0], SHORT_RANGE[1])
    long_mu, long_sigma = fit_lognormal_mu_sigma_from_range(LONG_RANGE[0], LONG_RANGE[1])
    
    for _ in range(N):
        prompt_len = sample_prompt_len(rng)
        # 保证不超过 max_model_len（你当前 example.py 里是 1024）
        # prompt_len = prompt_len
        max_tokens = sample_max_tokens(rng)
        requests.append({
            "prompt_len": int(prompt_len),
            "max_tokens": int(max_tokens),
            "prompt_token_id": int(prompt_token_id),
        })
        
    payload = {
        "seed": seed,
        "count": N,
        "prompt_token_id_default": prompt_token_id,
        "prompt_len_distribution": {
            "type": "mixture_truncated_lognormal",
            "mixture_prob": {"short": SHORT_BUCKET_PROB, "long": LONG_BUCKET_PROB},
            "fitting_interval_quantiles_in_log_space": {"q_low": 0.05, "q_high": 0.95},
            "short_bucket": {"range": [SHORT_RANGE[0], SHORT_RANGE[1]], "lognormal_mu": short_mu, "lognormal_sigma": short_sigma},
            "long_bucket": {"range": [LONG_RANGE[0], LONG_RANGE[1]], "lognormal_mu": long_mu, "lognormal_sigma": long_sigma},
        },
        "max_tokens_distribution": {
            "type": "discrete",
            "values_prob": {"8": 0.45, "16": 0.30, "32": 0.15, "64": 0.10}
        },
        "requests": requests
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} with {N} requests.")
    
if __name__ == "__main__":
    main()