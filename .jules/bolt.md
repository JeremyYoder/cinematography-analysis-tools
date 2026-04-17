## 2024-04-17 - Fast Hierarchy Tie-Breaking
**Learning:** In PyTorch inference loops, custom tie-breaking logic based on categorical hierarchy (like using dict.get() inside max() key function) can cause substantial overhead.
**Action:** Pre-calculate the fallback values into an array and use index-based comparison (e.g., `max(range(len(classes)), key=lambda i: (probs[i], precalculated_hierarchy[i]))`) for significantly faster execution while maintaining deterministic results.
