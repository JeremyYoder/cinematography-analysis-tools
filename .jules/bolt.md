## 2024-05-24 - Optimize get-preds.py iterative DataFrame creation
**Learning:** Creating pandas DataFrames within a loop and sorting them per iteration creates a bottleneck due to unnecessary object initialization and memory allocations. Python's built-in `max()` handles hierarchical sorting efficiently via custom tuple keys without creating intermediate objects.
**Action:** Before running heavy loop analytics in pandas, collect data via standard python containers, and initialize pandas DataFrames only once at the end of the loop.
