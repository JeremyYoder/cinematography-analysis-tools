## 2024-05-24 - CLI UX Empty States Validation
**Learning:** For CLI tools that load heavy ML models, validating inputs (like checking for an empty list of images) before triggering resource-intensive initialization functions greatly improves the Developer Experience (CLI UX) by failing fast and preventing unneeded resource allocation.
**Action:** Next time I optimize similar data-processing CLI scripts, I will ensure early validation is done before loading the heavy dependencies or model states.

## 2024-05-24 - File Extension and Path Validation in CLI UX
**Learning:** When validating paths for early exit, it's critical to check that the path actually exists using `os.path.exists` before calling `os.listdir` to avoid `FileNotFoundError` tracebacks. Additionally, when filtering by file extensions, using `f.lower().endswith(...)` prevents case-sensitivity bugs, and including a comprehensive list of supported formats ensures valid inputs aren't rejected.
**Action:** Always validate the existence of a path before reading it. Use case-insensitive checks and support broad file formats for ML data pipelines.

## 2024-05-24 - Recursive Image Checking for ML Dataset CLI UX
**Learning:** For ML applications processing image datasets, input directories often contain nested class subdirectories. Verifying directory emptiness using `os.listdir()` only checks the root level and falsely flags structured datasets as empty. Using `Path.rglob('*')` enables safe, recursive checks to correctly identify inputs without breaking expected functionality.
**Action:** When validating ML datasets before heavy loading, always perform a recursive search to account for nested directory structures.
