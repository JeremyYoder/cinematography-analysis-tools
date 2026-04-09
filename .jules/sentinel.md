## 2024-05-24 - Secure Temporary Directory Handling
**Vulnerability:** Arbitrary file deletion and symlink/race condition attacks (CWE-377/CWE-379) via hardcoded temporary directory creation (`train`) in `get-heatmaps.py`.
**Learning:** When creating temporary structures for internal processing (like formatting data for fastai's `ImageDataBunch`), never use predictable, hardcoded directory names within user-supplied paths, as subsequent `shutil.rmtree` could inadvertently delete pre-existing user data with the same name.
**Prevention:** Dynamically provision secure temporary directories using `tempfile.mkdtemp` and wrap file manipulations in `try...finally` blocks to guarantee safe restoration of user state and deterministic cleanup, irrespective of execution success or failure.
