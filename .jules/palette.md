## 2025-04-10 - CLI Developer Experience
**Learning:** For command-line tools processing multiple files (like `get-preds.py`), printing a raw dump of files (e.g. `print(files)`) creates poor Developer Experience and pollutes the console, especially with large datasets. Additionally, lacking progress indicators during long synchronous operations makes the process opaque.
**Action:** Replace raw file list dumps with summary counts (`Found X images`), ensure empty states are handled gracefully with early returns, and use simple `enumerate` loops to print clear progress indicators (`Processing idx/total`).
## 2025-04-16 - Custom Drop Zone Accessibility
**Learning:** When implementing custom interactive UI elements like `<div>`-based drop zones, mouse interactions (`click`, `drag`) are not enough. They inherently lack native keyboard accessibility, leaving screen reader and keyboard-only users unable to upload files.
**Action:** Always ensure full keyboard accessibility for custom interactive elements by adding `role="button"`, `tabindex="0"`, `:focus-visible` CSS outlines, and `keydown` event listeners for 'Enter' and 'Space' keys.
