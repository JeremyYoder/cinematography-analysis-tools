## 2025-04-10 - CLI Developer Experience
**Learning:** For command-line tools processing multiple files (like `get-preds.py`), printing a raw dump of files (e.g. `print(files)`) creates poor Developer Experience and pollutes the console, especially with large datasets. Additionally, lacking progress indicators during long synchronous operations makes the process opaque.
**Action:** Replace raw file list dumps with summary counts (`Found X images`), ensure empty states are handled gracefully with early returns, and use simple `enumerate` loops to print clear progress indicators (`Processing idx/total`).

## 2025-04-14 - Keyboard Accessibility for Custom Drop Zones
**Learning:** Custom `<div>`-based interactive elements (like drag-and-drop zones) are entirely invisible to keyboard and screen reader users by default. Without `tabindex="0"`, `role="button"`, and explicit `keydown` event listeners for 'Enter' and 'Space', keyboard-only users cannot access the core functionality of the application.
**Action:** Always test custom interactive elements with a keyboard to ensure they can be focused (`:focus-visible` outline) and triggered. If using a `<div>` as a button or file input trigger, add `role="button"`, `tabindex="0"`, an appropriate `aria-label`, and a keyboard event listener.
