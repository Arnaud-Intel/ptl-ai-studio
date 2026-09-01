"""Named example diffs so the CLI/launcher can offer a ready-to-run demo
instead of requiring a real git repo or hand-typed diff."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    name: str
    description: str
    diff_text: str


SAMPLES: list[Sample] = [
    Sample(
        name="Off-by-one loop bound",
        description="A one-line fix to a loop that skips the last item.",
        diff_text="""\
diff --git a/inventory.py b/inventory.py
index 3a9f1c2..7b5e0aa 100644
--- a/inventory.py
+++ b/inventory.py
@@ -12,7 +12,7 @@ def restock_report(items):
     low_stock = []
-    for i in range(len(items) - 1):
+    for i in range(len(items)):
         item = items[i]
         if item.quantity < item.reorder_threshold:
             low_stock.append(item)
     return low_stock
""",
    ),
    Sample(
        name="Missing null check",
        description="Adds a guard clause before touching a possibly-None user object.",
        diff_text="""\
diff --git a/user_profile.py b/user_profile.py
index 8c1a44e..f2d9b31 100644
--- a/user_profile.py
+++ b/user_profile.py
@@ -20,6 +20,8 @@ def display_name(user):
-    return user.nickname or user.full_name
+    if user is None:
+        return "Guest"
+    return user.nickname or user.full_name
""",
    ),
    Sample(
        name="New retry helper",
        description="A small new function: a retry decorator for flaky network calls.",
        diff_text="""\
diff --git a/net_utils.py b/net_utils.py
new file mode 100644
index 0000000..4b825dc
--- /dev/null
+++ b/net_utils.py
@@ -0,0 +1,15 @@
+import time
+import functools
+
+
+def retry(times=3, delay=1.0):
+    def decorator(fn):
+        @functools.wraps(fn)
+        def wrapper(*args, **kwargs):
+            last_exc = None
+            for _ in range(times):
+                try:
+                    return fn(*args, **kwargs)
+                except Exception as exc:
+                    last_exc = exc
+                    time.sleep(delay)
+            raise last_exc
+        return wrapper
+    return decorator
""",
    ),
]
