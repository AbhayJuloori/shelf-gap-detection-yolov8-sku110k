"""CI check: a fresh gap evaluation must match the committed report."""
import json
import sys

new, old = (json.load(open(p)) for p in sys.argv[1:3])
for key in ("chosen", "test", "test_default"):
    if new[key] != old[key]:
        sys.exit(f"{key} differs:\n new {new[key]}\n old {old[key]}")
print("gap report reproduces")
