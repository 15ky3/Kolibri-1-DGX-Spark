#!/usr/bin/env python3
"""Single-stream decode on a copy-heavy task (rewrite code with one change) —
the workload where n-gram speculation can help. Stdlib only."""
import json, os, time, urllib.request
base = f"http://127.0.0.1:{os.environ.get('PORT', '8895')}"
code = "\n".join(
    f"def handler_{i}(request, ctx):\n    data = request.get('payload_{i}')\n"
    f"    if data is None:\n        return {{'status': 400, 'error': 'missing payload_{i}'}}\n"
    f"    result = ctx.store.save('item_{i}', data)\n    return {{'status': 200, 'id': result}}\n"
    for i in range(14))
body = {"model": os.environ.get("SERVED_MODEL_NAME", "kolibri-1"), "max_tokens": 1500, "temperature": 0,
        "messages": [{"role": "user", "content":
                      "Gib den folgenden Python-Code vollständig wieder aus, ersetze dabei nur "
                      "'status' durch 'code'. Keine Erklärung.\n\n" + code}],
        "chat_template_kwargs": {"reasoning_effort": "none"}}
for _ in range(2):
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t0; n = r["usage"]["completion_tokens"]
    out = r["choices"][0]["message"]["content"]
ok = out.count("'code'") == 28 and "'status'" not in out
print(f"copy task: {n} tok in {dt:.1f}s = {n / dt:.1f} tok/s | correct={ok}")
