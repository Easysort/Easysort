# easysort/viewer.py
import os
from typing import List
from flask import Flask, request, render_template_string, send_file, abort

class ImageViewer:
    def __init__(self, image_paths: List[str], title: str = "Image Viewer"):
        # sanitize: strip, absolutize, keep only existing image files
        allowed = (".jpg", ".jpeg", ".png", ".webp", ".gif")
        cleaned = []
        for p in image_paths:
            if not p:
                continue
            p = p.strip()
            if not p.lower().endswith(allowed):
                continue
            ap = p if os.path.isabs(p) else os.path.abspath(p)
            if os.path.isfile(ap):
                cleaned.append(ap)
        self.paths = cleaned
        self.title = title

        self.app = Flask(__name__)
        self._bind_routes()
        host = "0.0.0.0"
        debug = False
        port = 8000
        print(f"Serving {len(self.paths)} images at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

    def _bind_routes(self):
        app = self.app
        viewer = self

        @app.route("/")
        def index():
            # 8 x N grid; default to 1000 per page
            page = max(int(request.args.get("page", 1)), 1)
            per_page = max(int(request.args.get("per_page", 1000)), 1)
            total = len(viewer.paths)
            pages = max((total + per_page - 1) // per_page, 1)
            page = min(page, pages)
            start = (page - 1) * per_page
            end = min(start + per_page, total)
            items = list(range(start, end))  # indices to render

            html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    .top { margin-bottom: 12px; color: #333; }
    .grid {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 12px;
    }
    .card { border: 1px solid #ddd; border-radius: 6px; padding: 6px; background: #fff; }
    .thumb { width: 100%; height: auto; display: block; }
    .name { font-size: 12px; color: #444; margin-top: 6px; word-break: break-all; }
    .pager { margin-top: 14px; }
    a { text-decoration: none; color: #0366d6; }
    input { padding: 4px 6px; }
    .controls { display: flex; gap: 8px; align-items: center; }
  </style>
</head>
<body>
  <div class="top">
    <div class="controls">
      <div><strong>{{ title }}</strong></div>
      <div>Showing {{ start+1 if total>0 else 0 }}–{{ end }} of {{ total }} (page {{ page }}/{{ pages }})</div>
      <form method="get">
        <label>Per page:
          <input type="number" name="per_page" value="{{ per_page }}" min="1" style="width:80px" />
        </label>
        <label>Page:
          <input type="number" name="page" value="{{ page }}" min="1" style="width:80px" />
        </label>
        <button type="submit">Go</button>
      </form>
    </div>
  </div>

  <div class="grid">
    {% for idx in items %}
      <div class="card">
        <a href="/img/{{ idx }}" target="_blank">
          <img class="thumb" src="/img/{{ idx }}" loading="lazy" />
        </a>
        <div class="name">{{ names[idx] }}</div>
      </div>
    {% endfor %}
  </div>

  <div class="pager">
    {% if page>1 %}
      <a href="?page={{ page-1 }}&per_page={{ per_page }}">← Prev</a>
    {% endif %}
    {% if page<pages %}
      {% if page>1 %}&nbsp;|&nbsp;{% endif %}
      <a href="?page={{ page+1 }}&per_page={{ per_page }}">Next →</a>
    {% endif %}
  </div>
</body>
</html>
"""
            names = [os.path.basename(p) for p in viewer.paths]
            return render_template_string(
                html,
                title=viewer.title,
                items=items,
                names=names,
                page=page,
                pages=pages,
                per_page=per_page,
                total=total,
                start=start,
                end=end,
            )

        @app.route("/img/<int:idx>")
        def img(idx: int):
            if idx < 0 or idx >= len(viewer.paths):
                return abort(404)
            path = viewer.paths[idx]
            if not os.path.isfile(path):
                return abort(404)
            return send_file(path)