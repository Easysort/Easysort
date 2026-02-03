"""
Web-based image validator with Registry integration.
Subclass `Validator` and override `categories`, `ground_truth_type`, and optionally `prediction_type`.
"""
from dataclasses import dataclass, field, fields, make_dataclass
from pathlib import Path
from typing import Optional, Type, List
from flask import Flask, request, jsonify, Response
from easysort.registry import RegistryBase, RegistryConnector
from easysort.helpers import current_timestamp, REGISTRY_LOCAL_IP
import base64, io

# Default ground truth dataclass - subclass can define their own
BaseGroundTruth = make_dataclass("BaseGroundTruth", [
    ("label", str), ("accepted_prediction", bool),
    ("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA)
], bases=(RegistryBase.BaseDefaultTypes.BASECLASS,))


class Validator:
    categories: List[str] = ["good", "bad"]  # Override in subclass
    ground_truth_type: Type = BaseGroundTruth  # Override in subclass
    prediction_type: Optional[Type] = None  # Set to show AI predictions
    prefix: str = ""  # Registry prefix to search
    suffixes: tuple = (".jpg", ".png", ".jpeg")
    bundle_size: int = 1  # 1-6 images shown together
    port: int = 8080

    def __init__(self, registry: RegistryBase):
        self.registry, self.app = registry, Flask(__name__)
        self._idx, self._files = 0, []
        self._setup_routes()

    def get_unlabeled(self) -> List[Path]:
        files = self.registry.LIST(self.prefix)
        return [f for f in files if f.suffix in self.suffixes and not self.registry.EXISTS(f, self.ground_truth_type)]

    def get_prediction_label(self, key: Path) -> Optional[str]:
        """Override to extract label from prediction. Return None if no prediction."""
        if not self.prediction_type: return None
        pred = self.registry.GET(key, self.prediction_type, throw_error=False)
        return getattr(pred, 'label', None) if pred else None

    def _image_b64(self, key: Path) -> str:
        data = self.registry.GET(key, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        buf = io.BytesIO(); data.save(buf, format="JPEG"); return base64.b64encode(buf.getvalue()).decode()

    def _save(self, keys: List[Path], label: str, accepted: bool):
        for key in keys:
            gt = self.ground_truth_type(
                id=self.registry.get_id(self.ground_truth_type), label=label, accepted_prediction=accepted,
                metadata=RegistryBase.BaseDefaultTypes.BASEMETADATA(model="human", created_at=current_timestamp()))
            self.registry.POST(key, gt, self.ground_truth_type, overwrite=True)

    def _setup_routes(self):
        @self.app.route("/")
        def index(): return HTML.replace("{{CATEGORIES}}", str(self.categories)).replace("{{BUNDLE}}", str(self.bundle_size))

        @self.app.route("/next")
        def next_item():
            if not self._files: self._files = self.get_unlabeled()
            if self._idx >= len(self._files): return jsonify({"done": True, "total": len(self._files)})
            bundle = self._files[self._idx:self._idx + self.bundle_size]
            images = [{"key": str(k), "b64": self._image_b64(k), "prediction": self.get_prediction_label(k)} for k in bundle]
            return jsonify({"done": False, "images": images, "idx": self._idx, "total": len(self._files)})

        @self.app.route("/label", methods=["POST"])
        def label():
            data = request.json
            keys = [Path(k) for k in data["keys"]]
            self._save(keys, data["label"], data.get("accepted", False))
            self._idx += len(keys)
            return jsonify({"ok": True})

        @self.app.route("/skip", methods=["POST"])
        def skip():
            self._idx += self.bundle_size
            return jsonify({"ok": True})

        @self.app.route("/back", methods=["POST"])
        def back():
            self._idx = max(0, self._idx - self.bundle_size)
            return jsonify({"ok": True})

    def run(self, host="0.0.0.0"):
        print(f"Validator running at http://{host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=False, threaded=True)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Validator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#1a1a1a;color:#fff;height:100vh;display:flex;flex-direction:column}
#images{flex:1;display:flex;gap:8px;padding:8px;justify-content:center;align-items:center;flex-wrap:wrap}
#images img{max-height:80vh;max-width:45%;object-fit:contain;border-radius:4px}
#bar{padding:12px;background:#222;display:flex;gap:16px;align-items:center;justify-content:center}
.btn{padding:8px 16px;border:none;border-radius:4px;cursor:pointer;font-size:14px;font-weight:500}
.cat{background:#4a4a4a;color:#fff}.cat:hover{background:#5a5a5a}
.accept{background:#2d7d46;color:#fff}.skip{background:#666}.back{background:#555}
#progress{color:#888;margin-left:auto}
#pred{background:#3a3a6a;padding:4px 12px;border-radius:4px;display:none}
#done{display:none;font-size:24px;text-align:center;padding:48px}
kbd{background:#333;padding:2px 6px;border-radius:3px;font-size:12px;margin-right:4px}
</style></head><body>
<div id="images"></div>
<div id="bar">
  <button class="btn back" onclick="back()"><kbd>B</kbd>Back</button>
  <span id="cats"></span>
  <span id="pred"><kbd>A</kbd>Accept: <span id="pred-label"></span></span>
  <button class="btn skip" onclick="skip()"><kbd>S</kbd>Skip</button>
  <span id="progress">-/-</span>
</div>
<div id="done">✓ All done!</div>
<script>
const CATS={{CATEGORIES}}, BUNDLE={{BUNDLE}};
let currentKeys=[], currentPred=null;

function render(data){
  if(data.done){document.getElementById('done').style.display='block';document.getElementById('bar').style.display='none';return}
  document.getElementById('images').innerHTML=data.images.map(i=>`<img src="data:image/jpeg;base64,${i.b64}">`).join('');
  currentKeys=data.images.map(i=>i.key);
  currentPred=data.images[0]?.prediction;
  document.getElementById('pred').style.display=currentPred?'inline':'none';
  document.getElementById('pred-label').textContent=currentPred||'';
  document.getElementById('progress').textContent=`${data.idx+1}/${data.total}`;
}

async function load(){render(await(await fetch('/next')).json())}
async function label(l,accepted=false){await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:currentKeys,label:l,accepted})});load()}
async function skip(){await fetch('/skip',{method:'POST'});load()}
async function back(){await fetch('/back',{method:'POST'});load()}

document.getElementById('cats').innerHTML=CATS.map((c,i)=>`<button class="btn cat" onclick="label('${c}')"><kbd>${i+1}</kbd>${c}</button>`).join('');
document.addEventListener('keydown',e=>{
  if(e.key>='1'&&e.key<='9'&&CATS[+e.key-1])label(CATS[+e.key-1]);
  else if(e.key.toLowerCase()==='a'&&currentPred)label(currentPred,true);
  else if(e.key.toLowerCase()==='s')skip();
  else if(e.key.toLowerCase()==='b')back();
});
load();
</script></body></html>"""


if __name__ == "__main__":
    # Example: Validator for classifying images as "person" / "empty" / "unclear"
    # with optional AI predictions from a custom result type
    
    ExampleGroundTruth = make_dataclass("ExampleGroundTruth", [
        ("label", str), ("accepted_prediction", bool),
        ("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA)
    ], bases=(RegistryBase.BaseDefaultTypes.BASECLASS,))

    class ExampleValidator(Validator):
        categories = ["person", "empty", "unclear"]
        ground_truth_type = ExampleGroundTruth
        prediction_type = None  # Set to your result dataclass to show predictions
        prefix = "verdis/gadstrup"
        bundle_size = 1

        def get_prediction_label(self, key: Path) -> Optional[str]:
            # Override to extract label from your prediction dataclass
            # Example: return pred.category if pred else None
            return None

    registry = RegistryBase(RegistryConnector(REGISTRY_LOCAL_IP))
    validator = ExampleValidator(registry)
    validator.run()
