"""Measure drift between native and spliced AV embeddings, per modality."""
import sys, json, torch
sys.path.insert(0, '.')
from pathlib import Path
import torch.nn.functional as F

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'runs/v6_tier1_b/step_1000.pt'
VARIANT = sys.argv[2] if len(sys.argv) > 2 else 'b'
VIDEO_VIB = sys.argv[3] if len(sys.argv) > 3 else 'sink'
VIDEO_ROOT = Path.home() / 'SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all'

from av_ib.model.av_model_v6 import AVModelV6
print(f"Building: variant={VARIANT}, video_vib={VIDEO_VIB}", flush=True)
model = AVModelV6(use_lora=True, variant=VARIANT, video_vib=VIDEO_VIB)
model.eval()
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ck['trainable_state'], strict=False)

# Find splicer
splicer = None
for k,v in vars(model.qwen).items():
    if v.__class__.__name__ == 'CMIBSplicer':
        splicer = v; break
assert splicer is not None, "no splicer"

# Register an INDEPENDENT pre-hook on the same text_model that runs AFTER the
# splicer's (registration order = call order). It receives the kwargs the
# splicer already modified, so to compare we need native too. Trick: capture
# native inside the splicer by wrapping its bound hook via the handle list.
captured = {}

# The splicer stored its handle as ("text_model_pre_hook", h). We re-wrap by
# registering our own hooks: one that records BEFORE (registered first), the
# splicer's runs second (already registered), one that records AFTER (registered last).
text_model = splicer.text_model

# But splicer is already attached, so its hook is already in the queue.
# We register a pre-hook now -> it runs AFTER splicer's (later registration).
# That gives us the AFTER state + the masks the splicer left on itself.
def after_hook(module, args, kwargs):
    ie = kwargs.get('inputs_embeds')
    if ie is not None:
        captured['after'] = ie.detach().float().clone()
    # masks are cleared by splicer at end of its hook... so grab from a stash.
    return None
h_after = text_model.register_forward_pre_hook(after_hook, with_kwargs=True)

# To get BEFORE + masks, wrap the splicer's capture so it stashes them before clearing.
_orig_clear = splicer._clear_per_call_state
def stash_then_clear():
    # Called at end of splice; grab masks before they're wiped
    if splicer._audio_mask is not None:
        captured['amask'] = splicer._audio_mask.detach().clone()
        captured['vmask'] = splicer._video_mask.detach().clone()
    _orig_clear()
splicer._clear_per_call_state = stash_then_clear

# For BEFORE: wrap the splicer's _on_text_model_pre via the registered handle.
# Re-register: remove splicer's hook, add a combined one (before-capture + splice).
for entry in list(splicer._handles):
    if isinstance(entry, tuple) and entry[0] == 'text_model_pre_hook':
        entry[1].remove()
        splicer._handles.remove(entry)
_orig_splice = splicer._on_text_model_pre
def before_and_splice(module, args, kwargs):
    ie = kwargs.get('inputs_embeds')
    if ie is not None:
        captured['before'] = ie.detach().float().clone()
    return _orig_splice(module, args, kwargs)
h_splice = text_model.register_forward_pre_hook(before_and_splice, with_kwargs=True)

ANN = Path.home() / 'SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json'
recs = [r for r in json.load(open(ANN)) if r.get('question_deleted',0)==0]
recs = [r for r in recs if (VIDEO_ROOT/f"{r['video_id']}.mp4").exists()][:5]
from av_ib.data.musicavqa import render_question

for r in recs:
    captured.clear()
    vp = str(VIDEO_ROOT/f"{r['video_id']}.mp4")
    q = render_question(r['question_content'], r.get('templ_values','[]'))
    _ = model.forward_generate([vp],[vp],[q])
    if not all(k in captured for k in ('before','after','amask','vmask')):
        print(f"  {r['video_id']}: missing {[k for k in ('before','after','amask','vmask') if k not in captured]}", flush=True)
        continue
    b = captured['before'][0]; a = captured['after'][0]
    am = captured['amask'].view(-1).bool(); vm = captured['vmask'].view(-1).bool()
    # masks may be shaped (B,T) or (B,T,1); align length to b
    if am.numel() != b.size(0): am = am[:b.size(0)]
    if vm.numel() != b.size(0): vm = vm[:b.size(0)]
    def drift(mask, name):
        if mask.sum()==0: print(f"    {name}: 0 tokens"); return
        bb=b[mask]; aa=a[mask]
        cos=F.cosine_similarity(bb,aa,dim=-1).mean().item()
        rel=((aa-bb).norm(dim=-1)/(bb.norm(dim=-1)+1e-6)).mean().item()
        print(f"    {name}: n={int(mask.sum()):4d}  cos={cos:.3f}  rel_L2={rel:.3f}", flush=True)
    print(f"  {r['video_id']}  Q={q[:45]}", flush=True)
    drift(vm,'video'); drift(am,'audio')
