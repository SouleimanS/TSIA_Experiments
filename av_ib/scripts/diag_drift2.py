import sys, json, torch
sys.path.insert(0, '.')
from pathlib import Path
CKPT, VARIANT, VVIB = sys.argv[1], sys.argv[2], sys.argv[3]
VR = Path.home()/'SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all'
from av_ib.model.av_model_v6 import AVModelV6
m = AVModelV6(use_lora=True, variant=VARIANT, video_vib=VVIB); m.eval()
m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False)['trainable_state'], strict=False)
ANN = Path.home()/'SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json'
recs=[r for r in json.load(open(ANN)) if r.get('question_deleted',0)==0]
recs=[r for r in recs if (VR/f"{r['video_id']}.mp4").exists()][:5]
from av_ib.data.musicavqa import render_question
for r in recs:
    vp=str(VR/f"{r['video_id']}.mp4")
    q=render_question(r['question_content'], r.get('templ_values','[]'))
    print(f"  {r['video_id']}  Q={q[:45]}", flush=True)
    m.forward_generate([vp],[vp],[q])
