"""Build presentation-ready figure + LaTeX table from the 10-clip token analysis."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("token_data.json"))["clips"]
def col(k): return np.array([c[k] for c in D], float)

# per-token attention (x1e4), averaged over clips
pt = {m: (col(f"pt_{m}_raw").mean()*1e4, col(f"pt_{m}_vib").mean()*1e4)
      for m in ("vid","aud","txt")}
# key norms
kn_aud = (col("k_norm_aud_raw").mean(), col("k_norm_aud_vib").mean())
kn_vid = (col("k_norm_vid_raw").mean(), col("k_norm_vid_vib").mean())
aud_cos = col("aud_cos_sim").mean()
vid_cos = col("vid_cos_sim").mean()
# total attention share (raw)
share = (col("attn_video_raw").mean(), col("attn_audio_raw").mean(), col("attn_text_raw").mean())
ratio = (col("n_video")/col("n_audio")).mean()

RAW, VIB = "#1f77b4", "#e67814"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# Panel 1: total attention share (the puzzle)
ax[0].bar(["Video","Audio","Text"], share, color=["#1f77b4","#2ca02c","#888"])
for i,v in enumerate(share): ax[0].text(i, v+0.01, f"{v*100:.0f}%", ha="center", weight="bold")
ax[0].set_ylim(0,1); ax[0].set_ylabel("Share of total attention")
ax[0].set_title(f"The puzzle: video gets ~90%\n(video has {ratio:.0f}x more tokens)", weight="bold")

# Panel 2: per-token attention (the answer)
x = np.arange(3); w=0.38
ax[1].bar(x-w/2, [pt["vid"][0],pt["aud"][0],pt["txt"][0]], w, label="raw", color=RAW)
ax[1].bar(x+w/2, [pt["vid"][1],pt["aud"][1],pt["txt"][1]], w, label="VIB", color=VIB)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Video","Audio","Text"])
ax[1].set_ylabel(r"Attention per token ($\times10^{-4}$)")
ax[1].set_title("The answer: per token,\naudio & text beat video", weight="bold")
ax[1].legend()

# Panel 3: audio key norm drops under VIB (the mechanism)
ax[2].bar(["raw","VIB"], kn_aud, color=[RAW,VIB])
for i,v in enumerate(kn_aud): ax[2].text(i, v+0.1, f"{v:.1f}", ha="center", weight="bold")
ax[2].set_ylim(0,15); ax[2].set_ylabel("Mean audio key norm")
ax[2].set_title(f"The mechanism: VIB rotates audio\n(cos={aud_cos:.2f}) -> keys shrink", weight="bold")

fig.suptitle("Why the model relies on video more than audio  (mean over 10 clips)",
             fontsize=14, weight="bold")
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig("axis2_summary.png", dpi=150)
print("saved axis2_summary.png")

# ── consistency check: how many clips show audio>video per token, and key drop ──
aud_gt_vid = int((col("pt_aud_raw") > col("pt_vid_raw")).sum())
key_drop = int((col("k_norm_aud_vib") < col("k_norm_aud_raw")).sum())
print(f"audio>video per token: {aud_gt_vid}/10 clips")
print(f"audio key shrinks under VIB: {key_drop}/10 clips")

# ── LaTeX table ──
with open("axis2_table.tex","w") as f:
    f.write(r"""\begin{tabular}{lccc}
\toprule
(mean over 10 clips) & Video & Audio & Text \\
\midrule
share of total attention & %.0f\%% & %.0f\%% & %.0f\%% \\
number of tokens & 1440--5760 & $\sim$130 & $\sim$68 \\
\rowcolor{Soft} attention \emph{per token} ($\times10^{-4}$) & %.2f & \textbf{%.2f} & \textbf{%.2f} \\
\midrule
audio cosine sim (raw vs VIB) & \multicolumn{3}{c}{%.2f \;(video: %.2f)} \\
audio key norm raw $\to$ VIB & \multicolumn{3}{c}{%.1f $\to$ %.1f} \\
\bottomrule
\end{tabular}
""" % (share[0]*100, share[1]*100, share[2]*100,
       pt["vid"][0], pt["aud"][0], pt["txt"][0],
       aud_cos, vid_cos, kn_aud[0], kn_aud[1]))
print("saved axis2_table.tex")
