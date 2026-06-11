# AVQA experiment pipeline (clean restart)

Dataset: AVQA (Yang et al., ACM MM 2022) — 57,335 multiple-choice QA over
57,015 ten-second VGGSound clips. Train 40,425 / val(test) 16,910.
Supervised SOTA reference: ~89% (HCRN+HAVF, 2022).

## Order

| step | script | where | what |
|---|---|---|---|
| 1 | `1_download_dataset.py` | login node | annotations (GitHub) + videos (HF Loie/VGGSound tarballs, ~100 GB). `download_videos_ytdlp.py` = fallback for clips missing from the mirror (~5% dead is normal) |
| 2 | `2_mine_abstention_labels.qsub` | qsub from `av_ib/` | corruption-flip labels on AVQA train (~10 h) |
| 3-5 | `3_…dpo` / `4_…sft` / `5_…av_dpo` | qsub | abstention training: video-VIB DPO, video-VIB SFT, video+audio-VIB DPO (need step 2) |
| 6 | `6_train_v6b_base.qsub` | qsub | v6b base training (rate penalty on) — independent of 2 |
| 7 | `7_train_lora_baseline.qsub` | qsub | vanilla + LoRA reference — independent of 2 |
| 8 | `8_evaluate_all.qsub` | qsub | accuracy (val split) for 6/7/untrained + risk-coverage for 3/4/5 |

Steps 3-5 and 6-7 can run in parallel. All outputs land in `runs/avqa/`.
Submit qsub scripts from the `av_ib/` directory (paths are relative to
`$PBS_O_WORKDIR`).
