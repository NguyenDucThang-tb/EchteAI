# SeaDronesSee training on Google Colab

Open `SeaDronesSee_ConvNeXt_QAT.ipynb` in Colab and select a GPU runtime. The
notebook performs the complete workflow:

The notebook clones `https://github.com/NguyenDucThang-tb/EchteAI.git`; create
that repository and push this workspace before opening Colab.

1. mount Google Drive;
2. clone/update EchteAI and install dependencies;
3. use public WebDAV via `rclone` to copy only the ~10 GB JPEG-compressed
   SeaDronesSee v2 release into `/content/seadronessee`;
4. verify the five foreground categories and ignore category ID 0;
5. train the FP32 baseline and save `best` plus per-epoch `last` checkpoints;
6. resume automatically from `fp32_last.pt` after an interrupted session;
7. calibrate and train selective M3 QAT, also resumable from `qat_last.pt`;
8. convert the best frozen-observer checkpoint to selective INT8;
9. evaluate FP32/INT8 on validation and run the CPU benchmark.

All durable artifacts are written to:

```text
/content/drive/MyDrive/EchteAI/seadronessee_m3/
```

The official 3,750-image test split has no public bounding-box annotations, so
the Colab config points local test metrics/benchmark to validation. Test images
remain available for a future challenge-submission exporter.

The default `800–960 / max 1600` resolution and batch size 1 target a Colab T4.
If CUDA runs out of memory, change `train_min_sizes` to `[640, 736, 800]` and
`max_size` to `1333`. On an A100, `[1024, 1152, 1280] / max 2048` preserves more
small swimmers and buoys.
