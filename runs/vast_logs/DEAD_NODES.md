# k-sweep batch postmortem (2026-07-11)

Destroyed 3 of 4 nodes at ~13:30Z after confirming they produced nothing:

- n1 (k=16, 44498251): stuck 4h in HF model download (CN host, huggingface
  unreachable). GPU never used. Log: header line only.
- n3 (k=1, 44504118): same failure, 2h13m. Log: header line only.
- n4 (k=8, 44504126): CUDA OOM at step 1000 (24GB 3090, backward pass,
  tried to alloc 632MiB with 591MiB free). Auto-restarted but never logged
  again; SSH then refused. Partial scores before crash:
    step 0: gap_closed -4.299 | step 500: 0.715 | step 1000: 0.809
- n2 (k=4): healthy, kept running. Its log is n2.log in this dir.

Relaunch notes for k=1/8/16:
1. onstart must verify huggingface.co reachable (curl -m 10) BEFORE burning
   hours; avoid CN-geolocation offers.
2. k=8 OOMs on 24GB with current batch/seq settings — needs
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and/or smaller batch,
   or a 48GB card (A6000/L40).
