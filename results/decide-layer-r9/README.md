# Round 9 closes, read for the decide layer (2026-11-10)

Build: 4B NVFP4 W4A8 · llama.cpp · PC layer, round 9, 232 of 240 pairs (a2 pending). Records live on the PC
(`C:/Users/xenod/mip-work/a100/pc/r9{a,b}-{gen,ctrl}`), too large to keep here.

- `r9_closes.py <out>`: every loop close and answer close of the General arms with control's result on the same key; writes
  `<out>.json` (`r9_closes.json` here) and `<out>.txt` (the text around each close, for reading).
- `r9_labels.py r9_closes.json`: the hand labels (class A-F, read case by case) and the class x outcome table.
- `r9_prevent.py`: of the keys control ended `stuck` and General ended right with no close, whether a General put on control's
  repeated equation came first.

The design these feed: [docs/plans/2026-11-10-decide-layer-design.md](../../docs/plans/2026-11-10-decide-layer-design.md).
