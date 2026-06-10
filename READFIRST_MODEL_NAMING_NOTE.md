# Model B / Model C Naming Mapping Note (Important!)

**The `modelB` and `modelC` naming in this code repository is reversed compared to the paper.**

- The paper's **Model B** (Intra-Separator / Internal FFI Conditioning, best-performing model)
  → use the **`*_modelC.py`** files in this repo
- The paper's **Model C** (Dual-Level Hybrid, Bottleneck + Internal FFI)
  → use the **`*_modelB.py`** files in this repo
- **Model A naming is consistent** — `*_modelA.py` (including the 64dim / 256dim variants) matches Model A in the paper, no conversion needed.

## File mapping

To run / reproduce the paper's **"Model B"**:
- `look2hear/models/tiger_modelC.py`
- `look2hear/datas/Libri2Mix16_modelC.py`
- `look2hear/system/audio_litmodule_modelC.py`
- `configs/tiger-small_modelC.yml`
- `audio_train_modelC.py`
- `inference_modelC.py`
- `WER_modelC.py`

To run / reproduce the paper's **"Model C"**:
- `look2hear/models/tiger_modelB.py`
- `look2hear/datas/Libri2Mix16_modelB.py`
- `look2hear/system/audio_litmodule_modelB.py`
- `configs/tiger-small_modelB.yml`
- `audio_train_modelB.py`
