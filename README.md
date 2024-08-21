# LDA-AQU: Adaptive Query-guided Upsampling via Local Deformable Attention
This repository presents the official PyTorch implementation of LDA-AQU (MM2024).

In this paper, we improve the upsampling operator commonly used in dense prediction tasks. Specifically, we introduce the local self-attention mechanism into the feature upsampling task and introduce the local deformation ability to reduce the semantic gap in the feature reorganization process.

<p align="center">
<img src="https://github.com/duzw9311/LDA-AQU/blob/main/figs/1_peformance.png" width=63% height=63% 
class="center">
</p>

Here are the overall architecture of proposed LDA-AQU.
<p align="center">
<img src="https://github.com/duzw9311/LDA-AQU/blob/main/figs/2_architecture.png"
class="center">
</p>

## Installation
Please see GETTING_STARTED.md for the basic usage of MMDetection.

## Training
```bash
bash tools/dist_train.sh configs/lda_aqu/fasterrcnn_r50_lau.py 4
```

## Testing
```bash
python tools/test.py configs/lda_aqu/fasterrcnn_r50_lau.py work_dirs/lda_aqu/latest.pth --eval bbox
```
## Weight
waiting.

## Acknowledgement
This repository is built upon the MMDetection library.

## Citation

```bash
waiting.
```