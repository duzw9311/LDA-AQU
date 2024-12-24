# LDA-AQU: Adaptive Query-guided Upsampling via Local Deformable Attention
This repository presents the official PyTorch implementation of **LDA-AQU (MM'2024)**.

In this [paper](https://arxiv.org/abs/2411.19585), we propose LDA-AQU, which incorporates local self-attention into the feature upsampling process and introduces local deformation capabilities to mitigate the semantic gap between interpolation points and their neighboring points selected during feature reassembly.


Here is the performance comparison of various upsampling operators integrated into the Faster RCNN detector on the MS COCO dataset.
<p align="center">
<img src="resources/1_peformance.png" width=63% height=63% 
class="center">
</p>

Here is the overall architecture of the proposed LDA-AQU.
<p align="center">
<img src="resources/2_architecture.png"
class="center">
</p>

## Installation
Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection.

## Training
```bash
bash tools/dist_train.sh configs/lda_aqu/fasterrcnn_r50_lau.py 4
```

## Testing
```bash
python tools/test.py configs/lda_aqu/fasterrcnn_r50_lau.py work_dirs/lda_aqu/latest.pth --eval bbox
```
## Weight
Model | AP | Link1 | Link2 |
--- |:---:|:---:|:---:
fasterrcnn_r50_lau             | 39.2 | [BaiduNetDisk](https://pan.baidu.com/s/1ljcF0FI1zyJJdwARGNlsew?pwd=0eiv) | [GoogleDrive](https://drive.google.com/file/d/1HE2pSYXsd-c_9NMfcN5b7CbJXpK0sxuW/view?usp=drive_link)

## Acknowledgement
This repository is built upon the [MMDetection](https://github.com/open-mmlab/mmdetection) library.

## Citation
If you find this paper helpful for your project, we'd appreciate it if you could cite it.
```
@inproceedings{du2024lda,
  title={LDA-AQU: Adaptive Query-guided Upsampling via Local Deformable Attention},
  author={Du, Zewen and Hu, Zhenjiang and Zhao, Guiyu and Jin, Ying and Ma, Hongbin},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={4919--4927},
  year={2024}
}
```