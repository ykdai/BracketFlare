## Nighttime Smartphone Reflective Flare Removal using Optical Center Symmetry Prior (CVPR2023)
[Paper](https://arxiv.org/abs/2303.15046) | [Project Page](https://ykdai.github.io/projects/BracketFlare) | [Video](https://www.youtube.com/watch?v=FM8kAM13zUA)

[Yuekun Dai](https://ykdai.github.io/), [Yihang Luo](https://github.com/Luo-Yihang), [Shangchen Zhou](https://shangchenzhou.com/), [Chongyi Li](https://li-chongyi.github.io/),  [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

### News

- **2022.03.22**: Our BracketFlare dataset and pretrained model is uploaded to the Google Drive and Baidu Netdisk now.
- **2022.03.21**: Our paper is selected as a **highlight (top2.5%)** at CVPR 2023.
- **2023.02.28**: Our paper *Nighttime Smartphone Reflective Flare Removal using Optical Center Symmetry Prior* is accepted by the CVPR2023 with **3 top ratings**. 🤗

### Data Download

|              |                        Baidu Netdisk                         |                         Google Drive                         | Number | Description                                                  |
| :----------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----- | ------------------------------------------------------------ |
| BracketFlare | [link](https://pan.baidu.com/s/1vdpnZa4ZXBjb9lXEdaswjQ?pwd=cvpr) | [link](https://drive.google.com/file/d/1UBGvI_KCPkEyCNQkFrNNPY5LnXjAm2Oi/view?usp=share_link) | 440    | We offer 440 normal-exposure and short-exposure nighttime image pairs. Short exposure images can be added to the corresponding normal-exposure image to synthesize the flare-corrupted image. |


### BracketFlare Data structure

```
├── BracketFlare
    ├── flare
    ├── gt
    ├── test
         ├── flare
         ├── gt
         ├── mask
         ├── lq
```

### Pretrained Model

The inference code based on MPRNet is released Now. You can download the pretrained checkpoints on [[GoogleDrive](https://drive.google.com/file/d/15AzR-VaiQO0l8Av-yE6gVuExcTwYAi45/view?usp=share_link) | [Baidu Netdisk](https://pan.baidu.com/s/1jWg_uZMlFhT5MOCWaQmWOQ?pwd=cvpr )]. Please place it under the `experiments` folder and unzip it.

### Inference Code

To generate the flare-free images from the `test/lq` folder,  you can run the `test.py` by using:

```
python test.py --i test/lq/ --o result/ --model_path expirements/net_g_last.pth
```

### Evaluation Code

To calculate different metrics with our pretrained model, you can run the `evaluate.py` by using:

```
python evaluate.py --input result/deflare/ --gt test/gt --mask test/mask
```

### License

This project is licensed under <a rel="license" href="https://github.com/ykdai/BracketFlare/blob/main/LICENSE">S-Lab License 1.0</a>. Redistribution and use of the dataset and code for non-commercial purposes should follow this license. This project is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), the detailed tutorial on training commands and config settings can be found [here](https://github.com/XPixelGroup/BasicSR/blob/master/docs/introduction.md). 

### Citation

If you find this work useful, please cite:

```
@inproceedings{dai2023nighttime,
  title={Nighttime Smartphone Reflective Flare Removal using Optical Center Symmetry Prior},
  author={Dai, Yuekun and Luo, Yihang and Zhou, Shangchen and Li, Chongyi and Loy, Chen Change},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
 }
```

### Contact

If you have any question (such as asking for the 4K version of our BracketFlare Dataset), please feel free to reach me out at `ydai005@e.ntu.edu.sg`.
