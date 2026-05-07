<div align="center">
<h2 class="papername"> UniEmo: Unifying Emotional Understanding and Generation with Learnable Expert Queries </h2>
<div>
    <a href="https://scholar.google.com.hk/citations?user=0GtAUPoAAAAJ&hl=zh-CN&oi=sra" target="_blank">Yijie Zhu</a>,
    <a href="https://zls030726.github.io/" target="_blank">Lingsen Zhang</a>,
    <a href="https://zitongyu.github.io/" target="_blank">Zitong Yu*</a>, 
    <a href="https://rshaojimmy.github.io/OrionLab/" target="_blank">Rui Shao*</a>,
    <a href="https://scholar.google.com/citations?user=lLg3WRkAAAAJ&hl=en" target="_blank">Tao Tan</a>,
    <a href="http://faculty.hitsz.edu.cn/guanweili" target="_blank">Liqiang Nie</a>
</div>

School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen<br>
Great Bay University<br>
Macau Polytechnic University<br>
*Corresponding author<br>
[![UniEmo (TIP2026)](https://img.shields.io/badge/UniEmo%20%28TIP2026%29-arXiv_2507.23372-b31b1b.svg?logo=arxiv)](https://arxiv.org/pdf/2507.23372)
[![EmoSym (ACM MM 2025)](https://img.shields.io/badge/EmoSym%20%28ACM%20MM%202025%29-arXiv_2407.14439-b31b1b.svg?logo=arxiv)](https://dl.acm.org/doi/abs/10.1145/3746027.3754549)

</div>

</div>

## 🔥 If you find this work useful for your research, please kindly cite our paper and star our repo.
## :fire: Updates
- [05/2026] :fire: The code is released. Enjoy it!
- [04/2026] :fire: UniEmo has been accepted by **TIP**!
- [07/2025] :fire: [arXiv paper](https://arxiv.org/pdf/2507.23372) released!
## 🔥 Introduction

This is the official repository for **UniEmo: Unifying Emotional Understanding and Generation with Learnable Expert Queries**.

In this work, we introduce **UniEmo**, a unified framework that synergistically integrates visual emotional understanding and emotion-conditioned image generation. UniEmo builds a hierarchical emotional understanding chain with learnable expert queries to progressively capture scene-level and object-level emotional cues. These representations are further used to guide emotional image generation, while the generation branch provides dual feedback to enhance emotional understanding.

The overall framework of UniEmo is shown below:

<div align="center">
<img src='assets/model.png' width='100%'>
</div>

This repository contains the official implementation of our code.

## 1. Data and Environment Preparation

Our codebase follows the same environment setup as [EmoGen](https://github.com/JingyuanYY/EmoGen). Please refer to the EmoGen repository for detailed instructions on environment installation, dataset preparation, and required pretrained models.

Before running our code, please make sure that:

- The Python environment has been properly configured following EmoGen.
- The required datasets have been downloaded.
- The required initial/pretrained models have been prepared according to the EmoGen instructions.

## 2. Stage 1: Emotional Understanding

### 2.1 Training

Before training the emotional understanding module, please modify the `config_vit.yaml` configuration file as follows:

```yaml
test_only: False
cal_emotion_space: False
data_dir: /path/to/your/EmoSet
```

Here, `data_dir` should be replaced with the path to your local EmoSet dataset.

After completing the configuration, launch Stage 1 training with:

```bash
accelerate launch --config_file ./accelerate/multi_gpu.yaml train_vit_class/train_vit.py
```
### 2.2 Evaluation for Understanding

Before evaluating the emotional understanding module, please modify the `config_vit.yaml` configuration file as follows:

```yaml
model_resume: /path/to/your/stage1_trained_model
test_only: True
cal_emotion_space: False
```

Here, `model_resume` should be set to the path of the trained Stage 1 model checkpoint.

After completing the configuration, launch the evaluation with:

```bash
accelerate launch --config_file ./accelerate/multi_gpu.yaml train_vit_class/train_vit.py
```




## 3. Stage 2: Joint Training of Emotional Understanding and Generation

### 3.1 Joint Training

Before starting joint training, please make sure that the `Dataset_balance`-related `.pkl` files have been downloaded and prepared following the instructions in [EmoGen](https://github.com/JingyuanYY/EmoGen).

Then, modify the `config.yaml` configuration file as follows:

```yaml
pretrained_model_name_or_path: /path/to/your/stable-diffusion-v1-5
train_data_dir: /path/to/your/dataset
VIT_path: /path/to/your/stage1_trained_VIT_model
```

Here, `pretrained_model_name_or_path` should be set to the path of your downloaded Stable Diffusion v1.5 model, `train_data_dir` should be replaced with the path to your training dataset, and `VIT_path` should be set to the path of the VIT model trained in Stage 1.

### 3.2 Evaluation

For emotional understanding evaluation, follow the same procedure as Stage 1 evaluation. Replace the Stage 1 evaluation checkpoint with the newly trained model.

For emotional generation evaluation, the emotion space for expert query and CLS features needs to be computed first. Please modify the `config_vit.yaml` configuration file as follows:

```yaml
model_resume: /path/to/your/VIT_query
test_only: True
cal_emotion_space: True
```

Here, `model_resume` should be set to the path of the new trained `VIT_query` model.

After completing the configuration, run:

```bash
accelerate launch --config_file ./accelerate/multi_gpu.yaml train_vit_class/train_vit.py
```

The computed emotion-space features will be saved in the `UniEmo_space` folder.
Then, modify the `file = []` field in `inference.py` by setting it to the checkpoint path saved from Stage 2 training:

```python
file = [
    "/path/to/your/stage2_trained_checkpoint"
]
```

Here, `/path/to/your/stage2_trained_checkpoint` should be replaced with the path of the model weights obtained from Stage 2 joint training.
Then, generate images for evaluation by running:

```bash
python training/inference.py --emotion_idx XXX
```

Here, `XXX` should be replaced with the target emotion index. This command will generate the corresponding image files for evaluation.
Finally, please refer to the evaluation pipeline in [EmoGen](https://github.com/JingyuanYY/EmoGen) for computing generation metrics such as FID, Emo-A, and other related evaluation scores.
The overall generation evaluation process is:
1. Generate images for each target emotion using our inference script.
2. Compute generation metrics such as FID and Emo-A using the evaluation scripts provided in EmoGen.
## 🔥 Visualization
### Emotional Image Generation Results

Emotion-evoking images generated by **UniEmo**:

<div align="center">
<img src='assets/vis.png' width='100%'>
</div>

UniEmo can generate visually coherent and emotionally expressive images conditioned on different target emotions. The results show that the generated images not only preserve clear semantic content, but also exhibit emotion-specific visual cues such as color tone, scene atmosphere, object appearance, and contextual composition.

### Attention Visualization of Expert Queries

Visualization of attention maps from the two types of expert queries:

<div align="center">
<img src='assets/atten.png' width='100%'>
</div>

The attention maps demonstrate that the proposed expert queries learn complementary emotional cues. Scene-level queries tend to focus on global contextual regions that shape the overall emotional atmosphere, while object-level queries attend to local emotion-related objects or details. This coarse-to-fine attention behavior supports UniEmo in extracting richer emotional representations for both understanding and generation.


## 📝 Citation

```bib
@misc{zhu2025uniemounifyingemotionalunderstanding,
      title={UniEmo: Unifying Emotional Understanding and Generation with Learnable Expert Queries}, 
      author={Yijie Zhu and Lingsen Zhang and Zitong Yu and Rui Shao and Tao Tan and Liqiang Nie},
      year={2025},
      eprint={2507.23372},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.23372}, 
}
@inproceedings{zhu2025emosym,
  title={EmoSym: A Symbiotic Framework for Unified Emotional Understanding and Generation via Latent Reasoning},
  author={Zhu, Yijie and Lyu, Yibo and Yu, Zitong and Shao, Rui and Zhou, Kaiyang and Nie, Liqiang},
  booktitle={Proceedings of the 33nd ACM International Conference on Multimedia},
  year={2025}
}
```
## Acknowledgement
* Lots of code are inherited from [EmoGen](https://github.com/JingyuanYY/EmoGen) and [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP). Thanks for all these great works.


