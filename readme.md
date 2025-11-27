# AI7102 Project  
## Zero-Shot Diffusion for Bounded Video Generation

This repository contains the implementation of our AI7102 project on **zero-shot diffusion for bounded video generation**.  
Our method builds upon Stability AI’s *generative-models* framework and supports controllable, keyframe-bounded video synthesis.



## 🔧 How to Use

### 1. Environment Setup
Our code relies on the official **[generative-models](https://github.com/Stability-AI/generative-models)** repository. Then follow the environment setup instructions provided in the  **generative-models** repository.



### 2. Pre-trained Model
Download the Stable Video Diffusion **SVD-XT** weights from:

👉 https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt

After downloading, specify the model path in:

```
scripts/sampling/configs/svd_xt.yaml
```

under the field:

```
ckpt_path
````



### 3. Video Generation (Inference)
To run inference, execute:

```bash
cd code/generative-models
python scripts/sampling/demo.py
````

This will generate bounded video sequences using our zero-shot diffusion pipeline.


### Reference

> Part of our implementation references the official ViBiDSampler baseline.
You can find the source here: 👉 **[https://github.com/vibidsampler/vibid](https://github.com/vibidsampler/vibid)**

### Dataset
Datasets used in our experiments: [DAVIS](https://davischallenge.org/) and [Pexels](https://www.pexels.com/videos/).
