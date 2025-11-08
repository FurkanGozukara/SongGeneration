---
language:
- en
- zh
pipeline_tag: text-to-audio
library_name: tencent-song-generation
---

# SongGeneration

<p align="center"><img src="img/logo.jpg" width="40%"></p>
<p align="center">
    <a href="https://levo-demo.github.io/">Demo</a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2506.07520">Paper</a>  &nbsp;|&nbsp; <a href="https://github.com/tencent-ailab/songgeneration">Code</a>  &nbsp;|&nbsp; <a href="https://huggingface.co/spaces/tencent/SongGeneration">Space Demo</a>
</p>


This repository is the official weight repository for LeVo: High-Quality Song Generation with Multi-Preference Alignment. In this repository, we provide the SongGeneration model, inference scripts, and the checkpoint that has been trained on the Million Song Dataset.

## Model Versions

| Model                     | Max Length |       Language       | GPU Menmory | RFT(A100) | Download Link                                                |
| ------------------------- | :--------: | :------------------: | :---------: | :-------: | ------------------------------------------------------------ |
| SongGeneration-base       |   2m30s    |          zh          |   10G/16G   |   1.26    | You were here |
| SongGeneration-base-new   |   2m30s    |        zh, en        |   10G/16G   |   1.26    | [Huggingface](https://huggingface.co/lglg666/SongGeneration-base-new) |
| SongGeneration-base-full  |   4m30s    |        zh, en        |   12G/18G   |   1.30    | [Huggingface](https://huggingface.co/lglg666/SongGeneration-base-full) |
| SongGeneration-large      |   4m30s    |        zh, en        |   22G/28G   |   1.51    | [Huggingface](https://huggingface.co/lglg666/SongGeneration-large) |
| SongGeneration-v1.5-small |     2m     | zh, en, es, ja, etc. |      -      |     -     | Coming soon                                                  |
| SongGeneration-v1.5-base  |   4m30s    | zh, en, es, ja, etc. |      -      |     -     | Coming soon                                                  |
| SongGeneration-v1.5-large |   4m30s    | zh, en, es, ja, etc. |      -      |     -     | Coming soon                                                  |

## Overview

We develop the SongGeneration model. It is an LM-based framework consisting of **LeLM** and a **music codec**. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. The music codec reconstructs the dual-track tokens into highfidelity music audio. SongGeneration significantly improves over the open-source music generation models and performs competitively with current state-of-the-art industry systems. For more details, please refer to our [paper](https://arxiv.org/abs/2506.07520).

<img src="https://github.com/tencent-ailab/songgeneration/blob/main/img/over.jpg?raw=true" alt="img" style="zoom:100%;" /> 

## License

The code and weights in this repository is released in the [LICENSE](LICENSE)  file.
