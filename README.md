## VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised Speech Representation Disentanglement for One-shot Voice Conversion (Interspeech 2021)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2106.10132)
[![GitHub Stars](https://img.shields.io/github/stars/Wendison/VQMIVC?style=social)](https://github.com/Wendison/VQMIVC)
[![download](https://img.shields.io/github/downloads/Wendison/VQMIVC/total.svg)](https://github.com/Wendison/VQMIVC/releases)

### [Run VQMIVC on Replicate](https://replicate.ai/wendison/vqmivc)
### Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/VQMIVC).

### Pre-trained models: [google-drive](https://drive.google.com/file/d/1Flw6Z0K2QdRrTn5F-gVt6HdR9TRPiaKy/view?usp=sharing) or [here](https://github.com/Wendison/VQMIVC/releases) | [Paper demo](https://wendison.github.io/VQMIVC-demo/)



## Acknowledgements:
* The content encoder is borrowed from [VectorQuantizedCPC](https://github.com/bshall/VectorQuantizedCPC), which also inspires the negative sampling within-utterance for CPC;
* The speaker encoder is borrowed from [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion);
* The decoder is modified from [AutoVC](https://github.com/auspicious3000/autovc);
* Estimation of mutual information is modified from [CLUB](https://github.com/Linear95/CLUB);
* Speech features extraction is based on [espnet](https://github.com/espnet/espnet) and [Pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder).



