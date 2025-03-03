### Pre-trained models: [google-drive](https://drive.google.com/file/d/1Flw6Z0K2QdRrTn5F-gVt6HdR9TRPiaKy/view?usp=sharing) or [here](https://github.com/Wendison/VQMIVC/releases) | [Paper demo](https://wendison.github.io/VQMIVC-demo/)


# Model Summary
(models_stats.py)

| Component         | Total Parameters | Total Mult-adds (M/G) | Input Size (MB) | Forward/Backward Pass Size (MB) | Params Size (MB) | Estimated Total Size (MB) | 
|-------------------|------------------|-----------------------|--------------------------|-----------------------|-----------------|----------------------------------|-------------------|---------------------------|--------------------|
| **Encoder**    | 1,580,096           |     25.76 M      | 0.03               | 2.18            | 6.32                            | 8.53              |
| **Encoder_spk**    | 1,726,848           |        101.64 M                | 0.03           | 1.56                           | 6.91              | 8.50                     | 
| **Decoder**   | 24,773,440       |        6.91 G         | 0.04        | 20.03           | 99.09                            | 119.17                 | 


### Total Model Summary:
- **Total Parameters**: 28,080,384
- **Total Mult-adds**: 3.32 G
- **Total Running Time**: 0.147558 s






## Acknowledgements:
* The content encoder is borrowed from [VectorQuantizedCPC](https://github.com/bshall/VectorQuantizedCPC), which also inspires the negative sampling within-utterance for CPC;
* The speaker encoder is borrowed from [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion);
* The decoder is modified from [AutoVC](https://github.com/auspicious3000/autovc);
* Estimation of mutual information is modified from [CLUB](https://github.com/Linear95/CLUB);
* Speech features extraction is based on [espnet](https://github.com/espnet/espnet) and [Pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder).



