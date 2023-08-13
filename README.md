# Automatic Isotopy Identification with Deep Learning

This repository contains the code for the paper "Towards the Automatic Identification of Isotopies" presented at [Media Mutations 14](https://www.mediamutations.org/).

The aim of the experiments was addressing the problem of automatic isotopy identification, a novel task in the field of automated content analysis aimed at reducing the cost of annotation for the study of medical dramas. The work first involved expanding a subset of the [Medical Dramas Dataset](https://osf.io/24tus/) by including subtitles and keyframes for each segment from the TV show Grey's Anatomy. On the basis of the obtained corpus, experiments were conducted using unimodal and multimodal transformer-based models ([CLIP](https://huggingface.co/docs/transformers/model_doc/clip), [BERT](https://huggingface.co/docs/transformers/model_doc/bert) and [MMBT](https://github.com/facebookresearch/mmbt)). Two different classification approaches were also compared: the first approach consisted in employing a single multiclass classifier, while the second involved using the one-vs-the-rest approach. 

# Contents

1. `subtitles` contains the software used for the aligning the temporal annotations and the subtitles. 
2. `keyframes` contains the scripts used for extracting the keyframes of the segments.
3. `models` contains the scripts used for running the experiments with CLIP, BERT and MMBT.

# Results

|                  | Multiclass        |                 One-vs-the-rest                |
| Model            | All F1            | All F1     | PP F1 (e) | SP F1 (e) | MC F1 (e) |
| ---------------- | ----------------- | ---------- | --------- | --------- | --------- |
| `CLIP (3)`       | 0.536 (3)         | 0.443 (2)  | 0.696 (3) | 0.559     | 0.566     |
| `BERT (3)`       | 0.672 (2)         | 0.563 (2)  | 0.788 (2) | 0.706     | 0.686     |
| `MMBT (3)`       | 0.723 (3)         | 0.592 (3)  | 0.818 (3) | 0.728     | 0.713     |

Test F1 scores of the best models. (e) refers to the number of epochs.

