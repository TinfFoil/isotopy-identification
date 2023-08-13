# Automatic Isotopy Identification with Deep Learning

This repository contains the code for the paper "Towards the Automatic Identification of Isotopies" presented at [Media Mutations 14](https://www.mediamutations.org/).

The aim of the experiments was addressing the problem of automatic isotopy identification, a novel task in the field of automated content analysis aimed at reducing the cost of annotation for the study of medical dramas. The work first involved expanding a subset of the [Medical Dramas Dataset](https://osf.io/24tus/) by including subtitles and keyframes for each segment from the TV show Grey's Anatomy. On the basis of the obtained corpus, experiments were conducted using unimodal and multimodal transformer-based models ([CLIP](https://huggingface.co/docs/transformers/model_doc/clip), [BERT](https://huggingface.co/docs/transformers/model_doc/bert) and [MMBT](https://github.com/facebookresearch/mmbt)). Two different classification approaches were also compared: the first approach consisted in employing a single multiclass classifier, while the second involved using the one-vs-the-rest approach. 

# Subfolders

1. `subtitles` contains the software used for the aligning the temporal annotations and the subtitles. 
2. `keyframes` contains the scripts used for extracting the keyframes of the segments.
3. `models` contains the scripts used for running the experiments with CLIP, BERT and MMBT.

# Results

F1-measures of the best models on test:

|    Model    | Multiclass F1 (All) | One-vs-the-rest F1 (All) |
|:-----------:|:-------------------:|:------------:|
|    `CLIP`   |        0.536        |    0.566     |
|    `BERT`   |        0.672        |    0.686     |
|    `MMBT`   |        0.723        |    0.713     |

# Meta

[Alice Fedotova](https://www.linkedin.com/in/alice-fedotova/) - [ffedox](https://github.com/ffedox) - [alice.fedotova@studio.unibo.it](mailto:alice.fedotova@studio.unibo.it)
