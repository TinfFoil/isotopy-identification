# Unimodal and multimodal automatic isotopy identification

This repository contains the code for the paper "Towards the Automatic Identification of Isotopies" (2023), presented at the [14th Media Mutations International Conference](10.21428/93b7ef64.59f47006) by Alice Fedotova and Alberto Barrón-Cedeño. 

# Overview

The aim of the experiments was addressing the problem of automatic isotopy identification, a novel task in the field of automated content analysis aimed at reducing the cost of annotation for the study of medical dramas. The work first involved expanding a subset of the [Medical Dramas Dataset](https://osf.io/24tus/) by including subtitles and keyframes for each segment from the TV show Grey's Anatomy. On the basis of the obtained corpus, experiments were conducted using unimodal and multimodal transformer-based models ([CLIP](https://huggingface.co/docs/transformers/model_doc/clip), [BERT](https://huggingface.co/docs/transformers/model_doc/bert) and [MMBT](https://github.com/facebookresearch/mmbt)). Two different classification approaches were also compared: the first approach consisted in employing a single multiclass classifier, while the second involved using the one-vs-the-rest approach. 

# Subfolders

1. `subtitles` contains the software used for the aligning the temporal annotations and the subtitles. 
2. `keyframes` contains the scripts used for extracting the keyframes of the segments.
3. `models` contains the scripts used for running the experiments with CLIP, BERT and MMBT.

# Requirements

The repository contains a `Pipfile` with all required dependencies, which can be installed using pipenv: 

```sh
pipenv install Pipfile
```

# Contact

[Alice Fedotova](https://www.linkedin.com/in/alice-fedotova/) – [ffedox](https://github.com/ffedox) – [alice.fedotova@studio.unibo.it](mailto:alice.fedotova@studio.unibo.it)
