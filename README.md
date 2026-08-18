# BiOTPrompt: Bidirectional Optimal Transport Guided Prompting for Disease Evolution-aware Report Generation

## Introduction
Radiology report generation (RRG) aims to automatically describe medical images via free-text reports. In clinical practice, comparing current and prior chest X-rays is essential for assessing disease progression, motivating the development of longitudinal RRG methods. However, most existing approaches often struggle to capture fine-grained temporal changes, as they often rely on unidirectional alignments or static reasoning pipelines, overlooking the bidirectional and asymmetric nature of disease evolution. To address these challenges, we propose BiOTPrompt, a novel framework for disease evolution-aware report generation, which introduces a Bidirectional Optimal Transport (BiOT) mechanism to explicitly model progression dynamics between historical and current chest X-rays. By analyzing the asymmetry between bidirectional transport plans, BiOTPrompt can identify newly emerged and resolved regions, which are then used to construct dynamic prompts that guide large language models (LLMs) in generating clinically relevant diagnostic reports. Furthermore, we incorporate a vision-language consistency constraint to ensure alignment between visual evidence and textual descriptions, mitigating hallucinations and enhancing factual accuracy. Extensive experiments on the Longitudinal-MIMIC dataset demonstrate that BiOTPrompt achieves state-of-the-art performance in both language quality metrics and clinical relevance, setting a new benchmark for longitudinal radiology report generation.

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
cd BiOTPrompt
pip install -r requirements.txt
```

**2. Prepare the training dataset**

Longitudinal-MIMIC: you can download this dataset from [here](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training

```bash
bash scripts/6-1.deep_run.sh
```

### Testing (For MIMIC-CXR)

```bash
bash scripts/6-2.deep_test.sh
```

## Acknowledgement

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) Some codes of this repo are based on R2GenGPT.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
