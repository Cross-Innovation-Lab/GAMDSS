<div align="center">
<h1>GAMDSS</h1> 
Evaluating and Correcting Human Annotation Bias in Dynamic Micro-Expression Recognition

 </div>

## Abstract
Existing manual labeling of micro-expressions is subject to errors in accuracy, especially in cross-cultural scenarios where deviation in labeling of key frames is more prominent. To address this issue, this paper presents a novel Global Anti-Monotonic Differential Selection Strategy (GAMDSS) architecture for enhancing the effectiveness of spatio-temporal modeling of micro-expressions through keyframe re-selection. Specifically, the method identifies Onset and Apex frames, which are characterized by significant micro-expression variation, from complete micro-expression action sequences via a dynamic frame reselection mechanism. It then uses these to determine Offset frames and construct a rich spatio-temporal dynamic representation. A two-branch structure with shared parameters is then used to efficiently extract spatio-temporal features. Extensive experiments are conducted on seven widely recognized micro-expression datasets. The results demonstrate that GAMDSS effectively reduces subjective errors caused by human factors in multicultural datasets such as SAMM and 4DME. Furthermore, quantitative analyses confirm that offset-frame annotations in multicultural datasets are more uncertain, providing theoretical justification for standardizing micro-expression annotations. These findings directly support our argument for reconsidering the validity and generalizability of dataset annotation paradigms. Notably, this design can be integrated into existing models without increasing the number of parameters, offering a new approach to enhancing micro-expression recognition performance.

## Overview
An overview of the proposed GAMDSS architecture is provided below. (a) The GAMDSS pipeline consists of the following steps: First, Dynamic Frame Reselection Mechanism reselects the three frames with the richest action changes based on different datasets. Second, a backbone model and feature processing method are selected. Next, spatio-temporal features are extracted at different stages using spatio-temporal units with two shared parameters. Where the temporal stream integrates the RMT module, which efficiently models long-term temporal dependencies through a retention mechanism based on Manhattan distance decay. Finally, the spatio-temporal features are integrated, and an auxiliary loss function is introduced to inject additional knowledge, thereby enabling the modeling of the complete evolution process of micro-expressions. (b) The designed method for extracting spatio-temporal features and their fusion approach, where Swish activation layers are employed to enhance feature nonlinearity and improve optimization stability. ![](./architecture.png)

## Getting Started

### Installation

**Step 1: Environment Setup:**
***Create and activate a new conda environment***

```bash
conda create -n GAMDSS python=3.8
conda activate GAMDSS
```

***Step 2: Install Dependencies:***

```bash
pip install -r requirements.txt
cd .\coding && pip install .
```
## Training
#### Training on CASME II dataset
```bash
cd .\cas2 && python train_c5.py
cd .\cas2 && python train_c3.py
```

#### Training on SAMM dataset
```bash
cd .\samm && python train_c5.py
cd .\samm && python train_c3.py
```

#### Training on CAS(ME)3 dataset
```bash
cd .\cas3 && python train_c7.py
cd .\cas3 && python train_c4.py
```

## Visual
#### T-SNE
```bash
cd .\visual && python tsne_samm_c5.py
cd .\visual && python tsne_cas2_c5.py
cd .\visual && python tsne_cas3_c7.py
```