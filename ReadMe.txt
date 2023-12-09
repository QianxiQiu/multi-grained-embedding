# End-to-end approach of multi-grained embedding of categorical features in tabular data

## Abstract

In recent years, it has been a commonly adopted strategy to transform categorical data into numerical one to suit popular learning approaches, such as neural networks. The above mentioned transformation has been undertaken popularly through feature embedding within the setting of representation learning, which has led to successful applications in natural language processing and knowledge graph. However, in the context of tabular data processing, the transformation of categorical features into numerical ones is still typically achieved by using handcrafted methods of category encoding. In this paper, we propose an end-to-end approach of multi-grained embedding of categorical features in tabular data in the setting of decision forests driven representation learning. Specifically, we incorporate an uncertainty-aware optimization strategy into the proposed approach to guide the process of end-to-end learning. The proposed approach has been evaluated experimentally on 12 real-world data sets. The experimental results show that the proposed approach outperforms 10 baselines in terms of feature transformation, leading to an improvement of classification accuracy by at least 3% on most data sets

## Code Overview

This repository contains the code implementation for the paper titled "End-to-end approach of multi-grained embedding of categorical features in tabular data." The code is organized as follows:

- `proposed.py` corresponds to the proposed(CB) method in the paper.

- `proposed_weka_tree.py` corresponds to the proposed(RF) method in the paper.
- `GCForest.py` corresponds to the sklearn version of the CF method in the paper.
- `GCForest_weka.py` corresponds to the Weka version of the CF method in the paper.

## Getting Started

### Prerequisites

- jdk version: jdk1.8.0_391
- weka3.8.4
- Python dependencies: 

  - numpy
  - pandas

  - jpype
  - torch
  - catboost
  - sklearn

### Usage

Refer to `main.py` .





