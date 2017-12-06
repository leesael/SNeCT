# SNeCT

Overview
---------------

**Motivation**: How do we integratively analyze large-scale multi-platform genomic data that are high dimensional and sparse? Furthermore, how can we incorporate prior knowledge, such as the association between genes, in the analysis systematically?

**Method**: To solve this problem, we propose a **S**calable **Ne**twork **C**onstrained **T**ucker decomposition method we call SNeCT. SNeCT adopts parallel stochastic gradient descent approach on the proposed parallelizable network constrained optimization function. SNeCT decomposition is applied to tensor constructed from large scale multi-platform multi-cohort cancer data, PanCan12, constrained on a network built from PathwayCommons database.

**Results**: The decomposed factor matrices are applied to stratify cancers, to search for top-k similar patients, and to illustrate how the matrices can be used for personalized interpretation. In the stratification test, combined twelve-cohort data is clustered to form thirteen subclasses. The thirteen subclasses have a high correlation to tissue of origin in addition to other interesting observations, such as clear separation of OV cancers to two groups, and high clinical correlation within subclusters formed in cohorts BRCA and UCEC. In the top-k search, a new patient’s genomic profile is generated and searched against existing patients based on the factor matrices. The similarity of the top-k patient to the query is high for 23 clinical features, including estrogen/progesterone receptor statuses of BRCA patients with average precision value ranges from 0.72 to 0.86 and from 0.68 to 0.86, respectively. We also provide an illustration of how the factor matrices can be used for interpretable personalized analysis of each patient.

![scheme_img](/img/scheme.png)


Paper
---------------

**SNeCT: Integrative cancer data analysis via large scale network constrained tensor decomposition**  
[Dongjin Choi](https://skywalker5.github.io/), [Lee Sael](http://www3.cs.stonybrook.edu/~sael/)  
[[PDF](https://arxiv.org/pdf/1711.08095.pdf), [Supplementary material](/paper/Supplementary_Information.pdf), [Slides](/slide/SNeCT_171114.pdf)]

Code
---------------
See Github repository for [SNeCT_Code](https://github.com/skywalker5/SNeCT_code)


Data
---------------
| Name | Structure | Size | Number of Entries | Download |
| :------------ | :-----------: | :-------------: |------------: |:------------------: |
| PanCan12     | Patient - Gene - Platform | 4,555 &times; 14,351 &times; 5 | 183,211,020 | [DOWN](https://datalab.snu.ac.kr/data/SNeCT/pancan12_tensor.tar.gz) |
| Pathway    | Gene - Gene | 14,351 &times; 14,351 | 665,429 | [DOWN](https://datalab.snu.ac.kr/data/SNeCT/pathway_network.tar.gz) |
