# TEMP TEST LINE

This document summarizes the 15 papers used in the AutoNAS literature review.
All papers have corresponding PDFs in the `papers/` folder.

---

## Paper 1: International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease

- **Authors:** R. Detrano et al.
- **Year:** 1989
- **Source:** American Journal of Cardiology, vol. 64, pp. 304–310
- **PDF:** `01_Detrano_1989_CAD_Probability.pdf`

### Abstract
Early computer-based diagnostic system for coronary artery disease (CAD) using a probability algorithm. Tested across multiple international sites for reliability.

### Advantages
Simple, interpretable statistical model; introduced the standard Cleveland heart disease dataset used in hundreds of subsequent studies.

### Datasets
Cleveland Heart Disease Dataset (303 instances, 14 attributes)

### Disadvantages
Uses statistical probability only; cannot learn complex nonlinear patterns; limited to 14 features.

### Results
~77% accuracy on Cleveland dataset

### Relevance to AutoNAS
Foundational dataset and benchmark for heart disease prediction; the Cleveland dataset is used in AutoNAS evaluation.

---

## Paper 2: DARTS: Differentiable Architecture Search

- **Authors:** H. Liu, K. Simonyan, Y. Yang
- **Year:** 2019
- **Source:** ICLR 2019
- **PDF:** `02_Liu_2019_DARTS.pdf`

### Abstract
Proposes differentiable NAS by relaxing the discrete search space to be continuous, enabling gradient-based optimization. Reduces search cost to 1.5 GPU-days.

### Advantages
Orders of magnitude faster than RL/evolution-based NAS; elegant continuous relaxation; competitive results.

### Datasets
CIFAR-10, ImageNet, Penn Treebank

### Disadvantages
Prone to collapse (skip connections); memory-intensive; discretization gap.

### Results
CIFAR-10: 2.76% error; ImageNet: 26.7% top-1 error; 1.5 GPU-days search cost

### Relevance to AutoNAS
Core NAS technique; DARTS' differentiable search directly inspires AutoNAS's architecture optimization strategy.

---

## Paper 3: Heart Disease Prediction Using Machine Learning Techniques: A Survey

- **Authors:** V. V. Ramalingam, A. Dandapath, M. K. Raja
- **Year:** 2018
- **Source:** Int. J. Eng. Technol., vol. 7, no. 2.8, pp. 684–687
- **PDF:** `03_Ramalingam_2018_HeartSurvey.pdf`

### Abstract
Comprehensive survey of ML techniques (SVM, KNN, NB, DT, RF, ANN) for heart disease prediction. Compares performance on Cleveland dataset.

### Advantages
Thorough comparison of multiple ML algorithms; identifies ensemble methods as superior; useful landscape overview.

### Datasets
Cleveland Heart Disease Dataset, Statlog Heart Disease Dataset

### Disadvantages
Survey only; no novel method proposed; does not cover deep learning or NAS.

### Results
Reports best surveyed accuracies: RF ~90%, NB ~87%, SVM ~86%

### Relevance to AutoNAS
Establishes traditional ML baselines that AutoNAS aims to surpass via automated neural architecture search.

---

## Paper 4: Neural Architecture Search with Reinforcement Learning

- **Authors:** B. Zoph, Q. V. Le
- **Year:** 2017
- **Source:** ICLR 2017
- **PDF:** `04_Zoph_2017_NAS_RL.pdf`

### Abstract
Pioneering work using an RNN controller trained with RL (REINFORCE) to generate neural network architectures. First to demonstrate NAS feasibility.

### Advantages
First successful NAS method; demonstrated that RL can discover architectures rivaling human-designed ones.

### Datasets
CIFAR-10, Penn Treebank

### Disadvantages
Extremely expensive (800 GPUs, 28 days); not scalable; search space limited.

### Results
CIFAR-10: 3.65% error; PTB: 62.4 perplexity

### Relevance to AutoNAS
Foundational NAS paper; AutoNAS builds on this concept of automated architecture discovery.

---

## Paper 5: Neural Architecture Search: A Survey

- **Authors:** T. Elsken, J. H. Metzen, F. Hutter
- **Year:** 2019
- **Source:** J. Mach. Learn. Res., vol. 20, pp. 1–21
- **PDF:** `05_Elsken_2019_NAS_Survey.pdf`

### Abstract
Comprehensive survey categorizing NAS by search space, search strategy, and performance estimation. Covers RL, evolution, gradient, and one-shot methods.

### Advantages
Definitive NAS survey; clear taxonomy; covers all major NAS approaches up to 2019.

### Datasets
Survey paper — reviews results on CIFAR-10, ImageNet, and others

### Disadvantages
Survey only; does not propose new methods; does not cover NAS for tabular/medical data.

### Results
Summarizes SOTA: best CIFAR-10 ~2.1% error; identifies key trends toward efficiency

### Relevance to AutoNAS
Provides the theoretical framework and taxonomy that AutoNAS's design follows.

---

## Paper 6: TabNet: Attentive Interpretable Tabular Learning

- **Authors:** S. Ö. Arık, T. Pfister
- **Year:** 2021
- **Source:** AAAI 2021
- **PDF:** `06_Arik_2021_TabNet.pdf`

### Abstract
Deep learning architecture for tabular data using sequential attention for feature selection. Provides interpretability and outperforms tree-based methods.

### Advantages
Designed for tabular data; interpretable feature selection; outperforms XGBoost on multiple benchmarks; supports self-supervised pre-training.

### Datasets
Forest Cover Type, Poker Hand, Sarcos, Higgs Boson, UCI tabular datasets

### Disadvantages
Requires careful hyperparameter tuning; slower training than GBDTs; not tested on medical data in original paper.

### Results
Forest Cover Type: 96.99% (vs 96.35% XGBoost); matches or outperforms GBDT on 6/6 datasets

### Relevance to AutoNAS
Demonstrates DL can excel on tabular data like heart disease datasets; AutoNAS extends this by automatically searching for optimal tabular architectures.

---

## Paper 7: AutoML: A Survey of the State-of-the-Art

- **Authors:** X. He, K. Zhao, X. Chu
- **Year:** 2021
- **Source:** Knowl.-Based Syst., vol. 212, p. 106622
- **PDF:** `07_He_2021_AutoML_Survey.pdf`

### Abstract
Comprehensive AutoML survey covering data preparation, feature engineering, HPO, and NAS. Compares NAS algorithms on CIFAR-10 and ImageNet.

### Advantages
Most complete AutoML survey; covers full pipeline; detailed NAS algorithm comparison; discusses resource-aware NAS.

### Datasets
Survey — reviews CIFAR-10, CIFAR-100, ImageNet, PTB results

### Disadvantages
Survey only; limited coverage of AutoML for healthcare; published 2021.

### Results
Summarizes SOTA: CIFAR-10 best 2.13% error; notes trend toward <1 GPU-day search cost

### Relevance to AutoNAS
Provides the methodological foundation for AutoNAS; our system implements key AutoML concepts (NAS + HPO) for heart disease prediction.

---

## Paper 8: Comprehensive Evaluation and Performance Analysis of Machine Learning in Heart Disease Prediction

- **Authors:** H. A. Al-Alshaikh et al.
- **Year:** 2024
- **Source:** Sci. Rep. (Nature), vol. 14, p. 9819
- **PDF:** `08_AlAlshaikh_2024_HeartML.pdf`

### Abstract
Hybrid feature selection (GA + RFEM) with class imbalance handling (USCOM). Evaluates RF, SVM, KNN, LR, DT, XGBoost, AdaBoost on Cleveland dataset.

### Advantages
Published in Nature Sci. Reports; novel GA+RFEM feature selection; addresses class imbalance; achieves 97.57% accuracy.

### Datasets
Cleveland Heart Disease Dataset (303 instances, 14 attributes)

### Disadvantages
No deep learning or NAS; limited to one dataset; manually designed pipeline.

### Results
RF with GA+RFEM: 97.57%; XGBoost: 96.52%; SVM: 95.65%

### Relevance to AutoNAS
State-of-the-art ML baselines on Cleveland dataset; AutoNAS aims to match or exceed these results through automated architecture search.

---

## Paper 9: ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware

- **Authors:** H. Cai, L. Zhu, S. Han
- **Year:** 2019
- **Source:** ICLR 2019
- **PDF:** `09_Cai_2019_ProxylessNAS.pdf`

### Abstract
Directly searches architectures on the target task without proxy. Uses path-level binarization for memory efficiency. Supports hardware-aware NAS.

### Advantages
Eliminates proxy tasks; memory-efficient; hardware-aware; only 200 GPU-hours on ImageNet (vs 48,000 for NASNet).

### Datasets
CIFAR-10, ImageNet

### Disadvantages
Binarization introduces gradient noise; search space manually defined; not tested on medical data.

### Results
ImageNet: 75.1% top-1; CIFAR-10: 2.08% error; 200 GPU-hours search cost

### Relevance to AutoNAS
Demonstrates practical NAS on target tasks without proxies; AutoNAS applies this principle by searching directly on the heart disease dataset.

---

## Paper 10: Regularized Evolution for Image Classifier Architecture Search

- **Authors:** E. Real, A. Aggarwal, Y. Huang, Q. V. Le
- **Year:** 2019
- **Source:** AAAI 2019, vol. 33, pp. 4780–4789
- **PDF:** `10_Real_2019_Regularized_Evolution.pdf`

### Abstract
Uses regularized evolution (aging mechanism) for NAS. Discovers AmoebaNet architectures competitive with RL-based NAS at lower cost.

### Advantages
Simple evolutionary approach; aging prevents stagnation; AmoebaNet achieves SOTA; more robust than RL-based search.

### Datasets
CIFAR-10, ImageNet

### Disadvantages
Still computationally expensive (3150 GPU-days); designed for image classifiers; limited to cell-based search space.

### Results
CIFAR-10: 2.13% error (AmoebaNet-A); ImageNet: 24.3% top-1 error

### Relevance to AutoNAS
Evolutionary NAS approach; AutoNAS can incorporate evolution-based search strategies alongside gradient-based methods.

---

## Paper 11: Efficient Neural Architecture Search via Parameter Sharing

- **Authors:** H. Pham, M. Y. Guan, B. Zoph, Q. V. Le, J. Dean
- **Year:** 2018
- **Source:** ICML 2018, pp. 4095–4104
- **PDF:** `11_Pham_2018_ENAS.pdf`

### Abstract
Proposes ENAS — shares parameters across child models via a single DAG. Reduces NAS cost by 1000× compared to original NAS.

### Advantages
1000× faster than standard NAS; parameter sharing is elegant and efficient; strong results on CIFAR-10 and PTB.

### Datasets
CIFAR-10, Penn Treebank

### Disadvantages
Shared parameters may bias search; controller still uses RL; limited search space.

### Results
CIFAR-10: 2.89% error; PTB: 55.8 perplexity; 0.5 GPU-days search cost

### Relevance to AutoNAS
Key efficiency technique; AutoNAS uses parameter sharing concepts to make architecture search feasible on modest hardware.

---

## Paper 12: SNAS: Stochastic Neural Architecture Search

- **Authors:** S. Xie, H. Zheng, C. Liu, L. Lin
- **Year:** 2019
- **Source:** ICLR 2019
- **PDF:** `12_Xie_2019_SNAS.pdf`

### Abstract
Reformulates NAS as optimization of a joint distribution using Gumbel-Softmax. Trains architecture and weights simultaneously via backpropagation.

### Advantages
End-to-end differentiable; theoretically principled; no separate controller needed; competitive results.

### Datasets
CIFAR-10, ImageNet

### Disadvantages
Gumbel-Softmax may not converge to discrete solutions cleanly; temperature tuning needed.

### Results
CIFAR-10: 2.85% error; ImageNet: 27.3% top-1 error; 1.5 GPU-days

### Relevance to AutoNAS
Alternative differentiable NAS approach; AutoNAS's search strategy draws from both DARTS and SNAS methodologies.

---

## Paper 13: NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search

- **Authors:** X. Dong, Y. Yang
- **Year:** 2020
- **Source:** ICLR 2020
- **PDF:** `13_Dong_2020_NASBench201.pdf`

### Abstract
Unified NAS benchmark with fixed cell-based search space (15,625 architectures) pre-evaluated on 3 datasets. Enables fair comparison of NAS algorithms.

### Advantages
Reproducible benchmark; fair comparison; multiple datasets; open-source with full training logs.

### Datasets
CIFAR-10, CIFAR-100, ImageNet-16-120

### Disadvantages
Fixed search space limits novelty; only cell-based; not designed for tabular/medical data.

### Results
Best architecture: 94.37% on CIFAR-10; reveals random search is surprisingly strong baseline

### Relevance to AutoNAS
Provides benchmarking methodology; AutoNAS builds a similar evaluation framework for heart disease models.

---

## Paper 14: Prediction of Heart Disease Using Classification Algorithms

- **Authors:** H. D. Masethe, M. A. Masethe
- **Year:** 2014
- **Source:** Proc. WCECS 2014, vol. II, pp. 809–812
- **PDF:** `14_Masethe_2014_Heart_Classification.pdf`

### Abstract
Compares J48, Naïve Bayes, SIMPLE CART, and other classification algorithms for heart disease prediction on Cleveland dataset.

### Advantages
Early comparison study; practical clinical focus; identifies J48 decision tree as best performer.

### Datasets
Cleveland Heart Disease Dataset (UCI)

### Disadvantages
Small dataset; limited algorithms tested; no deep learning; no feature selection optimization.

### Results
J48 Decision Tree: 99.07% (likely overfitted); Naïve Bayes: 85.03%

### Relevance to AutoNAS
Early heart disease ML baseline; AutoNAS demonstrates that automated deep learning can improve upon simple classifier approaches.

---

## Paper 15: Association Rule Discovery with the Train and Test Approach for Heart Disease Prediction

- **Authors:** C. Ordonez
- **Year:** 2006
- **Source:** IEEE Trans. Inf. Technol. Biomed., vol. 10, no. 2, pp. 334–343
- **PDF:** `15_Ordonez_2006_Association_Rules.pdf`

### Abstract
Uses association rules with a train-and-test methodology for heart disease prediction. Proposes constraints to reduce the number of irrelevant rules.

### Advantages
Interpretable rule-based approach; addresses rule explosion problem; medically meaningful rules.

### Datasets
Cleveland Heart Disease Dataset (303 instances)

### Disadvantages
Rule-based methods have limited predictive power; cannot model complex nonlinear relationships; accuracy below modern ML.

### Results
Reduced rules from >100,000 to manageable set; competitive accuracy for its era

### Relevance to AutoNAS
Demonstrates interpretable heart disease prediction; AutoNAS complements this with high-accuracy deep learning while maintaining model comparison capability.

---


## Paper 16: Heart Disease Prediction using Machine Learning Algorithms

- **Authors:** Mandavalli Sathi Ekambareesh, Kantipudi Budda Vara Prasad
- **Year:** 2024
- **Source:** International Journal of Multidisciplinary Research in Science, Engineering and Technology (IJMRSET), Vol. 7, Issue 9
- **DOI:** 10.15680/IJMRSET.2024.0709023
- **PDF:** `16_Ekambareesh_2024_Heart_ML_Algorithms.pdf`

### Abstract
Survey-style study on heart disease prediction using classical ML methods (SVM, KNN, Naive Bayes, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting) on UCI heart disease data.

### Advantages
Covers multiple baseline algorithms and compares them using a common setup and metrics (accuracy, precision, recall, F1).

### Datasets
UCI Heart Disease dataset (14 selected features from 76 available).

### Disadvantages
Primarily classical ML and survey-oriented; no NAS or deep neural architecture optimization.

### Results
Best reported model: Gradient Boosting with 88.5% accuracy.

### Relevance to AutoNAS
Directly supports AutoNAS baseline benchmarking on the same heart disease problem; provides classical-ML reference points that AutoNAS is designed to outperform with automated architecture search.

---
