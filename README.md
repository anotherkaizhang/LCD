# A. GOALS

## A.1 Importance and objective

Biomedical research focuses on treatment and interventions, in which causality is the key for such analysis. We care about “why” various things happen, “how” to make proper changes to achieve our goals, and “what” good treatments or interventions are. That is, it is essential to find and make use of causal relations in this research. Popular machine learning protocols focus on association analysis and do not solve such causal problems that most clinicians are interested in. In biomedical domains we mainly have passively observed data, such as Single-nucleotide polymorphism (SNP) data, Electronic Health Record (EHR) data, and medical claims data, the number of variables can range into tens of thousands, and usually there is very limited background knowledge to reduce the space of causal hypotheses. Such a domain certainly needs causal discovery methods, which aim to find the underlying causal relations from observational data with very little or no background information [1], and the abundance of data makes it possible. Furthermore, unlike traditional statistical analysis, causal analysis in biomedical discovery usually deals with large-scale data, while most methods for causal discovery, in their original forms, are computationally expensive. The field will benefit from reliable and computationally efficient methods for causal discovery and their smart and parallel implementations in widely-used programming languages. This will enable practitioners to conveniently perform large-scale causal analysis, develop improved or domain-specific causal discovery methods and, as a consequence, shift paradigms in biomedical discovery from association and prediction to causal analysis. 

Based on our experience with the well-known causal discovery package, **Tetrad**, we are planning on developing a new open-source package based on Python, for Large-scale Causal Discovery (LCD), that can correctly handle different types of data, causal relations, and practical issues (e.g., discrete/continuous data, temporal/non-temporal data, linear/nonlinear relations, and the presence of confounders and measurement error). To the best of our knowledge, currently there is no Python software package that includes most of the causal discovery algorithms that have been shown to be powerful, let alone their efficient implementations. Particularly this package will focus on causal discovery for large-scale biomedical data via GPU acceleration. Benefit of the GPU acceleration in causal discovery is analogous to the success of deep neural network models due to the introduction of GPU. Mature GPU interface in Python (e.g., PyTorch) can significantly push the fronts of computationally intensive tasks. 

## A.2 Existing open-source software for causal discovery

There are now three well-known open-source packages related to causal discovery: the Java-based package Tetrad, and R packages  PCALG and BNLEARN. One common limitation of these packages is that some state-of-the-art causal discovery algorithms are missing, such as those that can handle nonlinear causal mechanisms, non-Gaussian distributions, or data with confounders. In addition, none of these packages are scalable to the size of big biomedical data. 

We briefly analyze the pros and cons of each package. Tetrad includes well-known traditional causal discovery algorithms, such as PC [1], FCI [1], Greedy Equivalence Search (GES;  [2]), and their variants. It also includes several state-of-the-art algorithms, such as FASK [3], IMaGES [4], and FOFC [5]. Tetrad provides an interface for users to choose algorithms, set parameters, and visualize the output graph. However, the implemented algorithms mainly consider linear causal relationships. It is implemented in Java, which is less widely used in academia, leading to limited portability and expandability, and it is not straightforward to modify the source code for new developments. Tetrad does not have GPU hardware acceleration.

**PCALG** is open-source R package, including some well-known causal discovery algorithms, including the method based on the Linear, Non-Gaussian, Acyclic Model (LiNGAM; [6]). But the involved (conditional) independence tests only support Gaussian or discrete data, and many state-of-the-art causal discovery algorithms are missing. **BNLEARN** is an open-source R package for learning the graph structure of Bayesian networks, estimate their parameters, and perform certain types of inference. It includes a relatively complete list of algorithms in learning Markov equivalence classes and Markov blankets. However, algorithms that can uniquely learn the full causal graph are missing in the package.  Recently, another package, **Causal Discovery Toolbox** (CDT), has been developed. It is an open-source “partial” Python package, including some of existing causal discovery algorithms, but most of which are just directly calling the R implementation in ‘BNLEARN’ or ‘PCALG’ through R-Python bridge, which makes it computational less efficient. It has poor GPU support and can usually handle only small graphs. 

Note that some Python packages for causal analysis such as Causallib (IBM) and doWhy (Microsoft) are for causal estimation (estimation of causal effects) [7] but not causal discovery.

## B.1 Software development plan

### B.1.1 Python implementation of fundamental causal discovery methods 

We will provide Python open-source implementations of all those causal discovery methods that have asymptotic correctness guarantees and have been demonstrated to be effective, including PC, FCI, GES, and recently proposed methods based on functional causal models including LiNGAM [6], additive noise model [8], the post-nonlinear causal model [9], causal discovery from nonstationary or heterogeneous data [10], and causal discovery under measurement error [11] and missing values [12], as well as kernel-based conditional independence test (KCI-test; [13]) and other tests of conditional independence, as conditional independence test is an essential component of traditional conditional independence-based causal discovery. For reviews of such methods, one may see [14] [15][16].

### B.1.2 Acceleration by strategic conditional independence test and GPU parallelization. 

On top of the basic framework above, we will develop and implement a new GPU-accelerated parallel causal discovery methods to handle large-scale biomedical data. For example, taking the PC algorithm into consideration, the computational complexity is at least polynomially in the number of nodes (i.e., variables), making it computational infeasible for large graphs unless we make sparsity assumptions on the graph. The current research on parallel PC algorithms [17,18] focuses on distributing computations onto multiple cpu/gpu cores; however, the total amount of computation is not essentially reduced. The fundamental super-exponential-growth feature would soon override the speed-up via leveraging multi-cpu/gpu cores--dealing with large non-sparsity graphs still impossible. We found that in the current algorithm there interestingly exists lots of redundant computations. An edge can be tested repeatedly before it is deleted, but if the algorithm is run reversely, it will be tested only once and we can show the same result is guaranteed. Together with this new strategy, we identified a new multi-pass solution to reduce worse-case factorial complexity to linear without losing the convergence guarantee. Our reverse strategy will be parallelized as batch computation with GPU. We will use PyTorch, a mature deep learning framework that allows fast and efficient tensor computation for conditional independence test. We will carefully manage CPU-GPU memory I/O to maximize memory usage and parallelization efficiency. As a preliminary result, we have implemented a GPU-accelerated parallelized version of PC (Fig. 2). We will further consider additional causal discovery methods including FCI and GES. 

### B.1.3 Acceleration by Divide-and-Conquer for CPU users

In spite of GPU’s speedup, some biomedical researchers’ personal computers may still prefer CPU to the GPU. In order for large-scale causal discovery on such traditional architecture, we  develop and implement “divide-and-conquer” schemes for causal discovery for improved computational and statistical efficiency. Even if the whole variable set is causally sufficient, they suffer the potential issue that the divided groups may have hidden direct common causes, which may result in wrong edges in the result. We proved that once each divided variable set contains a variable Vj and its Markov blanket, then any estimated causal relation that involves the given variable Vj is correct. Fortunately, it is relatively easy to estimate Markov blankets, thanks to the progress in variable selection and graphical model estimation in statistics and computer science. This scheme also allows parallelization on GPUs.

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
Figure 1. The proposed evolvement from Tetrad to LCD with consideration of scalability and compatibility with state-of-the-art acceleration and visualization tools

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
Figure 2. Preliminary results of GPU-accelerated causal discovery method. The speedup on a 100 node graph is 400x, and such gain is bigger with larger graphs and more samples.



## References

1. 	Spirtes P, Glymour C, Scheines R. Causation, Prediction, and Search [Internet]. 2001. doi:10.7551/mitpress/1754.001.0001
2. 	Chickering DM. Optimal Structure Identification With Greedy Search. J Mach Learn Res. 2002; doi:10.1162/153244303321897717
3. 	Sanchez-Romero R, Ramsey JD, Zhang K, Glymour MRK, Huang B, Glymour C. Estimating feedforward and feedback effective connections from fMRI time series: Assessments of statistical methods [Internet]. Network Neuroscience. 2019. pp. 274–306. doi:10.1162/netn_a_00061
4. 	Ramsey JD, Hanson SJ, Hanson C, Halchenko YO, Poldrack RA, Glymour C. Six problems for causal inference from fMRI [Internet]. NeuroImage. 2010. pp. 1545–1558. doi:10.1016/j.neuroimage.2009.08.065
5. 	Kummerfeld E, Ramsey J. Causal Clustering for 1-Factor Measurement Models [Internet]. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16. 2016. doi:10.1145/2939672.2939838
6. 	Shohei Shimizu, Patrik O. Hoyer, Aapo Hyvärinen, Antti Kerminen. A Linear Non-Gaussian Acyclic Model for Causal Discovery. J Mach Learn Res. 2006;7: 2003–2030.
7. 	Pearl J. Causality: Models, Reasoning, and Inference. Cambridge University Press; 2000.
8. 	Patrik O. Hoyer Janzing, Dominik Mooij, Joris M Peters, Jonas Scholkopf, Bernhard. Nonlinear causal discovery with additive noise models. Advances in Neural Information Processing Systems 21. 2009;
9. 	Kun Zhang AH. On the identifiability of the post-nonlinear causal model. UAI ’09 Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. pp. 647–655.
10. Zhang K, Huang B, Zhang J, Glymour C, Schölkopf B. Causal Discovery from Nonstationary/Heterogeneous Data: Skeleton Estimation and Orientation Determination [Internet]. Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence. 2017. doi:10.24963/ijcai.2017/187
11. Zhang, K., Gong, M., Ramsey, J., Batmanghelich, K., Spirtes, P., and Glymour C. Causal Discovery with Linear Non-Gaussian Models under Measurement Error: Structural Identifiability Results. Proc Conference on Uncertainty in Artificial Intelligence (UAI’18). 2018.
12. Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellström, H., Glymour, C., and Zhang, K. Causal discovery in the presence of missing data. AISTATS 2019. 2019.
13. Kun Zhang Jonas Peters Dominik Janzing Bernhard Schölkopf. Kernel-based conditional independence test and application in causal discovery. UAI’11 Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence. 2011.
14. Glymour C, Zhang K, Spirtes P. Review of Causal Discovery Methods Based on Graphical Models. Front Genet. 2019;10: 524.
15. Spirtes P, Zhang K. Causal discovery and inference: concepts and recent methodological advances [Internet]. Applied Informatics. 2016. doi:10.1186/s40535-016-0018-x
16. Eberhardt F. Introduction to the foundations of causal discovery. Int J Data Sci Anal. 2017;3: 81–91.
17. Le T, Hoang T, Li J, Liu L, Liu H, Hu S. A fast PC algorithm for high dimensional causal discovery with multi-core PCs [Internet]. IEEE/ACM Transactions on Computational Biology and Bioinformatics. 2018. pp. 1–1. doi:10.1109/tcbb.2016.2591526
18. Behrooz Zare, Foad Jafarinejad, Matin Hashemi, Saber Salehkaleybar. cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU. In: arxiv.org [Internet]. Available: https://arxiv.org/abs/1812.08491





