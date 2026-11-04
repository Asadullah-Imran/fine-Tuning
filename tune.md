For your MUSE-GCN (Multi-omics Unified Spatial Embedding) model, I recommend conducting a systematic optimization study rather than simply trying different learning rates until you get a better ARI.

Your goal should be to answer three research questions:
1. Hyperparameter optimization: Which parameter settings produce the best domain identification performance?
2. Architectural optimization: Can changes to the GCN architecture, graph construction, or multi-omics fusion improve performance?
3. Robustness and generalization: Does the model consistently perform well across different datasets, seeds, and biological tissues?
These are three different experiments, and separating them will make your research much stronger.
1. What can you tune in MUSE-GCN?
Based on your existing architecture, I would divide the tuning into six categories.
A. Training hyperparameters
Start here
Parameter	What to experiment with
Learning rate	0.0001, 0.0005, 0.001, 0.005
Weight decay	0, 1e-5, 1e-4, 1e-3
Epochs	100, 200, 300, 500
Hidden dimensions	32, 64, 128, 256
Dropout	0, 0.1, 0.3, 0.5
Optimizer	Adam, AdamW

B. Loss-function weights
High priority
Your existing model has multiple objectives. Their relative importance can substantially affect the learned embedding.
Parameter	What to experiment with
β (beta)	Reconstruction-related weight
γ (gamma)	Spatial regularization-related weight
δ (delta)	Additional objective weight
κ (kappa)	Additional objective weight

You currently use β=25, γ=10, δ=1, κ=1. Test these systematically rather than changing all four simultaneously.

C. Graph construction
Important for spatial omics
Parameter	What to experiment with
Spatial k	Number of spatial neighbors
Feature k	Number of feature-similarity neighbors
Distance metric	Euclidean, cosine
Graph type	kNN, radius-based
Graph normalization	Different normalization strategies
Self-loops	Enabled vs disabled

D. Multi-omics fusion
Parameter	What to experiment with
Fusion	Concatenation, weighted fusion
Modality balance	Equal vs learned weighting
Embedding dimension	32, 64, 128
Fusion projection	Linear vs nonlinear
Modality-specific embeddings	Different dimensions vs shared dimensions

E. GNN architecture
Parameter	What to experiment with
GNN type	GCN, GAT, GraphSAGE, GIN
Number of layers	1, 2, 3
Attention heads	2, 4, 8 (for attention models)
Activation	ReLU, ELU, GELU
Residual connections	Enabled vs disabled

F. Clustering and inference
Parameter	What to experiment with
KMeans initialization	k-means++, random
Number of restarts	10, 20, 50
Number of clusters	Known ground truth vs estimated
Random seed	42, 1234, 2024, additional seeds


One important distinction: not everything above is a conventional hyperparameter. For example, changing GCN to GAT is an architectural change, while changing spatial k is a graph-construction choice. In your paper, report them as separate experimental studies.
2. My recommended optimization strategy
I would not start with a huge grid search. Your model has several interacting components, and testing every combination would become computationally expensive.

Instead, use a four-stage optimization pipeline.
1. Stage 1 — Baseline and reproducibility
   
   Run the existing MUSE-GCN configuration without modifications. Record all losses, metrics, and runtime.
2. Stage 2 — Hyperparameter sensitivity
   
   Determine which parameters actually influence ARI, NMI, AMI, and stability.
3. Stage 3 — Automated optimization
   
   Use Random Search followed by Bayesian Optimization to find promising parameter combinations.
4. Stage 4 — Architecture and robustness experiments
   
   Test graph construction, fusion, GNN architecture, and generalization across datasets.

Let's discuss how to conduct each stage.
Stage 1: Establish your baseline
First, freeze your current architecture and configuration.

Your starting configuration, based on your previous implementation:
Component	Existing setting
GNN	4 independent one-layer GCN encoders
RNA input	PCA 60/100
Graphs	Spatial + feature similarity
Fusion	Linear fusion
Reconstruction	Modality-specific + joint
Clustering	Final-epoch KMeans
β	25
γ	10
δ	1
κ	1
Seeds	42, 1234, 2024

Run the same configuration on your six datasets.

Record:
- ARI
- NMI
- AMI
- Homogeneity
- V-measure
- Silhouette score
- Training loss
- Individual loss components
- Training time
- GPU/CPU memory usage
Create a baseline experiment table.
Dataset	ARI	NMI	AMI	Silhouette
Human Lymph A1	...	...	...	...
Human Lymph D1	...	...	...	...
Mouse Brain E11	...	...	...	...
Mouse Brain E13	...	...	...	...
Mouse Brain E15	...	...	...	...
Mouse Brain E18	...	...	...	...

Keep the baseline results unchanged. You will need them to establish whether optimization actually improves MUSE-GCN.
Stage 2: Hyperparameter sensitivity analysis
This is particularly important for your research paper.

You want to discover:

Which hyperparameters have the greatest influence on spatial domain identification?

I recommend starting with one-factor-at-a-time (OFAT) experiments.

For example, suppose you want to tune β.
Experiment	β	γ	δ	κ
Baseline	25	10	1	1
B1	5	10	1	1
B2	10	10	1	1
B3	15	10	1	1
B4	25	10	1	1
B5	50	10	1	1
B6	100	10	1	1

Only β changes. Everything else remains fixed.

Repeat this for γ, δ, and κ.

You can then visualize the results:
Illustrative hyperparameter sensitivity
Hypothetical ARI values showing how performance changes with β. Replace with actual MUSE-GCN measurements.


0.420.4550.490.5250.56510152550100














Illustrative data only; these are not experimental results from MUSE-GCN.

This helps you identify:
- Parameters that significantly affect performance.
- Parameters with relatively little influence.
- Parameters that have a narrow optimal range.
- Parameters that produce unstable results.
Important: OFAT is a screening method, not proof of the globally optimal configuration. It cannot reveal interactions between parameters. For example, β=25 might work best when γ=10 but not when γ=20.

That is why we need the next stage.
Stage 3: Automated hyperparameter optimization
This is where I recommend using Optuna.









Optuna

Python hyperparameter optimization framework
Official website

Optuna supports:
- Random sampling.
- Bayesian-style optimization using TPE.
- Pruning unpromising trials.
- Parameter importance analysis.
- Visualization of optimization history and parameter relationships.
Which optimization algorithm should you use?
Method	How it works	My recommendation
Grid Search	Tests predefined combinations	Small experiments
Random Search	Randomly samples combinations	Initial exploration
Bayesian Optimization (TPE)	Uses previous trials to propose better configurations	Recommended
Hyperband	Allocates resources to promising trials	Useful if training is expensive
Genetic Algorithms	Evolves parameter combinations	Optional advanced experiment

I would use:

Random Search → Optuna TPE → Final multi-seed evaluation.
Suggested search space
Here is an initial search space for your model. These are proposed experimental ranges, not established optimal values.
Parameter	Search range
Learning rate	1e-4 to 5e-3 (log scale)
Weight decay	1e-6 to 1e-3 (log scale)
Dropout	0.0 to 0.5
Hidden dimension	32, 64, 128, 256
β	5 to 100 (log scale)
γ	0.1 to 30 (log scale)
δ	0.1 to 10 (log scale)
κ	0.1 to 10 (log scale)
Epochs	100 to 500

Do not immediately optimize all these parameters together. First, use sensitivity analysis to eliminate parameters that do not meaningfully affect performance.
How Optuna works







tech.preferred.jp


The workflow is:
1. Optuna proposes a combination of hyperparameters.
2. MUSE-GCN trains using those values.
3. Your evaluation function calculates a validation score.
4. Optuna records the result.
5. The next trial uses information from previous trials.
6. After multiple trials, you obtain promising configurations.
An example objective function:
import optunadef objective(trial):    lr = trial.suggest_float(        "lr", 1e-4, 5e-3, log=True    )    beta = trial.suggest_float(        "beta", 5, 100, log=True    )    gamma = trial.suggest_float(        "gamma", 0.1, 30, log=True    )    dropout = trial.suggest_float(        "dropout", 0.0, 0.5    )    config = {        "lr": lr,        "beta": beta,        "gamma": gamma,        "dropout": dropout    }    model = train_muse_gcn(config)    validation_ari = evaluate(model)    return validation_aristudy = optuna.create_study(    direction="maximize",



This is a conceptual integration example; train_muse_gcn() and evaluate() must be adapted to your existing implementation.

A critical research consideration: Your MUSE-GCN is unsupervised. If you use ground-truth domain labels to choose hyperparameters, you must designate validation data for that purpose and reserve independent test datasets or sections for final evaluation. Do not optimize against the same ground-truth labels you later use to claim test performance.

For cross-dataset generalization, I would use some sections for tuning and hold out other sections or datasets for final evaluation.
3. Beyond hyperparameters: what else should you experiment with?
This is where I think you can get the most interesting research results.

You have a multi-omics model, not just a conventional GCN. Therefore, you should investigate whether the design decisions in MUSE-GCN are actually helping it learn better representations.

I recommend five additional studies.
Experiment A: Graph construction optimization
Very high priority
Your architecture uses multiple graphs:
- RNA spatial graph
- RNA feature-similarity graph
- Auxiliary omics spatial graph
- Auxiliary omics feature-similarity graph
You should investigate whether the graph construction method is optimal.

For example:
Experiment	Spatial k	Feature k
G1	4	4
G2	6	6
G3	8	8
G4	10	10
G5	12	12
G6	6	10
G7	10	6

Why is this important?

Imagine that RNA and protein have different feature distributions.

If you construct their feature graphs using the same number of neighbors, you may not necessarily obtain equally informative graph structures.

You could investigate:
- Should both modalities have the same k?
- Should RNA have a denser graph than the auxiliary modality?
- Does a sparse graph preserve biological boundaries better?
- Does a radius graph outperform kNN?
This is a particularly interesting experiment for multi-omics domain identification.
Experiment B: Modality contribution and fusion optimization
Very high priority
Your model's central idea is integrating two omics modalities.

You should investigate how much each modality contributes to the final embedding.

I would design these experiments:
Fusion ablation experiments
RNA

Only modality

Auxiliary

Only modality



Fusion alternatives
RNA embedding + Auxiliary embedding

Concatenation

RNA embedding + Auxiliary embedding

Weighted fusion

RNA embedding + Auxiliary embedding

Attention-based fusion

Conceptual experimental alternatives, not a description of additional modules already implemented in MUSE-GCN.

Suggested experiments:
ID	Configuration
F1	RNA only
F2	Auxiliary omics only
F3	Both modalities, concatenation
F4	Both modalities, weighted fusion
F5	Both modalities, attention fusion

You can also test the weighting parameter:
\[
Z_{\text{joint}}=\alpha Z_{\text{RNA}}+
(1-\alpha)Z_{\text{Aux}}
\]

Try:
\[
\alpha \in \{0.2,0.4,0.5,0.6,0.8\}
\]

This will help answer whether RNA or auxiliary omics contributes more to domain identification.

One important point: keep the embedding dimensions and downstream clustering procedure comparable across fusion variants.
Experiment C: GNN architecture comparison
You previously considered GCN, GAT, GraphSAGE, GIN, GATv2 and GPS.

I would make this a separate architecture benchmark.









1. GCN

Your existing baseline. Useful for comparing neighborhood information aggregation.












2. GAT

Learns different attention weights for neighboring spots.












3. GraphSAGE

Aggregates neighborhood information using different aggregation functions.












4. GIN

Tests a different neighborhood aggregation mechanism.












5. GATv2

An alternative attention mechanism with more expressive attention ranking.



My recommendation:

Start with GCN, GAT and GraphSAGE. Then consider GATv2 and GIN.

You should not jump directly to GPS because it introduces a more complex architecture and potentially a much higher computational cost.

Keep the remaining MUSE-GCN pipeline identical when comparing these architectures.

Also, compare parameter counts and training budgets. Otherwise, improvements might come from a larger model rather than a better aggregation mechanism.
Experiment D: Number of GCN layers
Your existing model uses four independent one-layer encoders.

You can investigate whether deeper encoders improve representation quality.
Experiment	Layers per encoder
L1	1
L2	2
L3	3
L4	4

Why investigate this?
- One layer aggregates immediate neighborhood information.
- Two layers aggregate information from a larger neighborhood.
- Three or more layers can introduce oversmoothing.
For spatial domain identification, deeper is not automatically better. Excessive propagation can blur boundaries between neighboring biological domains.

Measure both clustering performance and embedding smoothness.
Experiment E: Clustering optimization
Your final-epoch embedding is clustered using KMeans.

You should investigate whether the clustering algorithm itself affects the results.
Method	Purpose
KMeans	Existing baseline
Gaussian Mixture Model	Probabilistic clustering
Leiden	Graph-community detection
Louvain	Modularity-based clustering
Spectral clustering	Graph-based clustering

For fair evaluation, use the same embedding and comparable cluster-count assumptions.

For your current benchmark, where ground-truth domain counts are known, you can use the known number of clusters as an evaluation setting. But also consider testing whether your method can estimate the number of domains without ground-truth information.

4. Robustness and generalization experiments
I would consider this essential for a publishable spatial multi-omics paper.

A model that achieves ARI 0.60 with one seed but ARI 0.35 with another seed is not necessarily reliable.

Your current seeds are:
seeds = [42, 1234, 2024]



For the final evaluation, increase this to at least five, preferably ten independent seeds where computationally feasible.

Calculate:
\[
\text{Mean ARI}=\frac{1}{n}\sum_{i=1}^{n}ARI_i
\]

\[
\text{SD}=
\sqrt{\frac{\sum_{i=1}^{n}(ARI_i-\overline{ARI})^2}{n-1}}
\]

Report:
\[
ARI = \text{Mean} \pm \text{SD}
\]

I would additionally conduct these experiments:
Experiment	What it investigates
Seed sensitivity	Stability across random initializations
Noise robustness	Performance under feature perturbations
Missing omics	Performance when some modality information is missing
Dropout robustness	Sensitivity to sparse expression measurements
Cross-section generalization	Whether learned configurations transfer between sections
Cross-tissue generalization	Whether the approach works beyond one tissue
Runtime analysis	Computational scalability

For your Human Lymph and Mouse Brain datasets, cross-section and cross-tissue evaluation could be especially informative.

For example, tune using Human Lymph A1 and evaluate on a held-out section. Then investigate how well the selected configuration transfers to the Mouse Brain datasets.

Do not claim that a configuration generalizes simply because it was optimized independently on every test dataset.
5. What does the literature suggest?
There are some particularly relevant papers you should study when designing these experiments.
Paper	Why it matters
MMSpa (2026)	Performs parameter sensitivity analysis on model and spatial graph parameters.  PLOS Biology
SpaMOAL (2026)	Relevant multi-omics architecture and loss-weight sensitivity analysis.  PLOS Biology
RMGAT (2026)	Particularly interesting systematic component ablation and multi-seed evaluation.  Springer Nature Link
STMGCN (2023)	Compares multi-view fusion and attention mechanisms.  Oxford Academic
Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges (2021)	Helps you design a rigorous optimization process.  arXiv

Two papers I would prioritize reading:
1. MMSpa: for how to conduct parameter sensitivity analysis.
2. RMGAT: for how to report component ablations and statistical evidence.
6. How I would organize your actual experiments
You don't need to perform every experiment immediately.

Here is the order I recommend for your team.
MUSE-GCN experimental roadmap
Suggested sequence, not fixed deadlines.
1. Experiment 1 — Baseline reproduction
   - Existing architecture
   - Existing parameters
   - All six datasets
   - Three initial seeds
   Output: Baseline results table.
2. Experiment 2 — Loss-weight sensitivity
   - β, γ, δ, κ
   - OFAT experiments
   - Loss curves and ARI sensitivity plots
   Output: Loss-weight sensitivity figure.
3. Experiment 3 — Training hyperparameters
   - Learning rate
   - Weight decay
   - Dropout
   - Hidden dimensions
   - Epochs
   Output: Optimal training configuration.
4. Experiment 4 — Graph sensitivity
   - Spatial k
   - Feature k
   - Graph normalization
   - Graph construction alternatives
   Output: Graph sensitivity analysis.
5. Experiment 5 — Automated optimization
   - Optuna
   - Random Search
   - TPE
   - Best configurations
   Output: Optimized MUSE-GCN.
6. Experiment 6 — Architecture and fusion ablations
   - GCN vs GAT vs GraphSAGE
   - One vs two vs three layers
   - RNA vs auxiliary vs both
   - Fusion alternatives
   Output: Architecture comparison and ablation tables.
7. Experiment 7 — Final validation
   - Five to ten seeds
   - Independent held-out datasets
   - Statistical comparisons
   - Runtime and robustness
   Output: Final experimental results for your paper.

7. What should your experiment tracking system look like?
I strongly recommend creating a separate experiment directory rather than continuously modifying your original training script.

For example:
MUSE-GCN/
│
├── configs/
│   ├── baseline.yaml
│   ├── hyperparameter_search.yaml
│   ├── graph_search.yaml
│   └── architecture.yaml
│
├── experiments/
│   ├── baseline/
│   ├── loss_sensitivity/
│   ├── training_hpo/
│   ├── graph_ablation/
│   ├── fusion_ablation/
│   ├── architecture_ablation/
│   └── robustness/
│
├── optimization/
│   ├── objective.py
│   ├── search_space.py
│   └── optuna_study.py
│
├── evaluation/
│   ├── metrics.py
│   ├── statistical_tests.py
│   └── visualization.py
│
├── results/
│   ├── experiment_logs/
│   ├── tables/
│   └── figures/
│
└── train.py


For experiment tracking, consider:
- MLflow — Documentation.
- Weights & Biases — Documentation.
- Optuna — Documentation.
Record every experiment's configuration, dataset, seed, code version, metrics, and runtime.

One implementation detail: save the best validation checkpoint, not automatically the final epoch. Also, don't use ground-truth labels to choose the best epoch in an unsupervised experiment.
8. Important research considerations
Before starting, there are four things I would enforce:

1. Do not optimize ARI on your final test datasets.

Use a validation protocol for configuration selection. Keep independent test datasets untouched until the final evaluation.

2. Do not change several components simultaneously during ablation.

Otherwise, you cannot identify which component caused the improvement.

3. Compare against tuned baselines.

If MUSE-GCN is tuned extensively but competing methods use their default parameters, your comparison may be unfair. Give baselines reasonable, documented tuning budgets.

4. Examine loss-scale balance.

Your β=25 and γ=10 might look reasonable, but their effective contributions depend on the magnitude of their underlying losses. Log each raw loss and its weighted contribution separately. A weight of 25 does not necessarily mean that objective dominates.

9. My recommendation: prioritize these experiments
If I were supervising this MUSE-GCN research, I would prioritize the following:
Priority 1
Essential

Loss-weight sensitivity + learning-rate optimization.

Priority 2
Essential

Spatial and feature graph neighbor optimization.

Priority 3
Essential

Modality contribution and fusion ablation.

Priority 4
High value

GCN vs GAT vs GraphSAGE comparison.

Priority 5
High value

Multi-seed stability and cross-dataset generalization.

10. Let's make this specific to your actual codebase
I want to take this one step further rather than give you a generic hyperparameter-tuning plan.

I need to inspect your actual MUSE-GCN implementation to identify:
- Every tunable parameter.
- Which parameters are currently hardcoded.
- Which loss functions depend on β, γ, δ and κ.
- Which parameters are coupled.
- Which experiments can be automated.
- Which parts need architectural modifications.
Can you upload your current AriseSpatialGlue_4Encoder_1Layer_DEC.py file?

I can then prepare a MUSE-GCN-specific optimization specification, including an exact Optuna search space, experiment configurations, ablation plan, and implementation roadmap.

That way, we optimize your actual model rather than accidentally changing something that could compromise its original research contribution.