# Appendix H Hard Tasks Architecture

This document describes the class hierarchy and data flow for Appendix H Hard Tasks (Frontier Task Difficulty Prediction).

## Key Differences from Experiment A

| Aspect | Experiment A | Appendix H Hard Tasks |
|--------|--------------|--------------|
| **Holdout Strategy** | Tasks (5-fold CV) | Agents (pre/post-frontier split by date) |
| **Goal** | Predict difficulty of new tasks | Predict difficulty of frontier tasks |
| **Training Data** | Train-fold tasks only | All tasks, pre-frontier agents only |
| **Evaluation** | Per-fold AUC, all agents | Per-agent AUC, post-frontier agents |
| **IRT Training** | Retrain per fold | Single baseline IRT (cached) |
| **Primary Metric** | Pooled ROC-AUC | Mean Per-Agent AUC (scale-free) |

## Class Hierarchy & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CLI ARGUMENTS                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  compare_methods.py:                    threshold_sweep.py:                      │
│  --dataset swebench                     --datasets swebench terminalbench        │
│  --frontier_definitions zero_pre        --thresholds 0.0 0.05 0.10 ...           │
│  --cutoff_date 20250501                 --l2_weight 0.001                        │
│  --forecast_dates                       --l2_residual 10.0                       │
│  --output_csv results.csv               --output_dir chris_output/threshold_sweep│
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────────────┐  ┌─────────────────────┐  ┌───────────────────────┐  │
│  │ responses.jsonl       │  │ embeddings.npz      │  │ llm_features.csv      │  │
│  │ (agent × task matrix) │  │ (5120-dim vectors)  │  │ (9-10 features)       │  │
│  └───────────┬───────────┘  └──────────┬──────────┘  └───────────┬───────────┘  │
│              │                         │                         │               │
│  ┌───────────┴───────────┐  ┌──────────┴──────────┐  ┌───────────┴───────────┐  │
│  │ Oracle IRT:           │  │ Baseline IRT:       │  │ agent_dates:          │  │
│  │ items.csv (β)         │  │ items.csv (β)       │  │ Dict[agent, YYYYMMDD] │  │
│  │ abilities.csv (θ)     │  │ abilities.csv (θ)   │  │ (from config)         │  │
│  └───────────────────────┘  └─────────────────────┘  └───────────────────────┘  │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DATASET CONFIGURATION                                   │
│                         (shared/config_base.py)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DatasetConfig (ABC)                                                             │
│  ├── responses_path: Path                                                        │
│  ├── oracle_irt_path: Path                                                       │
│  ├── oracle_abilities_path: Path                                                 │
│  ├── embeddings_path: Optional[Path]                                             │
│  ├── llm_judge_path: Optional[Path]                                              │
│  ├── cutoff_date: str (YYYYMMDD)                                                 │
│  ├── pre_threshold: float = 0.1                                                  │
│  ├── post_threshold: float = 0.1                                                 │
│  │                                                                               │
│  │  Lazy-loaded properties (computed on first access):                           │
│  ├── responses → Dict[agent, Dict[task, response]]                               │
│  ├── all_agents → List[str]                                                      │
│  ├── all_task_ids → List[str]                                                    │
│  ├── agent_dates → Dict[agent, str]                                              │
│  │                                                                               │
│  │  Abstract methods (implemented per dataset):                                  │
│  ├── name → str                                                                  │
│  └── get_agent_dates(agents) → Dict[agent, str]                                  │
│      │                                                                           │
│      ├──► SWEBenchConfig (swebench/config.py)                                    │
│      ├──► SWEBenchProConfig (swebench_pro/config.py)                             │
│      ├──► TerminalBenchConfig (terminalbench/config.py)                          │
│      └──► GSOConfig (gso/config.py)                                              │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                       │
│                      (shared/data_preparation.py)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  load_and_prepare_data(args, config) → ExperimentData                            │
│                                                                                  │
│  Step 1: Load Oracle IRT                                                         │
│          oracle_items = pd.read_csv(oracle_irt_path)     # β for all tasks       │
│          oracle_abilities = pd.read_csv(oracle_abilities_path)  # θ for all      │
│                                                                                  │
│  Step 2: Load responses (lazy from config.responses)                             │
│          {agent_id: {task_id: 0|1 or {"successes": k, "trials": n}}}             │
│                                                                                  │
│  Step 3: Split agents by cutoff date                                             │
│          split_agents_by_dates(agents, agent_dates, cutoff_date)                 │
│          → (pre_frontier_agents, post_frontier_agents)                           │
│                                                                                  │
│  Step 4: Load/train Baseline IRT (cached by hash of inputs)                      │
│          get_or_train_baseline_irt(responses_path, pre_agents, cutoff_date)      │
│          → (baseline_items_df, baseline_abilities_df)                            │
│          Cache key: SHA256(responses_filename + sorted_agents + cutoff)[:12]     │
│                                                                                  │
│  Step 5: Identify frontier tasks (multiple definitions)                          │
│          ┌─────────────────────────────────────────────────────────────────┐     │
│          │ identify_frontier_tasks_zero_pre()   # 0% pre, >0% post         │     │
│          │ identify_frontier_tasks_passrate()   # ≤10% pre, >10% post      │     │
│          │ identify_frontier_tasks_irt()        # P(solve)<0.5 all pre     │     │
│          │ identify_frontier_tasks_pre_only()   # ≤X% pre (for sweep)      │     │
│          └─────────────────────────────────────────────────────────────────┘     │
│          → frontier_tasks_by_def: Dict[def_name, List[task_id]]                  │
│                                                                                  │
│  Step 6: Identify anchor tasks for scale alignment                               │
│          identify_nontrivial_tasks(responses, pre, post, 0.1, 0.9)               │
│          → tasks with 10-90% pass rate in BOTH agent groups                      │
│                                                                                  │
│  Step 7: Filter eval agents (optional)                                           │
│          filter_eval_agents(post_agents, responses, frontier_tasks, percentile)  │
│          → remove bottom X% by frontier success rate                             │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ ExperimentData dataclass:                                                │     │
│  │ ├── config: DatasetConfig                                                │     │
│  │ ├── oracle_items: DataFrame         # β from full IRT (all agents)      │     │
│  │ ├── oracle_abilities: DataFrame     # θ from full IRT                   │     │
│  │ ├── baseline_items: DataFrame       # β from pre-frontier only IRT      │     │
│  │ ├── baseline_abilities: DataFrame   # θ from pre-frontier only IRT      │     │
│  │ ├── pre_frontier_agents: List[str]                                       │     │
│  │ ├── post_frontier_agents: List[str]                                      │     │
│  │ ├── train_task_ids: List[str]       # All tasks in baseline IRT         │     │
│  │ ├── frontier_tasks_by_def: Dict[str, List[str]]                          │     │
│  │ ├── anchor_task_ids: List[str]                                           │     │
│  │ ├── cutoff_date: str                                                     │     │
│  │ ├── train_responses: Dict           # Pre-frontier agents only          │     │
│  │ ├── baseline_ground_truth_b: np.ndarray                                  │     │
│  │ ├── eval_agents_by_def: Dict[str, List[str]]                             │     │
│  │ └── filtering_stats_by_def: Dict[str, AgentFilteringStats]               │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FEATURE SOURCES                                       │
│                   (experiment_new_tasks/feature_source.py)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TaskFeatureSource (ABC)                                                         │
│  ├── get_features(task_ids) → np.ndarray (n_tasks, feature_dim)                  │
│  ├── task_ids: List[str]                                                         │
│  ├── feature_dim: int                                                            │
│  ├── feature_names: Optional[List[str]]                                          │
│  │                                                                               │
│  │   ┌─────────────────────────┐     ┌─────────────────────────┐                │
│  ├──►│ EmbeddingFeatureSource  │     │   CSVFeatureSource      │                │
│  │   │ (loads .npz files)      │     │   (loads CSV columns)   │                │
│  │   │ 5120-dim embeddings     │     │   9-10 LLM judge feats  │                │
│  │   └───────────┬─────────────┘     └───────────┬─────────────┘                │
│  │               │                               │                               │
│  │               ▼                               ▼                               │
│  │   ┌─────────────────────────────────────────────────────────┐                │
│  │   │              RegularizedFeatureSource                    │                │
│  │   │   (wraps source + alpha regularization strength)         │                │
│  │   │   Used for per-group regularization in GroupedRidge      │                │
│  │   └───────────────────────┬─────────────────────────────────┘                │
│  │                           │                                                   │
│  │                           ▼                                                   │
│  │   ┌─────────────────────────────────────────────────────────┐                │
│  └──►│              GroupedFeatureSource                        │                │
│      │   (combines multiple RegularizedFeatureSources)          │                │
│      │   e.g., [Embeddings(α=10000), LLM(α=100)]                │                │
│      └─────────────────────────────────────────────────────────┘                │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PREDICTORS                                          │
│             (experiment_new_tasks/feature_predictor.py +                         │
│              experiment_appendix_h_hard_tasks/shared/prediction_methods.py)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ FeatureBasedPredictor (experiment_new_tasks)                             │    │
│  │ ├── source: TaskFeatureSource                                            │    │
│  │ ├── _scaler: StandardScaler                                              │    │
│  │ ├── _model: RidgeCV                                                      │    │
│  │ ├── fit(task_ids, ground_truth_b)                                        │    │
│  │ └── predict(task_ids) → Dict[task_id, β̂]                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ GroupedRidgePredictor (experiment_new_tasks)                             │    │
│  │ ├── source: GroupedFeatureSource                                         │    │
│  │ ├── Grid search over per-source alphas                                   │    │
│  │ ├── Feature scaling: StandardScaler → per-group 1/sqrt(α) scaling        │    │
│  │ ├── fit(task_ids, ground_truth_b)                                        │    │
│  │ └── predict(task_ids) → Dict[task_id, β̂]                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ FeatureIRTPredictor (experiment_appendix_h_hard_tasks/shared)                                │    │
│  │ Joint IRT + feature learning - optimizes IRT likelihood directly         │    │
│  │                                                                          │    │
│  │ Model:                                                                   │    │
│  │   β_i = features[i] @ w + bias + residual[i]                             │    │
│  │   P(success) = sigmoid(θ_j - β_i)                                        │    │
│  │                                                                          │    │
│  │ Constructor:                                                             │    │
│  │ ├── source: TaskFeatureSource                                            │    │
│  │ ├── use_residuals: bool = False                                          │    │
│  │ ├── init_from_baseline: bool = False                                     │    │
│  │ ├── l2_weight: float = 0.01      # Reg on feature weights               │    │
│  │ ├── l2_residual: float = 10.0    # Reg on residuals (high=use features) │    │
│  │ ├── l2_ability: float = 0.01     # Reg on mean(θ)² for identifiability  │    │
│  │ ├── max_iter: int = 500                                                  │    │
│  │ └── device: str = "cpu"                                                  │    │
│  │                                                                          │    │
│  │ Methods:                                                                 │    │
│  │ ├── fit(task_ids, ground_truth_b, responses, baseline_abilities, ...)   │    │
│  │ │   IMPORTANT: responses must be pre-filtered to pre-frontier agents     │    │
│  │ ├── predict(task_ids) → Dict[task_id, β̂]                                 │    │
│  │ ├── learned_abilities → Dict[agent_id, θ]                                │    │
│  │ ├── feature_weights → Dict[feature_name, weight]                         │    │
│  │ └── get_baseline_init_diagnostics() → drift metrics                      │    │
│  │                                                                          │    │
│  │ Initialization modes:                                                    │    │
│  │ ├── Standard: Empirical init + Ridge warm-start for weights              │    │
│  │ └── Baseline-Init: Start from Baseline IRT solution, learn corrections   │    │
│  │                                                                          │    │
│  │ Loss:                                                                    │    │
│  │   -Σ_ij log P(y_ij | θ_j, β_i)   # Negative log-likelihood              │    │
│  │   + l2_weight × ||w||²           # Weight regularization                 │    │
│  │   + l2_residual × ||r||²         # Residual regularization               │    │
│  │   + l2_ability × mean(θ)²        # Identifiability constraint            │    │
│  │                                                                          │    │
│  │ Optimizer: L-BFGS with line search                                       │    │
│  │ Supports: Binary (Bernoulli) and Binomial responses                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION COLLECTION                                    │
│                      (shared/prediction_methods.py)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  raw_predictions: Dict[method_name, Dict[task_id, β]]                            │
│                                                                                  │
│  Built-in predictions (from IRT models):                                         │
│  ├── "Oracle (upper bound)": oracle_items["b"].to_dict()                         │
│  └── "Baseline IRT (pre-frontier only)": baseline_items["b"].to_dict()           │
│                                                                                  │
│  collect_ridge_predictions(feature_sources, train_tasks, ground_truth_b, all):   │
│  ├── For each source: FeatureBasedPredictor(source).fit().predict()              │
│  └── Returns: {"Embedding + Ridge": {...}, "LLM Judge + Ridge": {...}}           │
│                                                                                  │
│  collect_grouped_ridge_predictions(feature_sources, ...):                        │
│  ├── Pairwise and full combinations of sources                                   │
│  ├── GroupedRidgePredictor with per-source alpha grids                           │
│  └── Returns: {"Grouped Ridge (Embedding + LLM Judge)": {...}}                   │
│                                                                                  │
│  collect_feature_irt_predictions(feature_sources, ..., baseline_abilities):      │
│  ├── Grid search: l2_weight × l2_residual × init_from_baseline                   │
│  │   l2_grid = [0.001, 0.01, 0.1, 1.0, 10.0]                                     │
│  │   init_grid = [False, True] (if baseline_abilities available)                 │
│  ├── Select best config by AUC on frontier tasks                                 │
│  └── Returns: FeatureIRTResults                                                  │
│      ├── predictions: Dict[method_name, Dict[task_id, β]]                        │
│      ├── abilities: Dict[method_name, Dict[agent_id, θ]]                         │
│      ├── diagnostics: List[Dict] (grid search info)                              │
│      └── baseline_init_diagnostics: Dict (drift from baseline)                   │
│                                                                                  │
│  collect_sad_irt_predictions(sad_irt_beta_dir):  [OPTIONAL]                      │
│  ├── Load extracted SAD-IRT beta CSV files                                       │
│  └── Each file → "SAD-IRT ({stem})": {...}                                       │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     SCALE ALIGNMENT (Secondary Metrics Only)                     │
│                         (shared/evaluation.py)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  NOTE: Scale alignment is NOT needed for the primary metric (Mean Per-Agent AUC) │
│        It is only used for secondary metrics: Pooled ROC-AUC and Oracle MAE      │
│                                                                                  │
│  Problem: Baseline IRT and Feature-IRT have different scales than Oracle IRT     │
│           (IRT is only identifiable up to affine transformation)                 │
│                                                                                  │
│  Solution: Fit affine transform on "anchor tasks" (10-90% pass rate in both)     │
│                                                                                  │
│  compute_scale_offset(predicted_β, oracle_β, anchor_tasks, method):              │
│  ├── method="constant": offset = mean(oracle_β - predicted_β) on anchors         │
│  │   Returns: {"offset": float}                                                  │
│  └── method="affine": fit oracle_β = slope × predicted_β + intercept             │
│      Returns: {"slope": float, "intercept": float, "r_squared": float}           │
│                                                                                  │
│  shift_to_oracle_scale(predicted_β, alignment_params):                           │
│  ├── constant: β_shifted = β_pred + offset                                       │
│  └── affine: β_shifted = slope × β_pred + intercept                              │
│                                                                                  │
│  NOTE: Alignment uses oracle information - ONLY valid for evaluation!            │
│        Never use aligned values for training.                                    │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION METRICS                                      │
│                         (shared/evaluation.py)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TWO AUC METRICS:                                                                │
│                                                                                  │
│  1. Mean Per-Agent AUC (SCALE-FREE, PRIMARY METRIC)                              │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │ compute_mean_per_agent_auc(predicted_β, responses, frontier, agents)│     │
│     │                                                                      │     │
│     │ For each post-frontier agent j:                                      │     │
│     │   AUC_j = sklearn.metrics.roc_auc_score(y_true, -predicted_β)       │     │
│     │   (higher β = harder = lower expected success → use -β for ranking)  │     │
│     │                                                                      │     │
│     │ Return mean ± SEM across agents                                      │     │
│     │                                                                      │     │
│     │ Key insight: AUC only depends on ranking, not scale!                 │     │
│     │ No oracle abilities or alignment needed.                             │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  2. Pooled ROC-AUC (REQUIRES ALIGNMENT, SECONDARY METRIC)                        │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │ compute_frontier_auc(oracle_θ, shifted_β, responses, frontier, ...)│     │
│     │                                                                      │     │
│     │ For each (agent j, task i) pair on frontier tasks:                   │     │
│     │   P(success) = sigmoid(θ_oracle[j] - β_shifted[i])                   │     │
│     │                                                                      │     │
│     │ Pool all pairs and compute ROC-AUC                                   │     │
│     │                                                                      │     │
│     │ Requires: oracle abilities, scale alignment                          │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  filter_agents_with_frontier_variance(responses, frontier_tasks, agents):        │
│  └── Remove agents that fail ALL frontier tasks (no ranking signal)              │
│      Required before compute_mean_per_agent_auc()                                │
│                                                                                  │
│  compute_method_metrics(predicted_β, oracle_items, oracle_θ, responses, ...):    │
│  └── Full pipeline: align → compute both AUCs → return metrics dict              │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DATE FORECASTING (Optional)                               │
│                 (shared/date_forecasting.py + frontier_evaluation.py)            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Goal: Predict when frontier tasks become solvable (50% probability)             │
│                                                                                  │
│  Key insight: P(success) = 0.5 when θ = β (from IRT)                             │
│  Combined with: Frontier ability is linear over time (R² ≈ 0.98)                 │
│                                                                                  │
│  Step 1: Compute ground truth dates                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ compute_first_capable_dates(oracle_items, oracle_θ, agent_dates)        │    │
│  │ For each task: find earliest agent where θ ≥ β                          │    │
│  │ Returns: FirstCapableDatesResult                                        │    │
│  │   ├── first_capable_dates: Dict[task_id, datetime]                      │    │
│  │   ├── tasks_without_capable_agent: List[task_id]                        │    │
│  │   ├── earliest_agent_date, latest_agent_date                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Step 2: Fit ability-over-time model                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ DateForecastModel.fit(abilities, agent_dates)                           │    │
│  │                                                                         │    │
│  │ 1. Group agents by date, take max ability per date                      │    │
│  │ 2. Compute cumulative max (frontier trajectory)                         │    │
│  │ 3. Identify "frontier points" where ability improved                    │    │
│  │ 4. Fit linear regression: θ_frontier = slope × days + intercept         │    │
│  │                                                                         │    │
│  │ Returns: slope, intercept, r_squared, n_frontier_points                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Step 3: Predict solvability dates                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ DateForecastModel.predict(predicted_β, task_ids)                        │    │
│  │                                                                         │    │
│  │ Invert regression: days = (β - intercept) / slope                       │    │
│  │ Convert to calendar date                                                │    │
│  │                                                                         │    │
│  │ Returns: Dict[task_id, (days, datetime)]                                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Step 4: Evaluate predictions                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ compute_date_forecast_metrics(predicted, ground_truth_days, task_ids)   │    │
│  │                                                                         │    │
│  │ Returns: mae_days, rmse_days, pearson_r, spearman_rho, n_tasks          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Only methods with learned abilities can forecast dates:                         │
│  ├── Oracle ✓                                                                    │
│  ├── Baseline IRT ✓                                                              │
│  ├── Feature-IRT ✓                                                               │
│  ├── Embedding + Ridge ✗ (no IRT, no abilities)                                  │
│  └── LLM Judge + Ridge ✗ (no IRT, no abilities)                                  │
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       FRONTIER EVALUATION LOOP                                   │
│                    (shared/frontier_evaluation.py)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  evaluate_all_frontier_definitions(                                              │
│      frontier_definitions,  # ["zero_pre", "passrate", "irt"]                    │
│      data,                  # ExperimentData                                     │
│      raw_predictions,       # Dict[method, Dict[task, β]]                        │
│      date_info,             # Optional[DateForecastingData]                      │
│      alignment_method,      # "affine" or "constant"                             │
│  ) → Dict[frontier_def, Dict[method, metrics]]                                   │
│                                                                                  │
│  For each frontier_def in frontier_definitions:                                  │
│  │                                                                               │
│  ├── Get frontier_task_ids from data.frontier_tasks_by_def[frontier_def]         │
│  │                                                                               │
│  ├── Filter eval agents:                                                         │
│  │   eval_agents = data.eval_agents_by_def.get(frontier_def, post_frontier)      │
│  │   eval_agents = filter_agents_with_frontier_variance(eval_agents)             │
│  │                                                                               │
│  ├── For each method_name, predicted_β in raw_predictions:                       │
│  │   ├── compute_method_metrics() → pooled AUC (requires alignment)              │
│  │   └── compute_mean_per_agent_auc() → scale-free AUC                           │
│  │                                                                               │
│  ├── (Optional) Date forecasting for methods with abilities                      │
│  │                                                                               │
│  ├── Consolidate SAD-IRT runs to "SAD-IRT (best)"                                │
│  │                                                                               │
│  └── print_comparison_table(results, ...)                                        │
│                                                                                  │
│  Returns: all_results[frontier_def][method_name] = {auc, mean_auc, mae_days, ...}│
│                                                                                  │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               OUTPUTS                                            │
│               (shared/output_formatting.py + shared/plotting.py)                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CONSOLE OUTPUT (print_comparison_table):                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ EXPERIMENT B: FRONTIER TASK DIFFICULTY PREDICTION                        │    │
│  │ Frontier definition: zero_pre (0% pre-frontier, >0% post-frontier)       │    │
│  │                                                                          │    │
│  │ Data Summary:                                                            │    │
│  │   - Pre-frontier agents: 76                                              │    │
│  │   - Post-frontier agents: 55 (filtered to 44)                            │    │
│  │   - Frontier tasks: 34                                                   │    │
│  │   - Anchor tasks: 123                                                    │    │
│  │                                                                          │    │
│  │ Method                                        Mean AUC ± SEM    ROC-AUC  │    │
│  │ ─────────────────────────────────────────────────────────────────────── │    │
│  │ Oracle (upper bound)                          0.7322 ± 0.039    0.8399  │    │
│  │ Baseline-Init Feature-IRT (Embedding)         0.6512 ± 0.032    0.7841  │    │
│  │ Baseline IRT (pre-frontier only)              0.4854 ± 0.031    0.7503  │    │
│  │ ...                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  CSV OUTPUT (save_results_csv):                                                  │
│  Columns: method, mean_per_agent_auc, sem, std, n_agents, auc, n_tasks, ...      │
│                                                                                  │
│  PLOTS (from threshold_sweep.py):                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ threshold_sweep_{dataset}.png                                            │    │
│  │   plot_threshold_sweep_auc(): Mean Per-Agent AUC vs pre-threshold        │    │
│  │   X: Pre-frontier Threshold (%), Y: Mean Per-Agent AUC                   │    │
│  │   Lines: Oracle, Baseline IRT, Feature-IRT (with error bars)             │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ date_forecast_{dataset}.png (if enabled)                                 │    │
│  │   plot_threshold_sweep_mae(): MAE vs pre-threshold                       │    │
│  │   X: Pre-frontier Threshold (%), Y: MAE (days)                           │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ ability_vs_date_{dataset}.png (if enabled)                               │    │
│  │   plot_ability_vs_date(): Agent abilities over time                      │    │
│  │   Scatter: agent abilities vs release date                               │    │
│  │   Red stars: frontier points (where cumulative max improved)             │    │
│  │   Dashed line: linear fit with R²                                        │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ predicted_vs_oracle_{dataset}.png                                        │    │
│  │   plot_predicted_vs_oracle_scatter(): β_pred vs β_oracle for frontier    │    │
│  │   Diagonal reference line, fit line with equation                        │    │
│  │   Stats: Pearson r, MAE, N                                               │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ predicted_vs_actual_dates_{dataset}.png (if enabled)                     │    │
│  │   plot_predicted_vs_actual_dates(): Predicted vs actual solvability      │    │
│  │   X: Actual days, Y: Predicted days                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  DIAGNOSTICS (shared/diagnostics.py):                                            │
│  ├── Grid search heatmap: AUC vs (l2_weight, l2_residual)                        │
│  ├── Training loss curves: loss over iterations                                  │
│  ├── Loss components: NLL, weight_reg, residual_reg, ability_reg                 │
│  └── feature_irt_diagnostics.json: full grid search results                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## End-to-End Data Flow Summary

```
┌────────────────────┐
│  CLI Arguments     │
│  --dataset         │
│  --frontier_defs   │
│  --cutoff_date     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ DatasetConfig      │  Lazy-loads responses, agent_dates, all_task_ids
│ (per-dataset impl) │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ load_and_prepare_  │  Splits agents by date, trains/loads baseline IRT,
│ data()             │  identifies frontier tasks, builds train_responses
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ ExperimentData     │
│ - oracle IRT       │
│ - baseline IRT     │
│ - agent splits     │
│ - frontier tasks   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Feature Sources    │  EmbeddingFeatureSource, CSVFeatureSource
│                    │  → GroupedFeatureSource
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Prediction         │  Oracle, Baseline, Ridge, Grouped Ridge,
│ Collection         │  Feature-IRT (grid search)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ evaluate_all_      │  Per frontier def: align, compute AUC,
│ frontier_defs()    │  optionally forecast dates
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Outputs            │
│ - Console table    │
│ - CSV file         │
│ - PNG plots        │
└────────────────────┘
```

## threshold_sweep.py Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ main()                                                                          │
│   For each dataset in args.datasets (PARALLEL via multiprocessing.Process):     │
│     run_single_dataset(dataset_name, thresholds, ...)                           │
└─────────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ run_single_dataset()                                                            │
│                                                                                 │
│ TRAIN PHASE (once, threshold-independent):                                      │
│ ├── Load config, load_and_prepare_data(threshold=0)                             │
│ ├── Train/load Oracle IRT (all agents)                                          │
│ ├── Train Baseline IRT (pre-frontier agents)                                    │
│ ├── Train Feature-IRT (fixed hyperparams: l2_weight, l2_residual)               │
│ ├── Collect all predictions into all_predictions dict                           │
│ └── (Optional) Fit date forecast models from all_abilities                      │
│                                                                                 │
│ Generate threshold-independent scatter plots:                                   │
│ ├── predicted_vs_oracle_{dataset}.png                                           │
│ └── predicted_vs_actual_dates_{dataset}.png (if date forecasting)               │
│                                                                                 │
│ EVALUATE PHASE (per threshold):                                                 │
│ for threshold in [0.0, 0.05, 0.10, ..., 0.30]:                                  │
│ ├── Identify frontier tasks: pre_rate <= threshold                              │
│ ├── Get eval agents (fixed set from threshold=0 for consistency)                │
│ ├── For each method in all_predictions:                                         │
│ │   ├── Compute Mean Per-Agent AUC                                              │
│ │   └── (Optional) Compute date forecast MAE                                    │
│ └── Append to results list                                                      │
│                                                                                 │
│ Save outputs:                                                                   │
│ ├── threshold_sweep_{dataset}.csv (all metrics per threshold/method)            │
│ ├── threshold_sweep_{dataset}.png (AUC plot)                                    │
│ ├── date_forecast_{dataset}.png (MAE plot, if enabled)                          │
│ └── ability_vs_date_{dataset}.png (if enabled)                                  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

## IRT Formulas

### Standard 1PL IRT Model

```
P(success | agent j, task i) = sigmoid(θ_j - β_i)

where:
  θ_j = agent j's ability (higher = more capable)
  β_i = task i's difficulty (higher = harder)
  sigmoid(x) = 1 / (1 + exp(-x))
```

### Feature-IRT Model (FeatureIRTPredictor)

```
β_i = features[i] @ w + bias + residual[i]

P(success | j, i) = sigmoid(θ_j - β_i)

Loss = -Σ_{i,j} log P(y_ij | θ_j, β_i)    # Negative log-likelihood
       + l2_weight × ||w||²                # Weight regularization
       + l2_residual × ||residual||²       # Residual regularization
       + l2_ability × mean(θ)²             # Identifiability constraint
```

**Baseline-Init Mode:**
- Initialize residuals from baseline IRT difficulties (r_i = β_baseline[i])
- Initialize abilities from baseline IRT abilities (θ_j = θ_baseline[j])
- Initialize feature weights to zero (w = 0)
- Model starts at Baseline IRT solution, learns feature-based corrections

### Date Forecasting Model

```
Frontier ability over time (from Experiment D):
  frontier_θ = slope × days_since_earliest + intercept
  (R² ≈ 0.98 for 2PL model on SWE-bench)

Solvability condition (50% probability):
  P(success) = 0.5 when θ = β

Date prediction by inversion:
  days = (β - intercept) / slope
  date = reference_date + timedelta(days)
```

## Frontier Task Definitions

| Definition | Pre-frontier Criterion | Post-frontier Criterion | Use Case |
|------------|------------------------|-------------------------|----------|
| `zero_pre` | pass_rate == 0% | pass_rate > 0% | Strictest: truly unsolved pre-frontier |
| `passrate` | pass_rate ≤ 10% | pass_rate > 10% | Standard: allows some early solves |
| `irt` | P(solve) < 50% for all pre-agents | First agent with θ ≥ β is post-frontier | Principled: IRT-based capability threshold |
| `pre_only` | pass_rate ≤ X% | (no filter) | Threshold sweep: monotonically increasing sets |

## Key Files

| Component | File |
|-----------|------|
| Entry point (comparison) | `compare_methods.py` |
| Entry point (sweep) | `threshold_sweep.py` |
| Data preparation | `shared/data_preparation.py` |
| Prediction methods | `shared/prediction_methods.py` |
| Evaluation metrics | `shared/evaluation.py` |
| Frontier loop | `shared/frontier_evaluation.py` |
| Date forecasting | `shared/date_forecasting.py` |
| Output formatting | `shared/output_formatting.py` |
| Plotting | `shared/plotting.py` |
| Config base | `shared/config_base.py` |
| Diagnostics | `shared/diagnostics.py` |
| Feature sources | `../experiment_new_tasks/feature_source.py` |
| Feature predictors | `../experiment_new_tasks/feature_predictor.py` |
| Dataset: SWE-bench | `swebench/config.py` |
| Dataset: SWE-bench Pro | `swebench_pro/config.py` |
| Dataset: TerminalBench | `terminalbench/config.py` |
| Dataset: GSO | `gso/config.py` |

## Key Design Patterns

### 1. Lazy Loading in Config
```python
# Data is loaded on first access, not in __init__
@property
def responses(self) -> Dict[str, Dict[str, int]]:
    if self._responses is None:
        self._responses = load_responses_dict(self.responses_path)
    return self._responses
```

### 2. Multiple Frontier Definitions
ExperimentData stores results per frontier definition in dicts:
```python
frontier_tasks_by_def: Dict[str, List[str]]    # def_name -> task_ids
eval_agents_by_def: Dict[str, List[str]]       # def_name -> agent_ids
filtering_stats_by_def: Dict[str, AgentFilteringStats]
```

### 3. No Data Leakage Enforcement
- Baseline IRT trained ONLY on pre-frontier agents
- `train_responses` pre-filtered to pre-frontier agents at call site
- Anchor tasks used only for evaluation alignment
- Ground truth β comes from baseline IRT, not oracle

### 4. Scale-Free Primary Metric
- Mean Per-Agent AUC requires no oracle information
- Uses ranking: `AUC(y_true, -predicted_β)`
- Higher β = harder task = lower expected success
- More robust than pooled AUC requiring scale alignment

### 5. Fail-Loudly Philosophy
```python
# From CLAUDE.md: "Never write code that silently skips or ignores missing data"
def get_features(self, task_ids):
    for task_id in task_ids:
        if task_id not in self._features:
            raise ValueError(f"Task {task_id} not found in feature source")
```

### 6. Caching for Baseline IRT
```python
def compute_baseline_irt_cache_key(responses_path, pre_agents, cutoff_date):
    key_str = f"{responses_path.name}|{sorted(pre_agents)}|{cutoff_date}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]

# Cache location: output_dir/baseline_irt/cache_{cache_key}/
# Contains: items.csv, abilities.csv, cache_info.json
```

### 7. Threshold Sweep Architecture
- **Train ONCE** (threshold-independent): Feature-IRT, date models
- **Evaluate per threshold**: Only frontier identification changes
- **Fixed eval agents**: Use agents with variance at threshold=0 across all thresholds
- **Parallel execution**: One Process per dataset via multiprocessing

## Debugging Checklist

### 1. No frontier tasks found?
- Check `cutoff_date` format (YYYYMMDD, no dashes)
- Verify agent_dates mapping in config returns dates for all agents
- Try relaxing thresholds: `--pre_threshold 0.2`
- Check if responses have post-frontier agents with any successes

### 2. AUC is NaN or 0.5?
- All agents may pass or fail all frontier tasks (no variance)
- Run `filter_agents_with_frontier_variance()` first
- Check `eval_agents_by_def` has agents with both 0s and 1s
- Verify predictions exist for all frontier tasks

### 3. Baseline IRT retraining unexpectedly?
- Cache key depends on: responses filename + sorted agents + cutoff
- Check `chris_output/experiment_appendix_h_hard_tasks/{dataset}/baseline_irt/cache_*/cache_info.json`
- Different agent list (e.g., new agents added) invalidates cache

### 4. Feature-IRT not improving over Baseline?
- Check `l2_weight` (too high suppresses feature learning)
- Verify `baseline_abilities` are passed when `init_from_baseline=True`
- Look at `baseline_init_diagnostics` for feature contribution ratio
- If feature contribution is near 0%, features may not be predictive

### 5. Date forecasting has high MAE?
- Check `ability_vs_date_{dataset}.png` for non-linear relationship
- Verify sufficient frontier points (≥2) for regression fit
- Some datasets lack agent date diversity (see `DATE_FORECAST_DATASETS`)
- R² < 0.8 indicates poor linear fit

### 6. Scale alignment failing? (only affects secondary metrics)
- Scale alignment only needed for Pooled ROC-AUC and Oracle MAE, not Mean Per-Agent AUC
- Check anchor_task_ids is not empty
- Anchor tasks need 10-90% pass rate in BOTH agent groups
- If no anchors, try adjusting `anchor_min_pass_rate` in config
- Check if affine fit has reasonable slope (not 0 or inf)

### 7. Memory issues with large datasets?
- Feature sources load all embeddings into memory
- Consider using `device="cuda"` for FeatureIRTPredictor if available
- Responses dict can be large; lazy loading helps but still in memory

### 8. Inconsistent results across runs?
- Feature-IRT uses L-BFGS which is deterministic given same initialization
- Grid search selects best by AUC; ties broken by first config
- Check if baseline IRT cache was invalidated between runs
