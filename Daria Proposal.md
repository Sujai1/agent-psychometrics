# Task embeddings

## Project description

At a high level, the project involves *using embeddings to predict properties of benchmark problems*.

Normally, estimating difficulty, pass@k, or the time needed to solve a problem requires actually running agents on that problem, which is computationally expensive, so it would be useful to have a way to estimate those quantities with much less compute.

* **Embedding type**: last token of last hidden state of a pre-trained or instruction-tuned LLM.

  * Based on my past experience training a code embedding model, *last token pooling* works better than any other pooling method to obtain an embedding of the whole input from individual token embeddings, because the last token has “attended” to the entire previous context. (Possible alternatives, which I found to underperform last token, include mean, weighted mean, and other token positions.)

  * I ran an ablation experiment to find the best layer to extract embeddings from, and found prediction performance to be monotonic in layer depth (*last is best*).

  * For the sake of generalizability, the project should *not* involve training a specialized embedding model for a particular objective or set of objectives. I have been using instruction-tuned LLMs from HuggingFace because my input format includes an instruction (see “Input format” below). I ran an ablation experiment to find the best open-weight LLM backbone, and found *Qwen3-VL-8B-Instruct* to yield the highest performance.

* **Embeddings to target scores**: *ridge regression* with 5-fold cross-validation to find the best regularization parameter. I switched to ridge regression after observing severe overfitting for standard linear regression (see “Failure modes” below). As an alternative kind of linear model (to possibly consider in the future), [Truong et al. (2025)](https://arxiv.org/abs/2503.13335v1) use an EM algorithm.

* **Metric**: So far, I have been using the *coefficient of determination* (R2) between the scores predicted from embeddings and the “ground-truth” labels as the main measure of prediction quality. This may not be the best approach if the label taken as “ground truth” is itself a proxy (e.g., IRT, which is only a proxy for difficulty), because it measures goodness of fit to the proxy rather than the true objective. In the future, it would be good to consider other metrics (Pearson *r*, Spearman *ρ*, MSE).

  Update: now I am also measuring ROC-AUC following [Truong et al. (2025)](https://arxiv.org/abs/2503.13335v1).

* **Input format**: task statement \+ solution \+ instruction.

  * I found that *removing solution results in a major drop in performance* for difficulty prediction, which is understandable because even humans will estimate difficulty more accurately if they have access to the solution, but this restricts the method to benchmarks with known solutions and undermines its application to real-world tasks.

  * Conversely, I found that *adding instruction boosts performance*, especially when the instruction is at the end (rather than the start) of the input, because this aligns the original purpose of the embedding (being used to predict the next token) with our purpose (the next token depends directly on the property we want to predict).

  * *Example instruction* (for difficulty prediction): “How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\\n”

  * I found that *the precise nature of the instruction does not matter*, as long as it makes the LLM think about the target property: changing the instruction above to request an integer between 0 and 10, or even a binary label “easy/hard”, caused only minor changes in performance. However, prompting the LLM to output something anti-correlated with the target property (like the above difficulty score when the target property is pass@k) lowers performance significantly.

I tried using embeddings to predict **IRT difficulty scores** (1D 1PL) for tasks in **SWE-Bench Verified** (*main experiment*), and later extended the experiments to (i) **IRT difficulty** and (ii) **number of turns to solve a task** in **SWE-Bench Pro** and (iii) **pass@5** in **Terminal-Bench**. In the main experiment, using an 80-20% train-test split, my best R2 performance (train/test) was **0.436/0.324**. While this performance is non-trivial, it is not yet good enough for a production-ready tool, so further work is needed to raise performance even higher.

Update: using IRT scores with improved initialization (hierarchical prior), the new best R2 is **0.491/0.388**, and the corresponding ROC-AUC is **0.851/0.833**.

## Expected results

The expected result is a **framework** (*embedding extraction algorithm \+ regression algorithm \+ input format*) such that users can obtain accurate predictions for a wide variety of quantifiable properties (difficulty, solve time, etc.) and coding tasks (web development, ML, etc.) while making only low-effort changes to the input (insert task statement and write a reasonable instruction, but *no* sophisticated prompt tuning).

At *minimum*, the project will result in a *blog post*; at *maximum*, in a *paper* submitted to a *main conference* (e.g., COLM). It would be exciting if the framework could generalize to domains beyond coding and also to real-world problems as opposed to benchmark problems. If the project succeeds, it would be interesting to visualize the embeddings (via PCA/t-SNE) and investigate *why* they are so versatile.

## Failure modes

### Overfitting on small benchmarks

Since embeddings are very high-dimensional (4096 in the case of Qwen3-VL-8B-Instruct) and most benchmarks have fewer tasks than components in the embedding vector, *linear regression is prone to overfitting*.

*Ridge regression is less likely to overfit* than simple linear regression: my very first experiment run, where I used simple regression, resulted in severe overfitting (perfect R2 on the train split and negative on the test split), after which I switched to ridge regression and no longer observed such issues.

The smallest benchmark I have used so far was Terminal-Bench (89 tasks). I tried to predict average (across agents) pass@5 and got R2 (train/test) \= 0.995/0.222. Overfitting is noticeable (0.995 on the train split), but the test score is still decent.

### Low R2 for some benchmarks or prediction objectives

When I extended the experiments to SWE-Bench Pro, I observed *modest R2 values*: 0.285/0.171 for predicting IRT difficulty and 0.267/0.152 for predicting the number of turns to successfully solve a task. It is possible that scores for some other benchmarks or objectives we try in the future turn out to be even lower. If scores are low in many cases important to potential users, we will need to change our approach.

## Timeline

* **Week 1** (past): exploration & preliminary findings  
* **Week 2**: run experiments on the core set of coding benchmarks and prediction objectives to maximize performance and finalize framework  
* **Week 3**: run the framework on a wider variety of domains to evaluate universality/generalizability  
* **Week 4**: visualization & analysis of results  
* **By COLM deadline** (late March): paper writeup & submission

# Success predictor

Another idea I can explore as a side project (time permitting) is training a classifier to *predict, given a task statement and a partial trajectory, whether or not the agent will eventually succeed* in solving the task. The application of this would be an *early-stopping feature* in Fulcrum’s coding system: if the classifier is confident that the agent is going to fail, the agent is stopped early to save compute and/or API costs.

When training such a classifier, we need to be careful about teaching it the truly relevant features rather than any spurious features correlated with success. In particular, *agent-identifying information* (like agent names or linguistic patterns associated with particular agents’ responses) is a likely confounder, because some agents are more capable than others.

So far, I ran an experiment with the *SWE-Bench Verified* trajectory dataset, applying Chris’s *unified format script* to remove correlation between trajectory formats and particular agents, and his *filtering script* to remove agent identifiers (which effectively removes agent’s chat messages and reasoning chains). I trained *Qwen3-VL-8B-Instruct* (the LLM that had the best embedding performance) with a *classification head* and with *LoRA*. My input format was “*task statement \+ solution \+ trajectory prefix*” (with prefix length randomized between 5000 and 10000 characters). I included solution because I observed that this resulted in a major performance boost for embeddings, but in practice we will want to deploy the classifier on open-ended tasks with no solutions known in advance. My test set performance (accuracy/ROC-AUC/F1-score) was **0.681/0.771/0.668**, which is interestingly better than an ablation run without filtering (0.640/0.700/0.610). In the test set, 50.41% of examples had the “success” label, so this performance is significantly above naive baselines like predicting a random label or always predicting success.

Future work includes *data augmentation* (we can reuse the same trajectory with different prefix lengths), *expansion to more benchmarks* beyond SWE-Bench Verified so that the classifier can handle more diverse coding tasks, and *A/B tests* for LLM backbone and input format.

The level of priority of this project is to be determined by future discussion. Currently, I plan to focus on embeddings, but I may spend more time on the success predictor if the embedding project doesn’t go well, or if it goes well easily and I have remaining time.