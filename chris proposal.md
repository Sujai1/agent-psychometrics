# **Bayesian Inference for Agentic Evaluation of Task Difficulty**

## **1 Introduction, Settings, & Motivation**

The core objective of this project is to derive calibrated estimates of **task difficulty** ($$\\beta$$) for **long-horizon** **agents** based on **human-interpretable** features of the tasks and agents. While standard evaluation metrics (Pass%) provide a retrospective summary, they fall short for different reasons in the two regimes of problem difficulty:

1. **The Solvable Regime:** For tasks that are solvable by current models, running full evaluations is computationally expensive. Can we accurately predict difficulty in a *cheaper* way than running the agent evaluation itself?  
2. **The Frontier Regime:** For tasks where current models fail (0-20% pass rate), we currently lack signal about real difficulty. Can we estimate *how far* these tasks are from being solved, or understand *what* about agents’ interactions with these tasks makes them so difficult? 

In answering these questions, we hope to answer other questions regarding our understanding of what makes a problem difficult. Is difficulty a general property of a problem, or does it depend heavily on what agent is trying to solve the task? If it’s a general property, what human understandable features make the problem hard for models? Do agents behave differently when they are close to being smart enough to solve a problem versus when they are hopelessly lost? 

To answer these questions, we propose a unified **Bayesian Difficulty Inference** framework. Unlike standard IRT approaches that treat difficulty as a black-box scalar derived solely from success outcomes, we model it as a latent variable informed by **Problem Structure** (Prior) and **Failure Semantics** (Posterior).

### 1.1 The Solvable Regime

In this regime, the new long horizon task is solvable by current agents, and the goal is to predict the task difficulty with lower computational cost than it takes to run pass@k for a given agent on that task. In particular, we will use only the problem statement and environment). 

This regime is very similar to the difficulty prediction trained in [Reliable and Efficient Amortized Model-based Evaluation](https://arxiv.org/pdf/2503.13335) (Truong et al., 2025). Our settings and goals are a little different from them however. They use QA/Math, which are very short-horizon tasks (so that running their difficulty predictor is often as expensive as running the agent itself **\[experiment C\]**), while we’ll use long-horizon agentic tasks. We want good predictive accuracy of the IRT difficulty score similar to them, but we want to do so at a much lower computational cost than running the agent evaluation. 

**Motivation**: In long-horizon domains like software engineering, valid RL learning signals are incredibly sparse and expensive to obtain. Current RL methods often sample tasks uniformly from a dataset. This is inefficient, as if the agent samples a task that it can complete with 100% certainty, or a task that it will never be able to complete, then GRPO has an advantage of 0 and there is zero gradient update. Existing work like [GRESO](https://arxiv.org/pdf/2506.02177) tries to avoid sampling these kinds of tasks for the Math RL setting; the efficiency gains could be even larger for long horizon SWE tasks. 

### 1.2 The Frontier Regime

In this regime, the new long horizon task is NOT solvable by current agents, and we want to predict the difficulty using as much test time compute (i.e. agent trajectories on the problem) as we want. In practice, in our SWE Bench Verified dataset, we only have one trajectory per agent x task pair available to us, so that’s probably what we’ll use.  

**Motivation:** Harder benchmarks often include tasks with a 0% pass rate. Among these tasks, we have no way of distinguishing harder from easier tasks using traditional automated methods like IRT. There are at least two reasons why people might care about knowing more the difficulty of these tasks and what makes them difficult:

* **Bottlenecks:** Understanding how failure modes relate to difficulty of problems is useful for improving agents. Knowing these bottlenecks can help researchers decide what to work on to improve model capabilities. For instance, [Nebius](https://arxiv.org/pdf/2508.03501) uses the fact that Qwen2.5-72B-Instruct is very poor at instruction following on SWE-Bench Verified as the starting point for their RL training strategy. It turns out that for SWE-Bench Verified, some researchers have already come up with a very good understanding of the [failure modes](https://arxiv.org/pdf/2509.13941) of agent trajectories. We hope to replicate this failure mode analysis using Lunette **\[experiment B.1\]**, where we can compare to the human labeled categories directly, and then test if our judge features generalize to other long horizon agentic coding tasks like SWE-Bench Pro, using the downstream difficulty prediction for very hard problems experiment **\[experiment B.2\].**   
* **Forecasting "Time-to-Solve":** We expect that harder currently-impossible tasks will take more time before models can solve them than easier currently-impossible tasks. It turns out that the experiments we run will be enough to give us precise forecasts about when in the future we expect each currently-impossible problem to be solvable **\[experiment D\]**. People are extremely interested in this — the [METR time horizon experiment](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) is a testament to how much people care about accurate forecasts of AI capabilities. But the METR time horizon paper is noisy, as there is large variance in agent ability to solve a task that takes humans equal amounts of time. The [Epoch AI Rosetta Stone paper](https://arxiv.org/pdf/2512.00193) does similar forecasting of model capabilities, but their predictions are at a *benchmark level*. Our forecasts apply to one benchmark, and take place at a *task-level* granularity.    f 

---

## **2 Theoretical Framework: Amortized Bayesian Inference**

We will use the IRT framework that every task $$i$$ has a latent difficulty $$\\beta\_i$$. Our main goal throughout will be to provide good estimates of $$\\beta\_i$$ with various affordances. We use the standard IRT model of success probability of an agent $$j$$ on task $$i$$: 

$$P(y\_{ij} \= 1 \\mid \\theta\_j, \\alpha\_i, \\beta\_i) \= \\sigma(\\alpha\_i(\\theta\_j \- \\beta\_i))$$

To address the solvable regime setting, we will train one linear model that predicts a prior estimate of this latent difficulty just using textual data and metadata of the problem and access to the problem environment from Lunette.  

To address the frontier regime setting for when we observe a new problem \+ failing trajectory data, we train a linear model to predict the update to our prior difficulty estimates. 

Philosophically, the Solvable Regime is the Prior ($$\\mu\_\\phi$$) and the Frontier Regime is the Posterior update. 

### 2.1 Handling Multiple Trajectories

Since we are allowing any amount of test time compute for the frontier regime, we can update our posterior for the difficulty of a problem using a *set* of $$k$$ failure trajectories $$\\{\\tau\_1, \\dots, \\tau\_k\\}$$. In our current SWE bench verified data, we have 1 trajectory per agent, task combination. This is a little weird, as it means that we do have multiple trajectories per problem, but every trajectory is from a different agent. For now, I plan to just average together the predicted updates to the item difficulty from each trajectory, but I feel like there are better ways to leverage the fact that we have multiple trajectories than simple averaging. Like I think a very simple easy improvement is to weigh smarter models more. 

### 

### 2.2 Goal 1: The No Trajectory Prior

**Objective:** Predict task difficulty using only static features.

This effectively replicates the method from [*Truong et al. (2025)*](https://arxiv.org/pdf/2503.13335), but adapted for the Agentic domain using **Lunette**.

* **Input ($$x\_i$$):**  
  * **Option 1:** Issue description embeddings. This is Daria’s domain.   
  * **Option 2: Lunette Features:** Environment information. Run a dummy agent on the problem that doesn’t actually do anything to get this sandbox. (e.g. number of files in swe bench pro problem statement or in the repo is allegedly predictive, have to read the paper to figure this out )  
* **Model**: A linear model predicting $$\\beta\_i$$ from $$x\_i$$.   
  * a linear model so that we can have interpretability  
* **Training:** We do the same training setup as in Truong et al. Simultaneously train the IRT ability parameters and the linear model parameters that calculate the model difficulty (using your favorite algorithm, like EM or whatever), to maximize log likelihood of the observed successes/failures.  
* **Inference:** Using this learned linear model, given a completely new problem with no agent runs on it (e.g. a problem from the held out set), we can calculate its $$x\_i$$ and then run the linear model to predict its difficulty $$\\beta\_i$$, which we can then use with a known agent ability $$\\theta\_j$$ to predict the probability that that agent gets that problem correct. **\[experiment A\]**

### 2.3 Goal 2: The "Failure-Informed" Posterior (Weak to Strong generalization) {#2.3-goal-2:-the-"failure-informed"-posterior-(weak-to-strong-generalization)}

**Objective:** Refine the difficulty estimate of really hard problems using failure trajectory from agents too weak to solve the problems.

It will be convenient to define notation for the strength of models. We can bin agents into three sets by their **submission dates** (reflecting the forecasting application, where the time cutoff is models whose submission dates are the current day), $$M\_1$$, $$M\_{2}$$, and $$M\_{3}$$, in increasing order of submission date. For instance, Claude 3.5 might be $$\\in M\_1$$, Claude 4.0 might be $$\\in M\_2$$, and Claude 4.5 might be in $$M\_3$$. The goal is to be able to use failing trajectories on $$M\_N$$ to estimate the difficulties of tasks that models $$M\_{N+1}$$ may be able to succeed. So we will not be estimating the difficulty of tasks that are vastly harder than anything that current models are able to accomplish. 

* **Input:**  
  * The Prior difficulty estimate $$\\hat{\\beta}\_{prior} \= \\mu\_\\phi(x\_i)$$ (derived from Goal 1's static analysis).  
  * A set of failure trajectories $$\\{\\tau\_{ij}\\}\_{j \\in M}$$ generated by the weak models $$j$$ on task $$i$$.  
* **Model**:  
  * For each trajectory, we will use Lunette as judge to predict the **set** of *failure modes* $$c(\\tau\_{ij})$$ of each trajectory (in a multiple select way, matching the paper mentioned in the next sentence). This will be based off of a manually curated prompt that relies on the taxonomy proposed in the [empirical analysis of swe bench failures paper](https://arxiv.org/pdf/2509.13941)  for the failure modes classification. Trajectories that did not fail will go into their own category. Daria can use a classification head on her embeddings for this.   
  * For each trajectory, we will also use Lunette Grading to get a feature vector of information about the trajectory, $$f(\\tau\_{ij})$$. See [section 2.4.2](#2.4.2-lunette-as-judge-for-trajectory-encodings) for more details. Daria will directly use her embeddings for this $$f$$  
  * We define the posterior difficulty as the prior plus a learned correction term $$\\Delta\_\\psi$$. This correction term is parametrized by a set of linear models, one for each failure mode $$c$$, with weights $$\\psi\_c$$: the predicted posterior mean difficulty score for problem $$i$$ is equal to $$\\mu\_{\\phi}(x\_i) \+\\left( \\frac{1}{|M|}\\sum\_{j \\in M} \\frac{1}{|c(\\tau\_{ij})|} \\sum\_{k \\in c(\\tau\_{ij})} \\psi\_{k}^Tf(\\tau\_{ij})\\right)$$ where $$M$$ is the set of models for which we have inference time trajectories for the task   
* **Training:** We treat this as a supervised regression problem where the goal is to predict the "True" IRT difficulty using only "Weak" failure signals.  
1. **IRT Model:** We will first train a standard 1d 2PL IRT model on all of the SWE Bench verified data. Actually, I have already done this. We will use these as the ground truth difficulty scores for every remaining part of this experiment (see the note in 2i).   
2. **Data Selection:** We want to construct   
   * a training set $$\\mathcal{D}\_{train}$$ consisting of tasks where 0-20% of models in $$M\_1$$ fail at each task (we can vary the value 20%, that was arbitrary), while more models in $$M\_2$$ have solved the tasks, so that we would have reasonable IRT difficulty scores for the tasks.  
     1. The Lunette grading agent should be using an LLM that’s also before the cutoff date that we’re training on  
     2.  In the real world analog, everything is one up, so $$M\_2$$ is supposed to represent the frontier of what models are capable of right now. So technically in the real world analog, we would be training the IRT model on only $$M\_1$$ and $$M\_2$$. However, we can’t actually do that in this experiment, because in the validation set, we have to train the IRT model on $$M\_3$$ as well, and if we train IRT on different models, we’ll get different scales for the difficulty since the difficulty scores are relative.   
   * a validation set $$D\_{valid}$$ consisting of different tasks where 0-20% of models in $$M\_2$$ fail each task, while more models in $$M\_3$$ have solved the tasks, so that we still have reasonable IRT difficulty scores for them at all. We will test if the weights $$\\psi$$ trained on $$D\_{train}$$ transfer over to $$D\_{valid}$$. **\[experiment B\]**   
3. **The Objective Function**: We freeze the prior model $$\\mu\_\\phi$$ and train the correction network $$\\Delta\_\\psi$$ to minimize the regularized mean squared error between the predicted posterior and the ground truth difficulty. The loss function $$\\mathcal{L}(\\psi)$$ is defined as:

$$\\mathcal{L}(\\psi)=\\frac{1}{|\\mathcal{D}\_{\\text{train}}|}\\sum\_{i \\in \\mathcal{D}\_{\\text{train}}}\\left\\| \\underbrace{\\left( \\mu\_\\phi(x\_i) \+ \\left( \\frac{1}{|M|}\\sum\_{j \\in M} \\frac{1}{|c(\\tau\_{ij})|} \\sum\_{k \\in c(\\tau\_{ij})} \\psi\_{k}^Tf(\\tau\_{ij})\\right)\\right)}\_{\\text{Predicted Posterior}} \- \\underbrace{\\beta\_i^{GT}}\_{\\text{True Difficulty}} \\right\\|^2 \\;+\\; \\lambda \\, \\|\\psi\\|\_2^2 $$

### 2.4 Problem and Trajectory Encodings 

Both the goals require some encoding of the inputs, either the problem text an environment $$x\_i$$, and/or the trajectories of the failing agents $$\\tau\_i$$. We will do this in two ways: Lunette Grading, and trajectory embeddings. This is the only main difference between what I think Daria and I will be doing within the framework of this proposal that I’ve written up. I will be using Lunette Grading. Even if trajectory embeddings end up doing better performance wise, it may still be useful to consider the Lunette Grading approach, as it computes more human interpretable features, which are useful to answering the questions that we posed in the introduction. For each of the below subsections, I will start off by taking inspiration for the grading prompts from the numerous similar LLM judge experiments done in the literature. I will proceed to list some of those below. 

#### 2.4.1 Lunette as judge for Problem Encodings

* [General Scales IRT](https://arxiv.org/pdf/2503.06378) scores problems on 18 different axes, such as Logical Reasoning, Domain Knowledge, Atypicality  
* [SWE-Bench-Pro](https://arxiv.org/pdf/2509.16941) (discussed more below) has some examples of useful information we could look for in the environment for problem difficulty. Examples are the number of files involved in the repository, which they found is quite predictive. They instruct the judge to run some commands to investigate the environment. We can see if Lunette can do similar things. 

I will evaluate these embeddings based on how well they do in the pipeline for Goal 1, as described in section [2.2]().

#### 2.4.2 Lunette as judge for Trajectory Encodings {#2.4.2-lunette-as-judge-for-trajectory-encodings}

* [SWE-Bench-Pro](https://arxiv.org/pdf/2509.16941) has analysis about the failure modes of the trajectories. They include their prompts at the end of the appendix.    
* [SWE-Bench-Verified Failure Analyses](https://arxiv.org/pdf/2509.13941) manually inspects failures of 150 runs and defines a taxonomy of failure modes. If we want to try predicting a fixed class, or even convert these into binary 0/1 features that Lunette grades, these are good directions to try.   
* [AgentDiagnose](https://aclanthology.org/2025.emnlp-demos.15.pdf) judges the quality of the trajectories for NNetNav-Live dataset, along axes like Backtracking, Task decomposition, Observation reading, Self-verification, Objective-quality on scales of 1-4. Their prompts are directly included in the appendix. **These prompts may be more useful for judging the trajectories that actually succeeded.** 

I will evaluate these embeddings based on how well they do in the pipeline for Goal 2, as described in section [2.3](#2.3-goal-2:-the-"failure-informed"-posterior-\(weak-to-strong-generalization\)).

---

## **3 Experimental Plan**

We focus on two core experiments to validate the Prior (Goal 1\) and the Posterior (Goal 2), along with some other experiments that will hopefully answer some of the questions that we posed in the introduction and justify that our approach really solves the issues we used to motivate it. 

### 3.1 Experiment A: Prior Validation (IRT AUC)

**Goal:** Prove that Lunette grading of problems can accurately predict success probability on new problems without running agents.

* **Protocol:**  
  * Train our Prior model (Section [2.2]()) on a train split of the SWE-bench dataset tasks.   
  * Evaluate AUC on a held-out test set of tasks.  
* **Metric:** **AUC (Area Under the Curve)** of predicting binary success for a held-out agent.  
  * *Baseline 1:* Predicting success using the average performance of that agent on tasks in the testing dataset (i.e. ignoring the actual task)  
  * *Ablation:* Using Text only LLM judges (by Text I mean the information available in SWE-Bench-Verified from the huggingface dataset, including the solution).   
  * *Mine:* Using Text \+ Lunette Features.  
  * *Daria:* Using Embeddings of the Text. Can maybe even combine with my approach by having Daria append the Lunette findings about the environment to her prompt that she’s embedding.   
  * *Ours without solution:* Text excluding the solution of the problem  
* **Plot**: bar graph showing the AUC’s for the different methods.   
* **Hypothesis:** Baseline \< Ablation \< Ours without solution \< Mine \< Daria. Including environment structure (Lunette) significantly improves AUC compared to text alone for long-horizon tasks. We’ll be able to read off from the linear regression which of the Lunette grading features were most important for predicting the difficulty of the problem. We should see some statistically significant coefficients for the features that require environment access. Hopefully excluding the solution doesn’t drop the performance into the ground for me, but we’ll see because apparently it did for Daria. 

### 3.2 Experiment B: Posterior Validation (IRT Difficulty Prediction for Harder Tasks) 

**Goal:** Prove that failure trajectories from a "dumb" model contain signal about the solvability for a "smart" model.

* **Protocol:** Construct $$D\_{train}, D\_{valid}$$ as described in Section [2.3](#2.3-goal-2:-the-"failure-informed"-posterior-\(weak-to-strong-generalization\)). For each task, sample about 20 agent trajectories that fail on that task as the $$M$$. Train the objective function using Ridge Regression.   
* **Metric:** **Pearson Correlation ($$r$$)** between Predicted Difficulty and Actual Strong Model IRT Difficulty score on $$D\_{valid}$$. We’ll use the same frozen prior model for all of these methods, even for the Ablation which doesn’t use Lunette for the trajectory judging.   
  * *Baseline*: Predicting difficulty using the frozen prior model only   
  * *Ablation*: Using non Lunette LLM judges.   
  * *Mine*: Using Lunette judging of the trajectories.  
  * *Daria*: Using Embeddings of the trajectories.   
  * *Ours without solutions*  
* **Plots**  
  * bar graph of the pearson correlations for the different methods   
  * scatterplot of our methods’ predicted difficulties against the actual IRT difficulty scores  
  * scatterplot of the predicted arrival date of models $$\\in M\_3$$ able to solve certain tasks vs the actual arrival date of models in $$M\_3$$ able to solve certain tasks,   
  * (more plots discussed below in some further variations of this experiment)  
* **Hypothesis:** Baseline \< Ablation \< Mine \< Daria. Here the ablation might be pretty similar to the Lunette results, since we’re using uploaded trajectory data so we don’t actually have environment access. As before, we have interpretable Lunette judge descriptions of the trajectory, along with coefficients showing that they’re real. This will show that we can estimate the difficulty of a problem that’s too hard for agents to actually solve themselves, with some help from knowing the solution of the problem and from some agent trajectories failing the problem.   
* The interpretable features will also help us answer what about the agents’ interactions with the tasks makes them so difficult.  

Extra experiments

* We can also forecast when models in $$M\_3$$ would arrive, and see if it lined up with when they actually arrived (**scatterplot** predicted vs actual) .  
* We can do an extra ablation where we **change the 20 agent trajectories** used for each task. If we see very similar difficulty predictions (measured using pearson correlation between the old and the new predictions), we can answer “Is difficulty a general property of a problem, or does it depend heavily on what agent is trying to solve the task?” in the yes it’s a general property of a problem  
  * **scatterplot** of the old vs the new predicted difficulty scores  
  * and then the coefficients on the Lunette judge features tell us human understandable features that make the problem hard for models in general   
* Do agents behave differently when they are close to being smart enough to solve a problem versus when they are hopelessly lost?   
  * **Plot:** Look at the failing trajectories on the easier impossible problems vs the harder impossible problems, and see if a 2D PCA or t-SNE visualization of the encodings of trajectories are different for the two groups. 

#### 3.2.1 Experiment B.1: 

As an initial goal to see if we have reasonable Lunette grading features, we can see if our Lunette evaluation matches the **manually labeled** reasons for failure for the 150 labeled failing trajectories from the pre-existing SWE bench trajectories, at [https://anonymous.4open.science/r/IssueSolvingEmpirical/README.md](https://anonymous.4open.science/r/IssueSolvingEmpirical/README.md). Note that these labels are multiple select; multiple different labels can apply to one trajectory (they’re labeled at specific steps of the trajectory).  
We can have Lunette literally try to predict the same label format that they use, and see how it does. This can be a good first experiment to try to run to see how good our Lunette grading evaluations are. As a baseline, use non Lunette LLM as a judge. 

### 3.3 Experiment C: Compute Costs — Waiting on Lunette server cost estimate

This experiment will hopefully answer “Can we accurately predict difficulty in a cheaper way than running the agent evaluation itself?” in the affirmative 

Measure cost in both compute time and literal dollars of running an agent on a task, vs doing the prior inference of a new task using Lunette on just the problem. **I’ll need information from the server side about what the cost is of running Lunette.** Hopefully it is not close. Might need to also report how much one-time compute we need to train the IRT model and the linear models.

**Results** (in results doc tab in more detail): Lunette is about 2x faster than a single agent trajectory on a task, which is good since we’re comparing to pass@k agent trajectories. 

### 3.4 Experiment D: Time Horizon Prediction — COMPLETED

![][image1]  
We plot the frontier IRT ability scores of the best performing model up to a certain date in the real world. The results are very linear, which lets us ground IRT ability into the actual real world time that models will have that ability. Since the $$\\theta$$ that an agent needs in order to be able to solve task $$i$$ with probability $$50\\%$$ is exactly the difficulty of the problem $$\\beta\_i$$ (because $$\\sigma(\\alpha\_i(\\theta \- \\beta\_i)) \= \\sigma(0) \= 0.5$$), this lets us convert from **IRT difficulty score** to **when in real life an agent will first be able to achieve 50% pass probability on problems of that IRT difficulty score.** This synergizes very well with our other results, which allow us to estimate the IRT difficulty score for any problem, as it allows us to precisely forecast when in the future each problem will be 50% solvable by the best agent.   
Note: if we’re trying to validate this result, we’ll have to redo the linear fit with data only up until the cutoff date that we’re pretending to be at (i.e. before any results that we’re trying to forecast). 

### 3.5 Experiment E: SWE Bench Pro Generalization

**Goal**: Prove that the learned encodings (Lunette judge features / trajectory embeddings, not the weights themselves) for predicting difficulty for smarter models from dumber models transfer over to other datasets 

We will use SWE-Bench-Pro for this experiment. We’ll repeat the exact same process as experiment B on SWE-Bench Pro, without changing the Lunette Grading features that we extract. If we find that the results do still hold on SWE-Bench Pro, then we can make a prediction using the entire SWE-Bench Pro dataset about the difficulty of tasks that models are currently failing on. This will help answer “Can we estimate how far these tasks are from being solved” 

## **4 Key Failure Modes and Risky Assumptions** 

1. Failure mode: Lunette judging of the problem takes more compute / cost than doing a full agent trajectory run on the task. If true, it makes our setting 1 pretty meaningless, and we should pivot to only doing setting 2\. We should do experiment C as soon as possible to figure this out.   
2. Failure mode: the trajectory data from SWE bench experiments (or even the SWE bench data itself) is not mature enough to give us interesting predictiveness. This is **very likely** to be true, and it’s the opinion that Sang holds.   
3. Failure mode: the Lunette / embedding encodings are useless without having the solutions of the problems. I don’t know what to do if this is true, and it seems very much like it could be true based on Daria’s preliminary findings.   
4. This posterior setup is really not grounded. The prior is fine, that’s grounded in the Sang paper which we know works. The posterior setup is combining two assumptions:  
   1. You can predict strong model performance (which IRT difficulty is a proxy for) at all, based on failing weaker model trajectories \+ solution information. This feels true ish, but might get cooked by emergent behavior. I don’t know how to reduce uncertainty on this assumption without actually doing the whole experiment.   
   2. The specific functional form of a linear model on trajectory features and the normal L2 loss is sufficient to predict IRT difficulty scores. This seems true enough based on our preliminary experimenting the past couple of days.   
5. Failure mode: Lunette judging doesn’t actually help on top of just LLM judging. Then Fulcrum Research is a little cooked, so that sounds like a y'all problem. But Experiment B.1. should help with this  
6. I might not be able to get Lunette / LLM grading to give useful feature encodings at all by trying a bunch of vaguely motivated prompts. If we hit this failure mode, Daria’s trajectory embeddings can still get us interesting results. We can even get a little bit of interpretability from the t-SNE plot I mention at the end of experiment B.   
7. Invisible Failure Mode: there’s a slight distribution shift, because in our training and evaluation, we’ll be using uploaded trajectory runs to Lunette, which don’t contain environment access, while for the actual new trajectory evaluations, Lunette will have access to the environment. Not sure how to get around this without doing a bunch of new agent runs ourselves, but it’s an invisible failure mode because the only way to see it is by running Lunette on a bunch of trajectories. 

## **5 Expectations**

I think a minimum line would be to write a blog post showing just the Prior part in experiment A, that using Lunette to get more information about the environment and either using those numbers directly or embedding them along with the rest of the problem improves the AUC significantly compared to a non Lunette baseline. 

An excellent outcome would be answering all the questions and running all the experiments that I proposed above, and writing a paper and submitting it to COLM. I really don’t think we’re going to make the ICML deadline, and I also wouldn’t be able to go to the conference anyways, so I’m more interested in submitting to COLM I think. 

## **6 Timeline** 

by EOD Saturday 1/10: try to set up Lunette on just the problem text itself, and see the cost of that compared to running a full agent on a task (experiment C). 

Week of 1/12: Run the main part of experiment B up to getting that the spearman rho of the predicted difficulties, using Daria’s code for the difficulty predictor prior for now. Run experiment B.1.

Week of 1/19: Go back and run my Lunette grading experiment A, using Daria’s code for training and evaluation that she hopefully did in the week of 1/12

Week of 1/26: Gather the other plots in the other bullet points of experiment B, then do experiment B.2, in that order (since touching SWE-Bench Pro seems like opening another can of worms that will take much longer). Plan out the paper structure and plots with Kaivu and Uzay and Dan Fried. 

After January: Actually write up the paper. Submit before the COLM deadline. 
