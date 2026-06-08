---
layout: page
title: Learning in continuously changing environments
description: Rethinking stability and plasticity in continual reinforcement learning
img: assets/img/12.jpg
importance: 1
category: work
---
Paper: <a href='https://arxiv.org/abs/2605.26357'>https://arxiv.org/abs/2605.26357</a>

Code [3D Four Rooms / Jax]: <a href='https://github.com/raymondchua/multi-timescale-successor-features-fourrooms'>https://github.com/raymondchua/multi-timescale-successor-features-fourrooms</a>

Code [MuJoCo / PyTorch]: <a href='https://github.com/raymondchua/multi-timescale-successor-features-mujoco'>https://github.com/raymondchua/multi-timescale-successor-features-mujoco</a>

This blog post is based on our recent work on Balancing Plasticity and Stability with Fast and Slow Successor Features, 
which explores how reinforcement learning (RL) agents can remain robust in environments that evolve continuously over time. 
A preliminary version of this work was also presented at the Computational and Systems Neuroscience conference (Cosyne) 
in Lisbon, Portugal in 2021.

## The challenge of learning in a changing world
Imagine learning to walk on ice during winter, only for the ground to slowly become dry pavement again by spring. 
Biological systems continuously adapt to environments that never remain fixed. Yet most Aritificial Intelligence(AI) 
systems struggle when the world changes over time. 

One of the central challenges in both neuroscience and AI is balancing:
- Plasticity - the ability to adapt to new experiences 
- Stability  - the ability to preserve previously learned knowledge

Artificial Neural Networks (ANNs) are known to be highly plastic, but this often comes at the cost of catastrophic 
forgetting, where learning new information disrupts previously acquired knowledge. 

This problem becomes even more severe in continual reinforcement learning, where
- Policies evolve during training,
- Data distribution shifts over time,
- And environmental dynamics themselves may change

## The problem with how continual reinforcement learning is usually studied 
Most prior work in continual RL studies abrupt task switches:
- One environment suddenly becomes another,
- or dynamics change instantaneously

However, real-world environments rarely evolve this way. Instead:
- Terrain gradually becomes slippery
- Robots slowly wear down
- Sensors drift over time
- Bodies change continuously


<figure style="text-align: center;">
<img src="/../assets/img/project_balancing_plasticity_stability/humanoid_motivation.png" alt="Figure 1: Humanoid continual RL setup example." width="75%" height="75%">
<figcaption style="text-align: left; margin-top: 10px;">Figure 1: A comparison 
of continual RL setup where tasks changes are typically abrupt (center) and in our study, we focus on gradual, continuous changes.</figcaption>
</figure>

These forms of non-stationarity are gradual, persistent, and often smooth. This distinction turns out to matter. Abrupt 
changes place enormous demands on rapid adaptation. But under gradual environmental drift, excessive plasticity may 
actually become harmful because the agent overwrites useful knowledge. 

This raises an important question:
Under continuous environmental change, is the main bottleneck really insufficient plasticity - or is it instability? 

Most researchers assume continual learning systems fail because they cannot adapt quickly enough.

Surprisingly, we found the opposite. Under gradual environmental change, the dominant problem is not insufficient 
plasticity, but insufficient stability.

## Introducing naturalistic continuous dynamics drift
To study this problem, we introduced naturalistic continuous changes into standard Mujoco embodiments. Rather than 
switching abruptly between tasks, we continuously perturbed the embodiment mass of the agents by sampling from a noisy 
sinusoidal process. This produced environments whose dynamics evolve smoothly over time. 

Figure 1 is an example based on the humanoid embodiment. This setup (Right in Figure 1) allowed us to investigate 
continual learning under persistent dynamics drift rather than discrete task boundaries.

## Surprisingly, stability matters more than plasticity
We first compared approaches designed to:
- Increase plasticity
- Versus approaaches designed to preserve stability

Plasticity-oriented methods periodically reset subsets of the network parameters to encourage continued adaptation. 
Stability-oriented methods instead rely on consolidation mechanisms that preserve previously learned knowledge by:

- Protecting important synaptic parameters
- or modeling synaptic weight changes across multiple timescales. 

## Key insight
Much of the continual learning literature has focused on restoring plasticity. We expected these methods to excel under 
changing environments.

Instead, the opposite happened. Under gradual and continuous environmental drift, the dominant bottleneck was not 
insufficient plasticity — it was instability. Methods designed to preserve stability consistently outperformed methods 
designed to increase plasticity.

Below we show results for the humanoid embodiment:

<figure style="text-align: center;">
<img src="/../assets/img/project_balancing_plasticity_stability/plasticity_stability_analysis_with_only_qvals_humanoid_full_train_run_forward.png" alt="Figure 2: Results on Humanoid." width="75%" height="75%">
<figcaption style="text-align: left; margin-top: 10px;">Figure 2: A comparison between methods that preserve plasticity or maintain stability when humanoid undergoes continuous mass changes.</figcaption>
</figure>

## Why predictive representations might matter

Among the stability-preserving approaches, the most effective mechanism was a neuro-inspired synaptic consolidation 
model that stabilizes learning across multiple timescales. Initially, this mechanism was applied directly to the 
parameters of the Q-value function.

If stability matters, then preserving memory becomes critical. But not all memories are equally useful. This led us to 
ask a deeper question:

If we can only preserve some aspects of past experience, what should we preserve?
From a computational neuroscience perspective, this is a particularly compelling question because the brain appears to 
build reusable predictive maps of the world — often referred to as cognitive maps — rather than merely storing cached 
action values (Q-values).

If predictive representations such as Successor Features (SFs) capture aspects of these cognitive maps, this raises the 
possibility that biological memory systems may preferentially consolidate predictive structure instead of task-specific 
value estimates, while still maintaining the behavioral flexibility needed for survival.

## Are Successor Features a better target for consolidation?

Interestingly, our results suggest that the answer depends on the severity of the environmental drift. When the 
environmental changes were relatively mild or moderate, consolidating Q-values remained surprisingly effective.

However, when the dynamics evolved to highly severe levels, such as right before the physics simulation became unstable, 
consolidating Successor Features became substantially more effective! 

Under mild drift, the environment remains relatively close to stationary, making cached value estimates sufficiently 
reliable. But as the dynamics increasingly evolve over time, preserving predictive structure becomes more important for 
robust adaptation.

<figure style="text-align: center;">
<img src="/../assets/img/project_balancing_plasticity_stability/humanoid_mass_quantification_analysis.png" alt="Figure 3: Mass quantification analysis from Humanoid." width="75%" height="75%">
<figcaption style="text-align: left; margin-top: 10px;">Figure 3: Quantification of mass changes for the Humanoid embodiment across three levels of mass dynamics variation: mild
(25%), moderate (50%), and severe (100%), corresponding to the maximum change allowed before the physical simulation becomes
unstable.</figcaption>
</figure>

## Why multiple timescales matter
The synaptic consolidation mechanism we used models memory across multiple timescales. Fewer consolidation variables 
correspond to shorter memory horizons, while more variables produce increasingly long-timescale memory traces.

Across the different embodiments, we consistently observed that longer-timescale consolidation improved robustness under 
continual environmental drift.

<figure style="text-align: center;">
<img src="/../assets/img/project_balancing_plasticity_stability/humanoid_timescales.png" alt="Figure 4: Number of consolidation variables analysis." width="75%" height="75%">
<figcaption style="text-align: left; margin-top: 10px;">Figure 4: Number of consolidation variables analysis.</figcaption>
</figure>

But this raised another question:

What roles do the individual timescales actually play during learning?

To investigate this, we introduced a cross-attention mechanism over the multi-timescale Successor Features.

<figure style="text-align: center;">
<img src="/../assets/img/project_balancing_plasticity_stability/humanoid_cross_attention.png" alt="Figure 5: Cross-attention architecture and humanoid results." width="75%" height="75%">
<figcaption style="text-align: left; margin-top: 10px;">Figure 5: Cross-attention architecture (a) and humanoid results (b). 
The set of SFs learned across different timescales are treated as Keys and Values, while the reward params are treated as Queries.</figcaption>
</figure>

The reward representation was treated as the query, while SFs operating at different timescales served as 
keys and values. This allowed us to measure how much each timescale contributed throughout training.

This provides evidence that long-timescale memories are not merely passive storage. Even while the environment changes, 
the agent continues to draw on slowly accumulated predictive knowledge. Thus, these results suggest that slowly changing 
predictive structure remains useful despite persistent non-stationarity.

## Is it simply about having more parameters?

One possible explanation for the improved performance is that the consolidation mechanism simply introduces additional 
parameters. To test this possibility, we scaled the baseline models so that their parameter count matched the 
consolidation-based models.

<figure style="text-align: center;">
<img src="/../assets/img/project_balancing_plasticity_stability/humanoid_capacity_scaling.png" alt="Figure 6: Capacity analysis." width="75%" height="75%">
<figcaption style="text-align: left; margin-top: 10px;">Figure 6: Capacity analysis using Humanoid. Increasing the parameter
count of TD3 and its variants did not consistently improve performance compared to SF + SC (star), suggesting the 
contribution of consolidating SFs beyond network capacity scaling alone</figcaption>
</figure>

Surprisingly, simply increasing model capacity was insufficient.

Even with comparable numbers of parameters, the baseline models still struggled under continuous dynamics drift. This 
suggests that the robustness arises from the consolidation mechanism itself rather than from additional capacity alone.

## Towards lifelong learning systems
For years, continual learning research has largely focused on restoring plasticity. Our results suggest that under 
gradual and persistent change, the more fundamental challenge may be stability.

More importantly, not all knowledge is equally worth preserving. Predictive representations such as Successor Features 
appear to provide a particularly effective target for long-term memory, especially under severe non-stationarity.

More broadly, building AI systems capable of lifelong adaptation may require moving beyond purely plastic learners 
toward architectures that preserve predictive structure across multiple timescales, a principle that biological memory 
systems may have exploited all along.


























