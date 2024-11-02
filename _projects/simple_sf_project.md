---
layout: page
title: Learning Successor Features the Simple Way
description: A simple and elegant approach to learning Successor Features for Continual Reinforcement Learning. 
img: assets/img/12.jpg
importance: 1
category: work
related_publications: true
---

Code base for the paper: <a href='https://github.com/raymondchua/simple_successor_features'>https://github.com/raymondchua/simple_successor_features</a>

# 1. Introduction

Deep Reinforcement Learning plays a crucial role in ensuring that intelligent systems can be relied upon to navigate 
complex and non-stationary environments. However, learning representations that are robust towards forgetting and 
interference remains a challenge.  

Successor Features (SFs) offers a promising solution. However, learning SFs directly from complex, high-dimensional inputs, 
such as pixels, can lead to representation collapse, where the representations fail to capture key features in the data.

In this blogpost, we introduce our new approach — Simple Successor Features (Simple SFs) — a streamlined, efficient way 
to learn SFs directly from pixels without requiring pre-training or the use of complex auxiliary losses. Our method not 
only prevents representation collapse but also demonstrates superior performance in mitigating interference, and to a 
lesser extent, forgetting. Simple SFs achieve these results in continual reinforcement learning across 2D and 3D 
environments as well as complex continuous control tasks in Mujoco. 

# 2. What are Successor Features exactly?

In Reinforcement Learning, an agent’s goal is to learn an optimal behavior (commonly known as policy function $\pi$) that 
corresponds to maximizing cumulative rewards. This is done by learning a Q-value function that estimates the future 
rewards for each state-action pair[1]. However, one challenge with this approach is that it can be hard to generalize 
knowledge across tasks, especially when the environment dynamics or the reward structure changes. 

Just to give a short history, SFs are an extension of Successor Representations (SRs), which were initially developed to 
generalize across tasks by focusing on the environment transition dynamics [2]. SRs rely on tabular basis representation
—a lookup table that stores each state individually which limits their scalability. SFs are a distributed, 
function-approximation variant of SRs, allowing replacing the tabular basis representation with a basis features vector, 
making them more suitable for high-dimensional inputs, such as pixels[3]. 

Just like SRs, we can decompose the Q-value function into two distinct components:

1. Successor Features $`( \psi )`$:  These capture the expected occupancy of each state, essentially providing a predictive map on where the agent might end
2. Task encoding $`(\boldsymbol{w})`$: Combined with the basis features, this component helps predict the reward value of a given state.

Mathematically, this means that for each state-action pair \(s,a\) can be defined as the linear combination of $`\psi(s,a)`$ 
and $`\boldsymbol{w}`$:

$$
\begin{align}Q(s,a) = \psi(s,a)^{\intercal}\boldsymbol{w}\end{align}
$$

# 3. Challenges of learning Successor Features from Pixels**

In SRs, the tabular basis representations are usually pre-defined, such as using information about the spatial location 
of the agent to design this representation, which can also be adapted for basis features in SFs. However, in the scenario 
that the basis features have to be learned from  high-dimensional inputs, such as pixels, things start to become tricky. 

The core learning mechanism for the basis features $\phi \in \mathbb{R}^{n}$ and SFs $\psi \in \mathbb{R}^n$ is the 
**SF-Temporal Difference (SF-TD) learning rule**, which updates the successor features based on the agent’s transitions. 
The SF-TD loss is defined as follows:

$$
\begin{align}L_{\phi, \psi} = \frac{1}{2} \left \| \phi(S_{t+1}) + \gamma {\psi}(S_{t+1}, a, \boldsymbol{w})) - 
\psi(S_{t},A_{t}, \boldsymbol{w}) \right \|^2\end{align}
$$

where action $a \sim \pi(S_{t+1})$ and $\gamma \in [0,1]$ is the discount factor. We consider each transition to be 
$(S_t, A_t, S_{t+1}, R_{t+1})$, where $S_t$ is the state at time-step $t$, $A_t$ is the action at time-step $t$, 
$S_{t+1}$ is the next state at time-step $t+1$ and $R_{t+1}$ 

When learning both the basis features $\phi$ and the SFs $\psi$ concurrently, this optimization can lead to representation 
collapse—where the learned features lose their discriminative characteristics across different states. This collapse occurs 
because the loss function is minimized if both $\phi(\cdot)$ and $\psi(\cdot)$ converge to constants across all 
states $S$. Specifically, this happens when $\phi(\cdot) = c_1$  and $\psi(\cdot) = c_2$ with $c_1 = (1-\gamma)c_2$. 

For a more detailed proof, See section 3.4 in the paper. 

To address this issue of representation collapse, some previous approaches have introduced *pretraining* [4] or 
*auxiliary losses* like reconstruction [5] and orthogonality [6] constraints. While these methods can help maintain feature 
diversity, they often add significant computational complexity. In contrast, our approach—**Simple Successor Features (Simple SFs)**
—achieves similar resilience against collapse without requiring these additional components, as we discuss next.

# 4.  Simple SFs: A New Approach

A key insight in designing Simple SFs is that **the basis features $\phi$ should not collapse to a constant**, as this 
would allow the loss in Eq. 2 to be minimized trivially, leading to representation collapse. To address this, rather 
than directly optimizing Eq. 2, we leverage the definition in Eq. 1 and instead optimize the following losses:

1. *Reward prediction loss* guides the task encoding vector $\boldsymbol{w}$ to capture reward-relevant information from the environment. Here, the basis features $\phi$ are treats as a constant:

$$
\begin{align}L_{\boldsymbol{w}} = \frac{1}{2}\left \|  R_{t+1} - \overline{\phi}(S_{t+1})^\top \boldsymbol{w} \right \|^2\end{align}
$$

1. *Q-SF-TD loss* allows the SFs $\psi$  to be learned using a Q-learning like loss, treating the task encoding vector $\boldsymbol{w}$ as a constant learning only the SFs $\psi$:

 

$$
\begin{align}L_{\psi} = \frac{1}{2}\left \| \hat{y} - \psi(S_t, A_t, \boldsymbol{w})^{\top}\boldsymbol{w} \right \|^2\end{align}
$$

By optimizing these losses, we ensure that the task encoding $\boldsymbol{w}$ effectively captures reward-relevant information, 
while the SFs $\psi$ are learned in a stable way without collapse, allowing Simple SFs to learn effectively from high-dimensional 
inputs, such as pixels.

![Architecture for Simple SFs for discrete actions](/../assets/img/project_simple_sf/our_model.png)