---
title: "Machine Unlearning: A Comprehensive Survey"
authors: Weiqi Wang, Zhiyi Tian, Chenhan Zhang, Shui Yu
year: 2024
---
# Abstract
> [!abstract]+
> As the right to be forgotten has been legislated worldwide, many studies attempt to design unlearning mechanisms to protect users' privacy when they want to leave machine learning service platforms. Specifically, machine unlearning is to make a trained model to remove the contribution of an erased subset of the training dataset. This survey aims to systematically classify a wide range of machine unlearning and discuss their differences, connections and open problems. We categorize current unlearning methods into four scenarios: centralized unlearning, distributed and irregular data unlearning, unlearning verification, and privacy and security issues in unlearning. Since centralized unlearning is the primary domain, we use two parts to introduce: firstly, we classify centralized unlearning into exact unlearning and approximate unlearning; secondly, we offer a detailed introduction to the techniques of these methods. Besides the centralized unlearning, we notice some studies about distributed and irregular data unlearning and introduce federated unlearning and graph unlearning as the two representative directions. After introducing unlearning methods, we review studies about unlearning verification. Moreover, we consider the privacy and security issues essential in machine unlearning and organize the latest related literature. Finally, we discuss the challenges of various unlearning scenarios and address the potential research directions.

---

# Personal Summary

Legislation : *Right to be Forgotten* -> remove data, not just from databases, but also from ML models
* hard to evaluate how a data point influenced the training
* naive way: retrain a model from scratch

3 main challenges of Machine Unlearning:  
- stochasticity of training : randomness of influence of data point on training
- incrementality of training : influence of points is done in batches, step by step
- catastrophe of unlearning: models who forget are much worse than retrained models


4 main areas:
- centralized unlearning
- federated unlearning
- unlearning verification
- privacy & security issues in MU

2 main unlearning approaches for _centralized unlearning_:
- Exact unlearning: mathematically remove the points.
- Approximate unlearning: reach with sufficient probability a state of retrained model
---
## Core Idea
>[!important] Objective 
>$$\mathcal{U}(M,D,D_e) \approx \mathcal{A}(D \setminus D_e)$$

## Main Evaluation Metric
- $L_2$**-Norm** : euclidean distance between the weights of the retrained model and the unlearned model
- **Kullback-Leibler divergence (KLD)**: measure the "distance" between the probability distributions of retrained and unlearned models
- **Privacy Leakage Metrics**:
	- membership inference attacks
	- backdoor mia
## Main tools
- Differential Privacy (DP): we want the two models to be $\epsilon$-indistinguishable and $\delta$-approximate
- Bayesian Variational Inference: KLD is intractable so used ELBO instead to minimize KLD between the two distributions
- Privacy Leakage Attacks:
	- membership inference attacks: determine if a sample was employed in the model updating process
	- model inversion attacks: reconstruct input features based on differences between the unlearned model and the base model.


## Research Questions:

>[!question] 
>**[XAI]**: Can XAI techniques that identify "concept vectors" within a model be used to overcome the _incrementality_ challenge? If we can identify that a data point primarily contributed to a specific learned concept (e.g., "the texture of a cat's fur"), could we design an unlearning algorithm that targets and dampens that specific concept vector in the model, rather than trying to reverse a complex sequence of gradient updates?

>[!question]
> **[AML]**: Could an adversary design a "catastrophic unlearning attack"? This would involve submitting data that, while seemingly normal, is crafted to deeply entangle its influence across the model during training. The adversary would then request this data to be unlearned, knowing that its removal is highly likely to trigger catastrophic unlearning and cripple the model.

---
# Centralized Machine Unlearning

## Exact Unlearning vs Approximate Unlearning

>[!note] *Exact Unlearning*
>The gold-standard to achieve, but requires high computational resources. Naive way is to retrain entirely. A famous and effective method (SISA) is based on sharding and splitting the dataset and models into multiple subsets and submodels. Unlearning a sample consists in retraining only the relevant submodel associated with the shard it was in. However, when many samples need to be unlearned, in different submodels, it ceases to be efficient.

>[!note] _Approximate Unlearning_
> It aims to efficiently create a model that is _statistically close_ to a retrained model, without providing a perfect guarantee. It saves significant computation and storage by directly modifying the trained model's parameters. The core challenge is precisely estimating the data's influence to avoid both incomplete unlearning and catastrophic forgetting

Some Exact and Approximate Unlearning methods and paradigms:
- Split Unlearning: works with DNNs but also almost all AI/ML models
	- SISA: involves many shards
	- ARCANE: involves one model per class (one-class classification)
	- Extensions to other use-cases: recommender systems with RecEraser
- Certified Data Removal: use Hessian matrix to evaluate the contribution of the erased data samples for unlearning subtraction (and proceed with update).
	- method that provides a formal, privacy-like guarantee of (ε, δ)-indistinguishability between the unlearned and retrained models.
	- Directly removing the influence of a sample gives information about it. Some technique uses a special (perturbed) loss but that works only for the last training sample.
- Bayesian-based Unlearning: unlearn the approximate posterior from remaining dataset. cannot remove wholly the learned knowledge from the forget set.
- Graph Unlearning. The influence of a single node is not contained within its own features; it propagates to its neighbors and beyond through the graph's edges. Unlearning must account for both node features and this complex connectivity.

>[!note]
>What about _fully_ unlearning a model ?
>What about successively learning and unlearning ?

## Research Questions
>[!question]
>**[AML]**: Since approximate unlearning relies on _estimations_ of a data point's influence (e.g., via the Hessian matrix), could an adversary design "amnesiac-resistant" data? This data would be specifically crafted so that its true influence on the model is systematically underestimated by common approximation techniques, allowing a "ghost" of its influence to persist even after the unlearning process is complete.

>[!question]
>**[XAI]**: In exact unlearning, how can we use XAI to move beyond random data partitioning? Could we first analyze the entire dataset with an XAI tool to understand data inter-dependencies, and then create partitions where the data within each partition is maximally independent of the data in other partitions? This could drastically reduce the number of sub-models that need to be retrained for a given unlearning request.

>[!question]
>**[Certified Data Removal - AML]**: Can an adversary launch a "Hessian-blinding" attack? This would involve crafting data points that create regions in the loss landscape with a near-zero or misleading Hessian value, causing certified removal algorithms to incorrectly estimate their influence as zero and fail to unlearn them.

>[!question] 
>**[Bayesian Unlearning - XAI]**: The posterior distributions in Bayesian methods can be incredibly complex and high-dimensional. Can XAI be used to "translate" the difference between the original and the unlearned posterior distributions into a human-understandable explanation, confirming that the model's "beliefs" have indeed shifted away from the forgotten data?

>[!question]
>**[Graph Unlearning - XAI]**: How can graph-based XAI methods (e.g., GNNExplainer) be adapted to not only explain a prediction but also to trace and visualize the multi-hop influence of a node to be unlearned? This "influence map" could then guide a more precise graph unlearning algorithm.

# Federated Unlearning

The core challenge is that the central server, which orchestrates the training, does not have access to the clients' private data. Therefore, standard retraining is impossible. Most methods focus on enabling the server to remove a specific client's entire contribution by tracking their historical model updates and applying a corrective procedure to the global model. This introduces new challenges in terms of efficiency, communication overhead, and a heightened risk of catastrophic unlearning affecting the entire network.

_skipping ..._

# Unlearning Evaluation and Verification

"How do we prove that unlearning worked?" The survey categorizes verification methods into several types:

- **Mathematical Metrics**: Comparing the unlearned model to a fully retrained model using distance metrics like L2-norm on the weights or KL/JS-Divergence on the output distributions.
- **Privacy-based Audits**: This is a more clever approach that re-frames verification as a privacy problem. The most prominent technique is using
- **Membership Inference Attacks (MIA)**. If an MIA can still determine that the "forgotten" data was part of the training set, the unlearning has failed. A specific method mentioned is
- **Membership Inference via Backdooring (MIB)**, where a user proactively adds a backdoor trigger to their data. To verify unlearning, they test if the trigger still works on the unlearned model

## Research Questions:

>[!question]
> **[AML]**: Can we use the ease (or difficulty) to craft an AE as a way to distinguish whether a model has unlearned the forget set or not?

>[!question]
> **[AML]**: Can we design a powerful new verification method called an "Unlearning-Evasion Attack"? This would be an adversarial attack that is specifically crafted to succeed on a model that has undergone an _imperfect_ approximate unlearning process, but fails on both the original model and a perfectly retrained one. Its success would be a definitive signal of incomplete unlearning.

>[!question]
> **[XAI]**: Can we move beyond binary MIA verification and use XAI to measure the _degree_ of forgetting? For example, by generating an explanation for a prediction on a forgotten data point, we could quantify the "explanatory residue." A high-confidence prediction that still relies on the correct features would indicate poor unlearning, while a low-confidence prediction with a nonsensical explanation would indicate successful unlearning.


# Privacy

The unlearning process, while designed to enhance privacy and security, paradoxically creates new attack surfaces.

- **Privacy Threats**: The difference between a model before and after an unlearning operation can leak a significant amount of information about the _erased data_ itself. An attacker with query access to both model versions can perform powerful inference and reconstruction attacks to recover properties of the forgotten data.
- **Security Threats**: Adversaries can directly attack the unlearning mechanism itself. This includes issuing unlearning requests designed to increase computational costs for the service provider, or uploading customized malicious data and then requesting its removal in a way that harms the model's utility for other users. A particularly insidious attack involves using the unlearning request itself as a means to insert a backdoor.





---
# Extra Notes

## Detailed Notes on "Machine Unlearning: A Comprehensive Survey"

This survey provides a structured and comprehensive overview of the burgeoning field of machine unlearning. The core idea is to enable a trained machine learning model to "forget" a specific subset of the data it was trained on. This is primarily motivated by privacy regulations like the "right to be forgotten" under GDPR, but it also has applications in model maintenance and security.

### Key Takeaways from the Paper

- **Motivation for Unlearning**:
    
    - **Privacy Preservation**: The primary driver is the legal and ethical need to remove a user's data and its influence from a model upon request.
        
    - **Model Utility**: Unlearning can be used to remove outdated, incorrect, or harmful data, thereby improving the model's performance and relevance.
        
    - **Security**: It serves as a mechanism to mitigate the effects of data poisoning and backdoor attacks by removing the malicious data's contribution to the model.
        
- **Fundamental Challenges**: The survey highlights three core difficulties in designing effective unlearning mechanisms:
    
    1. **Stochasticity of Training**: The inherent randomness in modern training processes (especially for deep neural networks) makes the final model state non-deterministic. This complicates the task of precisely calculating and removing the influence of any single data point.
        
    2. **Incrementality of Training**: The training process is sequential. The influence of one data point is dependent on the model's state, which was shaped by previous data points. This makes it difficult to disentangle and surgically remove a specific contribution.
        
    3. **Catastrophic Unlearning**: A significant and common problem where the act of unlearning causes a drastic and often exponential degradation in the model's overall performance, far beyond what would be expected from simply removing a small amount of data.
        
- **Taxonomy of Unlearning Methods**: The paper presents a clear classification of unlearning approaches:
    
    - **Centralized Unlearning**: This is the most developed area and is broken down into two main categories:
        
        - **Exact Unlearning**: This approach aims to produce a model that is identical to a model retrained from scratch on the remaining data. The goal is perfect unlearning, but it often comes at a high computational or storage cost. A common technique is to partition the data and the model, so only affected partitions need to be retrained (e.g., the SISA framework).
            
        - **Approximate Unlearning**: This method seeks to approximate the state of a retrained model, trading perfect accuracy for significant gains in efficiency (both computational and storage). It directly modifies the parameters of the already-trained model.
            
    - **Federated (Distributed) Unlearning**: This applies unlearning to federated learning scenarios, where the central server does not have access to the raw training data. This presents unique challenges, as unlearning must be performed without direct access to the data to be forgotten, often by manipulating the model updates from clients.
        
    - **Unlearning Verification**: This critical area focuses on methods to audit and verify that the unlearning process was successful. This is a non-trivial task, as simply checking for accuracy is insufficient. Methods include mathematical comparisons (L2-norm, KL divergence) and, more interestingly, using privacy attacks like membership inference to test if the model still "remembers" the supposedly forgotten data.
        
    - **Privacy and Security Issues**: The unlearning process itself can introduce new vulnerabilities. The difference in a model's state before and after unlearning can leak information about the data that was removed. Furthermore, the unlearning mechanism can be exploited by adversaries to harm the model.
        

---

## Integrating eXplainable AI (XAI) into Machine Unlearning

eXplainable AI (XAI) focuses on making the decisions and inner workings of AI models understandable to humans. The intersection of XAI and machine unlearning is a fertile ground for research, offering solutions to some of unlearning's biggest challenges.

### Why XAI is Crucial for Unlearning

1. **Enhanced Verification and Auditing**: A key question in unlearning is "How do we know the model has truly forgotten?" Statistical metrics are useful, but they don't provide deep insight. XAI can offer a new level of verification by explaining _why_ a model's predictions have changed. For regulators and users, an explanation that shows a model no longer relies on certain features or reasoning paths associated with their data can be much more convincing than a simple distance score.
    
2. **Diagnosing Catastrophic Unlearning**: When unlearning causes a model's performance to collapse, it's often unclear why. XAI techniques could act as a debugging tool. By generating explanations for the model's (now incorrect) predictions, we can perform a "post-mortem" to understand the unintended consequences of the unlearning algorithm, revealing which parts of the model's "knowledge" were inadvertently damaged.
    
3. **Improving Unlearning Algorithms**: At its core, unlearning is about removing the "influence" of data. XAI techniques, particularly influence functions, are designed to quantify this very thing. By adapting XAI methods, we might be able to create more precise and efficient approximate unlearning algorithms that can surgically target the impact of specific data points, thereby reducing the risk of catastrophic unlearning.
    

## Potential Research Questions (XAI)

>[!question] **Explainable Unlearning Verification**:
>    - How can we adapt feature-attribution methods (like SHAP or LIME) to create "unlearning explanations"? Could we design a framework that shows how the feature importance for a class of predictions changes after unlearning data from that class?
>    - Can counterfactual explanations be used for verification? For example, a successful unlearning verification could be framed as: "The model predicted X, and it would still predict X even if the forgotten data had never existed."

> [!question] **XAI-Guided Unlearning**: 
>  - Can we develop an unlearning algorithm that uses XAI to identify the specific neurons or sub-networks most influenced by the data to be unlearned, and then focus the unlearning process on just those components?
>    - Could influence functions be used not just to approximate the effect of removing data, but to create a more robust unlearning process that is less likely to cause catastrophic forgetting? 

>[!question] **Predictive Risk Assessment**:
>    - Can XAI be used to identify "high-risk" data points _before_ unlearning? That is, data points whose removal is likely to have a disproportionately large and negative impact on the model. This could allow for a more strategic approach to data removal.

---

## Integrating Adversarial Machine Learning (AML) into Machine Unlearning

Adversarial Machine Learning (AML) is the study of attacks on machine learning models and the defenses against them. The unlearning process is a new and interesting attack surface, and it can also be used as a powerful defense.

### Why AML is a Critical Dimension of Unlearning

1. **Securing the Unlearning Process**: As the survey points out, unlearning introduces new security threats. An unlearning mechanism can be seen as an API that modifies a production model. An adversary could exploit this API to degrade the model's performance, introduce biases, or even embed new backdoors. Research in AML is needed to understand and defend against these threats.
    
2. **Unlearning as a Robust Defense**: The paper mentions that unlearning can be used to mitigate backdoor attacks. This is a powerful idea. When a poisoned data point or a backdoor trigger is identified, a robust unlearning mechanism could be the most effective way to "disinfect" the model without the need for complete and costly retraining.
    
3. **Adversarial Verification**: Beyond standard verification, we can use adversarial techniques to probe the limits of unlearning. If a model has truly forgotten a data class, it should not be disproportionately vulnerable to adversarial examples from that class. This provides a more stringent and security-focused method of verification.
    

## Potential Research Questions (AML)

>[!question] **Attacks on Unlearning Mechanisms**:
>- Can an adversary craft a specific data point and a subsequent unlearning request for it that is designed to act as a "Trojan horse," damaging the model in a predictable way?   
>- How can "backdoor poisoning attacks" be tailored to the unlearning process? For instance, could an attacker submit a batch of data and then request the unlearning of a carefully selected subset, with the unlearning process itself being the trigger for a backdoor? 
>- Can an adversary exploit the inaccuracies of approximate unlearning? For example, by creating data whose influence is systematically underestimated by the unlearning algorithm, leading to a "ghost" of the data remaining in the model.

>[!question] **Unlearning for Adversarial Robustness**:
>- How can we design an "online" unlearning system for security, where as soon as a data poisoning attack is detected, the malicious data can be immediately and efficiently unlearned from the deployed model?
>- What is the interplay between unlearning and adversarial training? After unlearning a backdoor, should the model undergo a round of adversarial training to "harden" it against similar future attacks?

>[!question] **Verification through Adversarial Probing**:
>- Can we develop an "adversarial forgetting score"? This would measure a model's robustness to adversarial examples generated from the forgotten data. A successful unlearning process should result in the model being no more vulnerable to these attacks than a model that never saw the data in the first place.
>- Can we create "unlearning-aware" adversarial attacks? These would be inputs specifically designed to fail on a properly unlearned model but succeed on a model where the unlearning was incomplete, thus serving as a powerful auditing tool.

