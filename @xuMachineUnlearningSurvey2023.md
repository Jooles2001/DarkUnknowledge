---
title: "Machine Unlearning: A Survey"
authors: Heng Xu, Tianqing Zhu, Lefeng Zhang, Wanlei Zhou, Philip S. Yu
year: 2023
---
> [!abstract]+
> Machine learning has attracted widespread attention and evolved into an enabling technology for a wide range of highly successful applications, such as intelligent computer vision, speech recognition, medical diagnosis, and more. Yet a special need has arisen where, due to privacy, usability, and/or the right to be forgotten, information about some specific samples needs to be removed from a model, called machine unlearning. This emerging technology has drawn significant interest from both academics and industry due to its innovation and practicality. At the same time, this ambitious problem has led to numerous research efforts aimed at confronting its challenges. To the best of our knowledge, no study has analyzed this complex topic or compared the feasibility of existing unlearning solutions in different kinds of scenarios. Accordingly, with this survey, we aim to capture the key concepts of unlearning techniques. The existing solutions are classified and summarized based on their characteristics within an up-to-date and comprehensive review of each category’s advantages and limitations. The survey concludes by highlighting some of the outstanding issues with unlearning techniques, along with some feasible directions for new research opportunities. CCS Concepts: • Security and privacy → Human and societal aspects of security and privacy.

---

# Desiderata

>[!definitions] Consistency
> Assume there is a set of samples $X_e$, with the true labels $Y_e$: $\{y_1^e, y_2^e, ..., y_n^e\}$. Let $Y_n: \{y_1^n, y_2^n, ..., y_n^n\}$ and $Y_u: \{y_1^u, y_2^u, ..., y_n^u\}$ be the predicted labels produced from a retrained model and an unlearned model, respectively. If all $y_i^n = y_i^u$ for $1 \le i \le n$, the unlearning process $\mathcal{U}(\mathcal{A}(\mathcal{D}),\mathcal{D},\mathcal{D}_u)$ is considered to provide the consistency property.

>[!definition] Accuracy
> Given a set of samples $X_e$ in the remaining dataset, where their true labels are $Y_e: \{y_1^e, y_2^e, ..., y_n^e\}$. Let $Y_u: \{y_1^u, y_2^u, ..., y_n^u\}$ denote the predicted labels produced by the model after the unlearning process, $w_u = \mathcal{U}(\mathcal{A}(\mathcal{D}),\mathcal{D},\mathcal{D}_u)$. The unlearning process is considered to provide the accuracy property if all $y_i^u = y_i^e$, for $1 \le i \le n$.

>[!definition] Verifiability
> After the unlearning process, a verification function $V(\cdot)$ can make a distinguishable check, that is, $V(\mathcal{A}(\mathcal{D})) \ne V(\mathcal{U}(\mathcal{A}(\mathcal{D}),\mathcal{D},\mathcal{D}_u))$. The unlearning process $\mathcal{U}(\mathcal{A}(\mathcal{D}),\mathcal{D},\mathcal{D}_u)$ can then provide a verifiability property.

---

# Taxonomy

| Schemes                 | Basic Ideas                                                                                                                                                 | Advantages                                                                                                  | Limitations                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Data Reorganization** |                                                                                                                                                             |                                                                                                             |                                                                                                                 |
| Data Obfuscation        | Intentionally adds some choreographed dataset to the training dataset and retrains the model.                                                               | Can be applied to almost all types of models; not too much intermediate redundant data need to be retained. | Not easy to completely unlearn information from models.                                                         |
| Data Pruning            | Deletes the unlearned samples from sub-datasets that contain those unlearned samples. Then only retrains the sub-models that are affected by those samples. | Easy to implement and understand; completes the unlearning process at a faster speed.                       | Additional storage space is required; accuracy can be decreased with an increase in the number of sub-datasets. |
| Data Replacement        | Deliberately replaces the training dataset with some new transformed dataset.                                                                               | Supports completely unlearning information from models; easy to implement.                                  | Hard to retain all the information about the original dataset through replacement.                              |
| **Model Manipulation**  |                                                                                                                                                             |                                                                                                             |                                                                                                                 |
| Model Shifting          | Directly updates model parameters to offset the impact of unlearned samples on the model.                                                                   | Does not require too much intermediate parameter storage; can provide theoretical verification.             | Not easy to find an appropriate offset value for complex models; calculating offset value is usually complex.   |
| Model Replacement       | Replaces partial parameters with pre-calculated parameters.                                                                                                 | Reduces the cost caused by intermediate storage; the unlearning process can be completed at a faster speed. | Only applicable to partial models; not easy to implement and understand.                                        |
| Model Pruning           | Prunes some parameters from already-trained models.                                                                                                         | Easy to completely unlearn information from models.                                                         | Only applicable to partial machine learning models; original model structure is usually changed.                |

---

| Verification Schemes        | Basic Ideas                                                                     | Advantages                                                  | Limitations                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Empirical Evaluation**    |                                                                                 |                                                             |                                                                                                     |
| Retraining-based            | The unlearned model is compared with a retrained model.                         | Easy to implement and understand.                           | Not easy to choose a suitable evaluation metric; retrained models are hard to obtain in some cases. |
| Attack-based                | Attacks a model to verify whether the model has unlearned specific information. | Can provide strong verification from a privacy perspective. | Only applicable to specific unlearning scenarios.                                                   |
| Accuracy-based              | Evaluates accuracy on some specific datasets.                                   | Easy to implement and understand.                           | Hard to show whether a model has completely unlearned information.                                  |
| Relearning-time-based       | Tests the time it takes to relearn information.                                 | Easy to implement and understand.                           | Only applicable to specific machine learning models.                                                |
| **Theoretical Calculation** |                                                                                 |                                                             |                                                                                                     |
| Theory-based                | Verifies the unlearning process based on some theoretical method.               | Can provide strong verification.                            | Hard to understand; sometimes, only provides an upper bound.                                        |
| Information bound-based     | Measures the amount of leaked information through the unlearning process.       | Can provide strong verification from a privacy perspective. | Not easy to implement and understand.                                                               |

---
# Key Takeaways

Machine unlearning is an essential and rapidly growing field, driven by the legal "right to be forgotten" and the practical need to maintain and secure AI models. This survey's core contribution is a clear taxonomy that organizes the diverse unlearning strategies into two main categories: **Data Reorganization** (manipulating the training data itself through pruning or obfuscation) and **Model Manipulation** (directly altering the trained model's parameters through shifting or pruning). The choice between these methods involves a critical trade-off between the level of unlearning guarantee (from exact to approximate) and the efficiency of the process.

# Future Works

