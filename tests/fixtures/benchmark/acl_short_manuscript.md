# Language Models Do Hard Arithmetic Tasks Easily and Hardly Do Easy Arithmetic Tasks

Andrew Gambardella\* Yusuke Iwasawa Yutaka Matsuo

University of Tokyo

## Abstract

The ability (and inability) of large language models (LLMs) to perform arithmetic tasks has been the subject of much theoretical and practical debate. We show that LLMs are frequently able to correctly and confidently predict the first digit of n-digit by m-digit multiplication tasks without using chain of thought reasoning, despite these tasks require compounding operations to solve. Simultaneously, LLMs in practice often fail to correctly or confidently predict the last digit of an n-digit by m-digit multiplication, a task equivalent to 1-digit by 1- digit multiplication which can be easily learned or memorized. We show that the latter task can be solved more robustly when the LLM is conditioned on all of the correct higher-order digits, which on average increases the confidence of the correct last digit on 5-digit by 5-digit multiplication tasks using Llama 2-13B by over 230% (0.13â†’0.43) and Mistral-7B by 150% (0.22â†’0.55).

## 1 Introduction

The development of large language models (LLMs) (Brown et al., 2020) has given new life to the deep learning revolution, and seen mass adoption within not just the scientific community, but also society at large. These LLMs, being the first known â€œgeneralâ€ machine learning model developed by humanity (Morris et al., 2024), have been applied to various tasks dealing with natural language such as those commonly encountered in school curricula (Hendrycks et al., 2021), and even branching off into tasks such as text-to-image generation (Saharia et al., 2022) and hierarchical planning (Wang et al., 2023).

Despite the generality and far-reaching consequences of LLMs, there are still many significant limitations making difficult the direct application of LLMs to certain tasks. One such limitation is the poor performance of LLMs on arithmetic tasks, such as elementary addition, subtraction, multiplication, and division (Nogueira et al., 2021). Not only do modern LLMs perform poorly on these tasks, but some tasks such as n-digit by m-digit multiplication and division, which require compounding operations to solve, appear to be unlearnable by pure autoregressive transformer architectures unless they decompose the problem into multiple steps, such as with chain of thought reasoning (Wies et al., 2022; Liu et al., 2023). As such, several solutions have been proposed, such as fine-tuning so that chain of thought reasoning is automatically used for problems which require compounding operations (Liu et al., 2023; Kojima et al., 2022) or fine-tuning to call outside tools, such as a calculator (Schick et al., 2024).

While we most likely cannot expect simply training models with more parameters to allow for the solving of tasks which require compounding operations without chain of thought, we believe that analyzing the limitations and abilities of autoregressive LLMs when attempting to solve these tasks directly may shed light on unknown properties of LLMs. We therefore use Monte Carlo Dropout (MC Dropout) (Gal and Ghahramani, 2016) to analyze the performance of LLMs which were trained with dropout and which have open weights available, such as Llama 2 (Touvron et al., 2023) and Mistral (Jiang et al., 2023), in carrying out arithmetic tasks.

MC Dropout allows one to interpret neural networks which were trained with dropout as Bayesian neural networks, as neural networks trained with dropout have been shown to be equivalent to a Bayesian approximation to a Gaussian process. This allows one to obtain empirical Bayesian confidence distributions over neural network weights or outputs by doing multiple forward passes through the neural network with dropout on, during test time (Gal and Ghahramani, 2016). MC Dropout is one of many ensemble-based methods for uncertainty quantification (Ovadia et al., 2019; Ashukha et al., 2020), and has been applied to analyze the confidence of transformer architectures (Shelmanov et al., 2021) and to implement tree-based LLM prompting (Mo and Xin, 2023).

Our results when applying MC Dropout to Llama 2 and Mistral in arithmetic tasks were surprising. We found that all models could confidently and correctly predict the first digit result of n-digit by m-digit multiplication problems, despite it most likely being impossible for any autoregressive LLM to have learned a general algorithm for doing so without decomposing the problem into multiple steps, as finding this digit in general requires solving the entire multiplication problem1. We also found that all models struggled to correctly output the last digit of n-digit by m-digit multiplication problems, despite it being very easy to learn an algorithm for doing so, as calculating the last digit is equivalent to 1-digit by 1-digit multiplication. Finally, we show that the confidence of LLMs in predicting the last digit can be increased by conditioning the generation of the last digit on the correct intervening digits, despite the computation of the last digit not depending on the correct computations of the higher-order digits at all.

## 2 Experiments

We evaluate the HuggingFace (Wolf et al., 2019) implementations of Llama 2-7B, Llama 2-13B, and Mistral-7B (Touvron et al., 2023; Jiang et al., 2023) in 2-shot settings, where the 2-shot examples are of correct n-digit by m-digit multiplications. Sections 2.1 and 2.2 show results on the 3-digit by 3-digit multiplication task 592 392, and averages over multiple problems with varying digit length are provided in Section 2.3. Details about the prompt and hyperparameters are given in Appendix A, details about the tokenizers for the models are given in Appendix B, and details about the use of dropout in the training of the models is given in Appendix C.

## 2.1 Unconditional Answer Generation

We first study a version of the problem in which the answer is generated with the language model conditioned on the few shot examples and the problem to be solved, but is provided with none of the digits to be generated (i.e., the normal few-shot arithmetic scenario), which we refer to as â€œunconditionalâ€ generation in an abuse of terminology. Our main results for these experiments are in Figures 1 and 2.

In Figure 1 we can see that both Llama 2-7B and Llama 2-13B can confidently and correctly predict the first digit of the 3-digit by 3-digit multiplication task 592  392, which equals 232064. This should be surprising as it is not immediately apparent from the problem that the first digit of the solution should be 2, and the only way to discover this is to compute the multiplication. As LLMs most likely cannot perform n-digit by m-digit multiplication in the general case without decomposing the problem into steps, the output of the first digit in this case is unlikely to be the output of a multiplication algorithm learned by the LLM.

![](images/0d09f62ed04b008cbfd2e76976a5d3167aa464d4c238a3873dd80b4d418bec83.jpg)

<details>
<summary>histogram</summary>

| Tokens | Frequency |
| :--- | :--- |
| 1 | 3 |
| 2 | 98 |
</details>

![](images/e13354cc1d2e9c2c790edb26d46bbf0b2b170e697bb49971348cc777e80cded3.jpg)

<details>
<summary>histogram</summary>

| Tokens | Frequency |
| ------ | --------- |
| 1      | 3         |
| 2      | 89        |
| 3      | 1         |
| 5      | 1         |
| 7      | 5         |
| t      | 4         |
</details>

Figure 1: Confidence and accuracy of Llama 2-7B and Llama 2-13B predicting the first digit of the result of 592 392. Both language models are able to confidently and correctly predict that the first digit should be 2, despite this not being immediately apparent from the problem.

Conversely, in Figure 2, we can see that both Llama 2-7B and Llama 2-13B can neither confidently nor correctly predict the last digit of the same problem, despite doing so being equivalent to 1-digit by 1-digit multiplication. This is a case in which any reasonable model should be able to confidently and correctly solve the task, as not only could the algorithm to solve the task be learned by an autoregressive language model, but the information needed to solve this task could also very easily be memorized by language models with billions of weights.

![](images/880db339aeaff8ecea786477506c44ceb9d2930d76b9206f955c7ab8a9c9ba05.jpg)

<details>
<summary>histogram</summary>

| Tokens Range | Frequency |
| ------------ | --------- |
| -1 to 0      | 3         |
| 0 to 2       | 40        |
| 2 to 4       | 15        |
| 4 to 6       | 13        |
| 6 to 8       | 15        |
| 8 to 9       | 13        |
| 9 to 10      | 1         |
</details>

![](images/f25561f5e3447cf05ad54498982c567f56bee88002f3be0a6fae9736ed369423.jpg)

