# PSU_CMPSC448_Midterm_Project

Project Overview: As Large Language Models (LLMs) such as GPT, Claude, Gemini, Llama, and Qwen increasingly generate the text we read online, we face an emerging digital authorship crisis: Can we still tell who—or what—wrote a piece of text? Different LLMs may leave distinct linguistic, stylistic, and structural “fingerprints” in their responses, raising an intriguing question: Can machine learning identify the AI behind the text? 

In this project, each team can obtain or construct data with at least three components: “(LLM_name, LLM_Input, LLM_output)”. Your dataset should contain responses from at least 3 different LLM families. Examples include,
but are not limited to: GPT, Claude, Gemini, Llama, Qwen, DeepSeek, other LLMs. You may obtain data from: publicly released datasets; publicly available LLM outputs; LLM APIs or interfaces; interactions that you generate yourself; your own LLM interactions that you are comfortable using for this project. You should clearly document where your data came from and how it was collected or constructed. Do not include private, sensitive, confidential, copyrighted-inappropriately-reproduced, or third-party information that you do not have the right to use. Try to keep the number of examples from different LLMs reasonably balanced.

Required Models: You must implement and evaluate the following two models: Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN, like LSTM). 
You may additionally experiment with other classifiers or combine models to obtain better performance. However, CNN and RNN are required.

RQ Questions Explored in this project:
-RQ1 — Can we identify which LLM generated a response?
Use LLM_output as input, conduct classification to detect which LLM generated it. Report the performance of your CNN&RNN performance
-RQ2 — Does the user's prompt help identify the LLM?
Compare compare classifiers using: Input only; Output only; Input + Output
-RQ3 — Do LLM fingerprints generalize across tasks or domains?
Investigate whether a classifier trained on one type of task can identify LLMs on another type of task.

Packages used:
PyTorch, NumPy, pandas, sklearn, huggingface (for dataset)
