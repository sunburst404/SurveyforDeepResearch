# SurveyforDeepResearch
### 准备阅读的论文

##### ✅A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications

[A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications](https://arxiv.org/pdf/2506.12594v1)

> This survey examines the rapidly evolving field of Deep Research systems -- AI-powered applications that automate complex research workflows through the integration of large language models, advanced information retrieval, and autonomous reasoning capabilities. We analyze more than 80 commercial and non-commercial implementations that have emerged since 2023, including OpenAI/Deep Research, Gemini/Deep Research, Perplexity/Deep Research, and numerous open-source alternatives. Through comprehensive examination, we propose a novel hierarchical taxonomy that categorizes systems according to four fundamental technical dimensions: foundation models and reasoning engines, tool utilization and environmental interaction, task planning and execution control, and knowledge synthesis and output generation. We explore the architectural patterns, implementation approaches, and domain-specific adaptations that characterize these systems across academic, scientific, business, and educational applications. Our analysis reveals both the significant capabilities of current implementations and the technical and ethical challenges they present regarding information accuracy, privacy, intellectual property, and accessibility. The survey concludes by identifying promising research directions in advanced reasoning architectures, multimodal integration, domain specialization, human-AI collaboration, and ecosystem standardization that will likely shape the future evolution of this transformative technology. By providing a comprehensive framework for understanding Deep Research systems, this survey contributes to both the theoretical understanding of AI-augmented knowledge work and the practical development of more capable, responsible, and accessible research technologies. The paper resources can be viewed at [this https URL](https://github.com/scienceaix/deepresearch).

📢这篇survey的介绍deep research的概念，区别于LLm的方面我觉得还是很清晰的，deep research的为什么具有强大的推理能力也是解释很清晰，强调其具备**自主工作流能力**、**专用研究工具集成**和**端到端研究编排**三大核心特征。其调查了很多的deep research平台，包括了OpenAI、Gemini、Perplexity、dzhng和一些开源的/deep-research，并且提出每一个deep research该有的特点时都会结合具体例子阐述。
但是框架我觉得不是很清楚，三个框架的图只是功能的堆砌；deep research的流程也未涉及，评估我觉得也有点主观了。



##### ✅Deep Research System Card 

https://cdn.openai.com/deep-research-system-card.pdf

📢本身的技术实现是一点不给看啊，光是评估自己的deep research风险还行和数据集有更新（更新了啥也不晓得），没啥危害性所以可以给pro用户商用使用。**评估方法**可以借鉴。



##### ✅A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges

[[2508.05668v1\] A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges](https://arxiv.org/abs/2508.05668v1)\

📢❎这篇survey主要聚焦的是 **Search Agents搜索信息**的方法、优化agents架构方法、深度信息挖掘方面的能力，而Deep Research 代表了一种更先进的代理式实现，具备多步推理、代码执行和动态路径调整等能力。没怎么读



##### ✅Toolformer: Language Models Can Teach Themselves to Use Tools

[2302.04761](https://arxiv.org/pdf/2302.04761)NeurIPS 2023
训练LLMs使用工具api

Toolformer，通过自我监督的方式学习如何使用不同的工具，例如搜索引擎、计算器和翻译系统，方法是通过简单的**API调用**。这是通过对大量经过过滤的API调用进行微调实现的。在数学上比较优异，但在QA上提升一般，但是提出了“让LLMs学习怎么使用工具”推动了DEEP RESAEARCH的发展。



##### ✅DEEP RESEARCH AGENTS: A SYSTEMATIC EXAMINATION AND ROADMAP

https://arxiv.org/pdf/2506.18096

📢💥💥💥对构成深度研究代理的基础技术、架构组件、未来的路线图进行了详细分析，我觉得讲的挺清晰的，比较凝练。没太多废话。





##### ✅Deep Research: A Survey of Autonomous Research Agents

[Deep Research: A Survey of Autonomous Research Agents](https://arxiv.org/pdf/2508.12752)

> Abstract: The rapid advancement of large language models (LLMs) has driven the development of agentic systems capable of autonomously performing complex tasks. Despite their impressive capabilities, LLMs remain constrained by their internal knowledge boundaries. To overcome these limitations, the paradigm of deep research has been proposed, wherein agents actively engage in planning, retrieval, and synthesis to generate comprehensive and faithful analytical reports grounded in web-based evidence. In this survey, we provide a systematic overview of the deep research pipeline, which comprises four core stages: planning, question developing, web exploration, and report generation. For each stage, we analyze the key technical challenges and categorize representative methods developed to address them. Furthermore, we summarize recent advances in optimization techniques and benchmarks tailored for deep research. Finally, we discuss open challenges and promising research directions, aiming to chart a roadmap toward building more capable and trustworthy deep research agents. 
>

📢这篇主要是说明他们提出的deep research所拥有的四个比较有特色的流程：planning, question developing, web exploration, and report generation。
文中说了与前两篇综述的关系：以**能力形成和集成为中心**，揭示方法之间的深层联系，为演化设计范式提供洞察。所以我读起来的感觉就是全文就只有这四个方面如何工作的、有哪些工作方式和可有可无的公式，不过大体上还是很清晰但是真正的内容感觉很少。





##### Universal Deep Research: Bring Your Own Model and Strategy

[Universal Deep Research: Bring Your Own Model and Strategy](https://arxiv.org/pdf/2509.00244)https://arxiv.org/search/?searchtype=author&query=Molchanov%2C+P)   自定义deep research的研究策略

> Abstract: Deep research tools are among the most impactful and most commonly encountered agentic systems today. We observe, however, that each deep research agent introduced so far is hard-coded to carry out a particular research strategy using a fixed choice of tools. We introduce Universal Deep Research (UDR), a generalist agentic system that wraps around any language model and enables the user to create, edit, and refine their own entirely custom deep research strategies without any need for additional training or finetuning. To showcase the generality of our system, we equip UDR with example minimal, expansive, and intensive research strategies, and provide a user interface to facilitate experimentation with the system.
>



##### Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training

[Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training](https://arxiv.org/pdf/2508.00414)    腾讯ai lab的一个开源的deep research agent



##### DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments

[[2504.03160\] DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)   上交和其他实验室 开源的

> Large Language Models (LLMs) equipped with web search capabilities have demonstrated impressive potential for deep research tasks. However, current approaches predominantly rely on either manually engineered prompts (prompt engineering-based) with brittle performance or reinforcement learning within controlled Retrieval-Augmented Generation (RAG) environments (RAG-based) that fail to capture the complexities of real-world interaction. In this paper, we introduce DeepResearcher, the first comprehensive framework for end-to-end training of LLM-based deep research agents through scaling reinforcement learning (RL) in real-world environments with authentic web search interactions. Unlike RAG-based approaches that assume all necessary information exists within a fixed corpus, our method trains agents to navigate the noisy, unstructured, and dynamic nature of the open web. We implement a specialized multi-agent architecture where browsing agents extract relevant information from various webpage structures and overcoming significant technical challenges. Extensive experiments on open-domain research tasks demonstrate that DeepResearcher achieves substantial improvements of up to 28.9 points over prompt engineering-based baselines and up to 7.2 points over RAG-based RL agents. Our qualitative analysis reveals emergent cognitive behaviors from end-to-end RL training, including the ability to formulate plans, cross-validate information from multiple sources, engage in self-reflection to redirect research, and maintain honesty when unable to find definitive answers. Our results highlight that end-to-end training in real-world web environments is not merely an implementation detail but a fundamental requirement for developing robust research capabilities aligned with real-world applications. We release DeepResearcher at [this https URL](https://github.com/GAIR-NLP/DeepResearcher).



##### AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents

[[2502.05957\] AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents](https://arxiv.org/abs/2502.05957)  港大的开源项目，应该是集成了之前的deep research项目

> Large Language Model (LLM) Agents have demonstrated remarkable capabilities in task automation and intelligent decision-making, driving the widespread adoption of agent development frameworks such as LangChain and AutoGen. However, these frameworks predominantly serve developers with extensive technical expertise - a significant limitation considering that only 0.03 % of the global population possesses the necessary programming skills. This stark accessibility gap raises a fundamental question: Can we enable everyone, regardless of technical background, to build their own LLM agents using natural language alone? To address this challenge, we introduce AutoAgent-a Fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone. Operating as an autonomous Agent Operating System, AutoAgent comprises four key components: i) Agentic System Utilities, ii) LLM-powered Actionable Engine, iii) Self-Managing File System, and iv) Self-Play Agent Customization module. This lightweight yet powerful system enables efficient and dynamic creation and modification of tools, agents, and workflows without coding requirements or manual intervention. Beyond its code-free agent development capabilities, AutoAgent also serves as a versatile multi-agent system for General AI Assistants. Comprehensive evaluations on the GAIA benchmark demonstrate AutoAgent's effectiveness in generalist multi-agent tasks, surpassing existing state-of-the-art methods. Furthermore, AutoAgent's Retrieval-Augmented Generation (RAG)-related capabilities have shown consistently superior performance compared to many alternative LLM-based solutions.
>



##### OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation

[[2505.23885\] OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation](https://arxiv.org/abs/2505.23885)港大和camel-ai 做的开源

Large Language Model (LLM)-based multi-agent systems show promise for automating real-world tasks but struggle to transfer across domains due to their domain-specific nature. Current approaches face two critical shortcomings: they require complete architectural redesign and full retraining of all components when applied to new domains. We introduce Workforce, a hierarchical multi-agent framework that decouples strategic planning from specialized execution through a modular architecture comprising: (i) a domain-agnostic Planner for task decomposition, (ii) a Coordinator for subtask management, and (iii) specialized Workers with domain-specific tool-calling capabilities. This decoupling enables cross-domain transferability during both inference and training phases: During inference, Workforce seamlessly adapts to new domains by adding or modifying worker agents; For training, we introduce Optimized Workforce Learning (OWL), which improves generalization across domains by optimizing a domain-agnostic planner with reinforcement learning from real-world feedback. To validate our approach, we evaluate Workforce on the GAIA benchmark, covering various realistic, multi-domain agentic tasks. Experimental results demonstrate Workforce achieves open-source state-of-the-art performance (69.70%), outperforming commercial systems like OpenAI's Deep Research by 2.34%. More notably, our OWL-trained 32B model achieves 52.73% accuracy (+16.37%) and demonstrates performance comparable to GPT-4o on challenging tasks. To summarize, by enabling scalable generalization and modular domain transfer, our work establishes a foundation for the next generation of general-purpose AI assistants.



其他开源deep research项目  Aworld，openmanus



##### Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models

[[2411.19443\] Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models](https://arxiv.org/abs/2411.19443)

> Iterative retrieval refers to the process in which the model continuously queries the retriever during generation to enhance the relevance of the retrieved knowledge, thereby improving the performance of Retrieval-Augmented Generation (RAG). Existing work typically employs few-shot prompting or manually constructed rules to implement iterative retrieval. This introduces additional inference overhead and overlooks the remarkable reasoning capabilities of Large Language Models (LLMs). In this paper, we introduce Auto-RAG, an autonomous iterative retrieval model centered on the LLM's powerful decision-making capabilities. Auto-RAG engages in multi-turn dialogues with the retriever, systematically planning retrievals and refining queries to acquire valuable knowledge. This process continues until sufficient external information is gathered, at which point the results are presented to the user. To this end, we develop a method for autonomously synthesizing reasoning-based decision-making instructions in iterative retrieval and fine-tuned the latest open-source LLMs. The experimental results indicate that Auto-RAG is capable of autonomous iterative interaction with the retriever, effectively leveraging the remarkable reasoning and decision-making abilities of LLMs, which lead to outstanding performance across six benchmarks. Further analysis reveals that Auto-RAG can autonomously adjust the number of iterations based on the difficulty of the questions and the utility of the retrieved knowledge, without requiring any human intervention. Moreover, Auto-RAG expresses the iterative retrieval process in natural language, enhancing interpretability while providing users with a more intuitive experience\footnote{Code is available at \url{[this https URL](https://github.com/ictnlp/Auto-RAG)}.



##### Open Data Synthesis For Deep Research

[Open Data Synthesis For Deep Research](https://arxiv.org/pdf/2509.00375)            引入了InfoSeek，这是一个可扩展的框架，用于综合复杂的深度研究任务。将问题分解成子问题、协调多步推理。



##### Deep Research Bench: Evaluating AI Web Research Agents

[2506.06287](https://arxiv.org/pdf/2506.06287)评估文章



##### Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents

[Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents](https://arxiv.org/pdf/2507.05495)               深度研究比较器



### deep research 实例

1. https://chatgpt.com/share/67b142cb-030c-8011-b0c8-8a1071dd8094 
   [OpenAI Deep Research 案例分享 - 深度研究技术综合研究报告 | Yue Shui 博客](https://syhya.github.io/zh/posts/2025-02-15-deep-research-tech-report/)


   一个博主分享的用OpenAI Deep Research写报告的实例（*对“深度研究”技术进行深入研究，并编写一份综合研究报告*）

2. [OpenManus/examples/use_case at main · FoundationAgents/OpenManus · GitHub](https://github.com/FoundationAgents/OpenManus/tree/main/examples/use_case)
   官网的图片显示不出来，github上又只有这一个例子
   [OpenManus 项目概览 | mannaandpoem/OpenManus - KoalaWiki](https://opendeep.wiki/mannaandpoem/OpenManus/overview)

3. https://github.com/langchain-ai/open_deep_research?tab=readme-ov-file
   langchain 一个开源的deep research框架，调用其他LLM的api key 实现深度研究
   类似的，dify低代码平台

4. https://www.doubao.com/chat
   豆包免费，



### 其他

我在other中添加了使用langchain/open_deep_research的记录供参考



### 每日进度

| 时间      |                                                              |
| --------- | ------------------------------------------------------------ |
| 2025.9.1  | 阅读A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications，和之前读的一起加起来读到了第四章。papers文件夹里有论文原文与批注、论文概况。 |
| 9.2       | 看到第七章                                                   |
| 9.3-9.4   | 看完余下章节，总结了一下全篇（主要是前几章）                 |
| 9.5-9.6   | 看完openai的Deep Research System Card ，了解了一些评估与其能力、存在风险 |
| 9.7-9.8   | 又找了一些论文和开源的项目准备看，看部分了DEEP RESEARCH AGENTS:A SYSTEMATIC EXAMINATION AND ROADMAP，比第一篇survey清晰一些，框架更为清楚但具体细节未交代需要找开源的看。 |
| 9.9       | 读完了DEEP RESEARCH AGENTS:A SYSTEMATIC EXAMINATION AND ROADMAP |
| 9.11      | 看了一些实例，用豆包试了一下deep research                    |
| 9.12      | 阅读Toolformer论文（让agent学会在何种情况使用api）           |
| 9.13-9.14 | 读了综述论文Deep Research: A Survey of Autonomous Research Agents，我觉得明天可以看代码的实际流程了 |
| 9.16-9.17 | 读了langchain的open deep research项目的部分代码，其实我感觉就是提示词工程和适当的工具调用 |
| 9.18-9.21 | 断断续续的读完了open deep research的主要代码，知道了流程（papers文件夹的画了简易的流程图-xmind文件），最主要的就是发给llm的提示词和工具的调用（当然也包括”在哪“和使用什么工具）<br />这两天也把这个项目部署了一下，默认的openai可以使用；goole的gemini有些问题；使用openrouter可以使用qwen模型，但是要花钱；阿里百炼平台使用qwen-plus的api会出现格式问题（qwen不支持openai的兼容模式，而langchain/open deep research使用的是openai的格式），改的地方还无从下手<br />也可以本地部署，官方说的是使用ollama管理本地llm，这个还没开始研究。 |







### 有困难的地方

| 时间 | 位置                        | 具体问题                                                     | 是否解决 |
| ---- | --------------------------- | ------------------------------------------------------------ | -------- |
| 9.1  | 关于deep research的实际用例 | 我还没有用过，得花时间试一下比如perplexity，openai deep research | yes      |
| 9.6  | deep research框架           | 还要找个开源的看看框架，流程之类的                           | yes      |
|      |                             |                                                              |          |
|      |                             |                                                              |          |
