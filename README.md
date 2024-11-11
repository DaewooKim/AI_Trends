# Contents
* **[DL Basics](#dl-basics)**
* **[Transformer](#transformer)**
* **[Non-Transformer](#non-transformer)**
* **[LLM](#llm)**
* **[SLM](#slm)**
* **[Foundation Model](#foundation-model)**
* **[AI agent](#ai-agent)**
* **[Training Optimization](#training-optimization)**
* **[Inference Optimization](#inference-optimization)**
* **[Training Infrastructure](#training-infrastructure)**
* **[Inference Infrastructure](#inference-infrastructure)**
* **[HW architecture](#hw-architecture)**
* **[LLM SW framework](#llm-sw-framework)**
* **[AGI](#agi)**
* **[Multi-Modal](#multi-modal)**
* **[Model Compression](#model-compression)**
* **[Personal Tech Blogs](#personal-tech-blogs)**
* **[Big Tech Blogs](#big-tech-blogs)**
* **[Founder's Blogs](#founders-blogs)**
* **[Community](#community)**

# DL Basics

### Lecture

| Date | Title | Homepage |
|---|---|---|
| 2024.08 | Speech and Language Processing (3rd ed. draft) | [paper](https://web.stanford.edu/~jurafsky/slp3/) | 
| 2024.09 | TinyML and Efficient Deep Learning Computing (6.5940, Fall, 2024) | [paper](https://hanlab.mit.edu/courses/2024-fall-65940) |
| 2024.09 | ECE 5545: Machine Learning Hardware and Systems | [paper](https://abdelfattah-class.github.io/ece554) | 

# Transformer 

### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.10 | What Matters in Transformers? Not All Attention is Needed | [paper](https://arxiv.org/pdf/2406.15786) |
| 2024.10 | Differential Transformer | [paper](https://arxiv.org/pdf/2410.05258) |
| 2024.10 | Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces | [paper](https://arxiv.org/pdf/2410.09918) | 
| 2024.11 | TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters | [paper](https://arxiv.org/pdf/2410.23168) | 

### Blog
| Date | Title | Article |
|---|---|---|
| 2021.12 | A Mathematical Framework for Transformer Circuits | [homepage](https://transformer-cuitcuits.pub/2021/framework/index.html) | 
| 2023.12 | Fact Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level (Post 1) | [homepage](https://lesswrong.com/) |
| 2023.12 | Fact Finding: Simplifying the Circuit (Post 2) | [homepage](https://www.lesswrong.com/s/hpWHhjvjn67LJ4xXX/p/3tqJ65kuTkBh8wrRH) | 
| 2023.12 | Fact Finding: Trying to Mechanistically Understanding Early MLPs (Post 3) | [homepage](https://www.lesswrong.com/posts/CW5onXm6uZxpbpsRk/fact-finding-trying-to-mechanistically-understanding-early) 
| 2023.12 | Fact Finding: How to Think About Interpreting Memorisation (Post 4) | [homepage](https://www.lesswrong.com/posts/JRcNNGJQ3xNfsxPj4/fact-finding-how-to-think-about-interpreting-memorisation) |
| 2023.12 | Fact Finding: Do Early Layers Specialise in Local Processing? (Post 5) | [homepage](https://www.lesswrong.com/posts/xE3Y9hhriMmL4cpsR/fact-finding-do-early-layers-specialise-in-local-processing) | 
| 2023.10 | Generative AI exists because of the transformer | [homepage](https://ig.ft.com/generative-ai) | 
| 2024.08 | Transformer Explainer | [homepage](https://poloclub.github.io/transformer-explainer/) | 

# Non-Transformer
### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.07 | The Illusion of State in State-Space Models | [paper](https://arxiv.org/pdf/2404.08819) |
| 2024.10 | Were RNNs All We Needed? | [paper](https://arxiv.org/pdf/2410.01201)|

# LLM
### Paper
| Date | Title | Paper |
|---|---|---|
| 2022.10 | Memorization without Overfitting: Analyzing the Training Dynamics of Large Language Models | [paper](https://arxiv.org/pdf/2205.10770) | 
| 2024.02 | The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits | [paper](https://arxiv.org/pdf/2402.17764) |
| 2024.07 | Distilling System 2 into System 1 | [paper](https://arxiv.org/pdf/2407.06023v1 | 
| 2024.09 | Chain of Thought Empowers Transformers to Solve Inherently Serial Problems | [paper](https://arxiv.org/pdf/2402.12875) |
| 2024.09 | Large Language Monkeys: Scaling Inference Compute with Repeated Sampling | [paper](https://arxiv.org/pdf/2407.21787) |
| 2024.10 | Training Language Models to Self-Correct via Reinforcement Learning | [paper](https://arxiv.org/pdf/2410.12917) |
| 2024.10 | LLMs Know More Than They Show: On The Intrinsic Representation of LLM Hallucinations| [paper](https://arxiv.org/pdf/2410.02707) |
| 2024.10 | On The Planning Abilities of OpenAI's o1 Models: Feasibility, Optimality, and Generalizability |[paper](https://arxiv.org/pdf/2409.19924) | 
| 2024.10 | GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models | [paper](https://arxiv.org/pdf/2410.05229) |
| 2024.10 | Thinking LLMs: General Instruction Following With Thought Generation | [paper](https://arxiv.org/pdf/2410.10630) |
| 2024.10 | Inference Scaling for Long-Context Retrieval Augmented Generation | [paper](https://arxiv.org/pdf/2410.04343) | 
| 2024.10 | GEAR: An Efficient KV Cahce Compression Recipe for Near-Lossless Generative Inference of LLM | [paper](https://arxiv.org/pdf/2403.05527) | 
| 2024.10 | System 2 Thinking in OpenAI’s o1-Preview Model: NearPerfect Performance on a Mathematics Exam | [paper](https://arxiv.org/pdf/2410.07114) |
| 2024.10 | A Comparative Study on Reasoning Patterns of OpenAI's o1 Model | [paper](https://arxiv.org/pdf/2410.13639) |
| 2024.10 | Observational Scaling Laws and the Predictability of Language Model Performance | [paper](https://arxiv.org/pdf/2405.10938) | 
| 2024.10 | A Hitchhiker's Guide to Scaling Law Estimation | [paper](https://arxiv.org/pdf/2410.11840) |
| 2024.10 | LoLCATs: On Low-Rank Linearizing of Large Language Models | [paper](https://arxiv.org/pdf/2410.10254) | 
| 2024.11 | Baysian Scaling Laws For In-Context Learning | [paper](https://arxiv.org/pdf/2410.16531) |
| 2024.11 | Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation | [paper](https://arxiv.org/pdf/2411.00412) | 

### Blog
| Date | Title | Article |
|---|---|---|
| 2024.09 | How OpenAI's o1 changes the LLM training picture - part 1 | [article](https://airtrain.ai/how-openai-o1-changes-the-llm-training-picture-part-1) |
| 2024.10 | How OpenAI's o1 changes the LLM training picture - part 2 | [article](https://airtrain.ai/how-openai-o1-changes-the-llm-training-picture-part-2) | 
| 2024.10 | Noam Brown, Ilge Akkaya & Hunter Lightman of OpenAI’s o1 Research Team on Teaching LLMs to Reason Better by Thinking Longer | [article](https://www.sequoiacap.com/podcast/training-data-noam-brown/) | 
| 2024.10 | aman.ai - OpenAI o1| [article](https://aman.ai/primers/ai/o1/) |
| 2024.10 | Reasoning Series, Part 1: Understanding OpenAI o1 | [article](https://leehanchung.github.io/blogs/2024/10/08/reasoning-understanding-o1/) |
| 2024.10 | Introducing quantized Llama models with increased speed and a reduced memory footprint | [article](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/) |

### Lectures
| Date | Title | Homepage |
|---|---|---|
| 2024 Fall | TinyML and Efficient Deep Learning Computing (MIT) | [homepage](https://hanlab.mit.edu/courses/2024-fall-65940) |
| 2024 Fall | Large Language Models: Methods and Applications (CMU) | [homepage](https://cmu-llms.org/schedule) |
| 2024.10 | LLM Engineer's Handbook: Master the art of engineering Large Lanuage Models from concept to production | [homepage](https://github.com/PacktPublishing/LLM-Engineers-Handbook) | 

# SLM
### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.10 | Small Language Models: Survey, Measurements, and Insights | [paper](https://arxiv.org/pdf/2409.15790) |


# Foundation Model
### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.08 | The Llama 3 Herd of Models (Meta) | [paper](https://arxiv.org/pdf/2407.21783) |
| 2024.10 | Granite 3.0 (IBM) | [paper](https://github.com/ibm-granite/granite-3.0-language-models/tree/main) |


# AI Agent

### Blog
| Date | Title | Article |
|---|---|---|
| 2024.09 | Choosing Between LLM Agent Frameworks | [homepage](https://towardsdatascience.com/choosing-between-llm-agent-frameworks-69019493b259) |

### Lectures
| Date | Title | Homepage |
|---|---|---|
| 2024.09 | Large Language Model Agents MOOC, Fall 2024 | [homepage](https://llmagents-learning.org/f24) |

# Training Optimization

### Paper
| Date | Title | Paper |
|---|---|---|
| 2023.12 | FP8-LM: Training FP8 Large Language Models | [paper](https://arxiv.org/pdf/2310.18313) |
| 2024.10 | Scaling FP8 training to trillion-token LLMs | [paper](https://arxiv.org/pdf/2409.12517v1) |
| 2024.10 | LoRA vs Full Fine-tuning: An Illusion of Equivalence | [paper](https://arxiv.org/pdf/2410.21228) | 

# Inference Optimization

### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.08 | Keep the Cost Down: A Review on Methods to Optimize LLM's KV Cache Consumpution | [paper](https://arxiv.org/pdf/2407.18003) |
| 2024.10 | LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | [paper](https://arxiv.org/pdf/2404.16710) |

### Lectures
| Date | Title | homepage |
|---|---|---|
| 2024.10 | Efficient LLM Deployment and Serving Meetup | [youtube](https://youtu.be/_mzKptPj0hE?list=TLGGGMMXxmJZcj0wNDExMjAyNA) |

# Training Infrastructure

### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.10 | A Survey on Data Synthesis and Augmentation for Large Language Models | [paper](https://arxiv.org/pdf/2410.12896) | 
| 2024.11 | Data Movement Limits To Forntier Model Training | [paper](https://epochai.org/files/limits_to_distributed_training.pdf) |

### Blog
| Date | Title | Article |
|---|---|---|

# Inference Infrastructure

### Paper
| Date | Title | Paper | 👀 |
|---|---|---|---|
| 2024.10 | LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | [paper](https://arxiv.org/pdf/2404.16710) | |

### Blog
| Date | Title | Article |
|---|---|---|
| 2023.12 | Last Mile Data Processing with Ray | [article](https://medium.com/pinterest-engineering/last-mile-data-processing-with-ray-629affbf34ff) |
| 2024.07 | Ray Infrastructure at Pinterest | [article](https://medium.com/pinterest-engineering/ray-infrastructure-at-pinterest-0248efe4fd52) |
| 2024.07 | Ray Batch Inference at Pinterest (Part 3) | [article](https://medium.com/pinterest-engineering/ray-batch-inference-at-pinterest-part-3-4faeb652e385) |

# HW Architecture

### Paper
| Date | Title | Paper | 👀 | 
|---|---|---|---|
| 2024.10 | Addition is All You Need For Energy-Efficient Language Models | [paper](https://arxiv.org/pdf/2410.00907) |  | 
| 2024.10 | Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective | [paper](https://arxiv.org/pdf/2410.04466) |  |

### Blog
| Date | Title | Article |
|---|---|---|
| 2024.07 | The Evolution of AI Capabilities and the Power of 100,000 H100 Clusters | [article](https://www.naddod.com/blog/ai-evolution-100k-h100-clusters) |
| 2024.09 | From H100, GH200 to GB200: How NVIDIA Builds AI Supercomputers with SuperPod | [article](https://www.naddod.com/blog/how-nvidia-builds-ai-supercomputers-with-superpod) |

### Lectures
| Date | Title | Homepage |
|---|---|---|
| 2024.01 | ECE 5545: Machine Learning Hardware and Systems | [homepage](https://abdelfattah-class.github.io/ece5545/) | 

# LLM SW framework
| Date | Title | Homepage |
|---|---|---|
| 2024.10 | bitnet.cpp (official inference framework for 1-bit LLMs) | [homepage](https://github.com/microsoft/BitNet) |

# AGI
### Blog
| Date | Title | Homepage |
|---|---|---|
| 2024.06 | Situational Awareness - Leopold Aschenbrenner | [homepage](https://situational-awareness.ai/) |
| 2024.10 | How Could Machines Reach Human-Level Intellige? | [slide](https://drive.google.com/file/d/1F0Q8Fq0h2pHq9j6QIbzqhBCfTXJ7Vmf4/view?usp=drivesdk) |

# Multi-Modal
| Date | Title | Homepage |
|---|---|---|
| 2024.10 | Introducting Multimodal Llama3.2 | [homepage](https://www.deeplearning.ai/short-courses/introducing-multimodal-llama-3-2/?utm_campaign=metac2-launch&utm_content=314134584&utm_medium=social&utm_source=twitter&hss_channel=tw-992153930095251456) |

# Model Compression

### Paper
| Date | Title | Paper |
|---|---|---|
| 2024.08 | LLM Pruning and Distillation in Practice: The Minitron Approach | [paper](https://arxiv.org/pdf/2408.11796) |

### Blog
| Date | Title | Homepage |
|---|---|---|
| 2024.09 | Fine-tuning LLMs to 1.58bit: extreme quantization made easy | [article](https://huggingface.co/blog/1_58_llm_extreme_quantization) | 

# Personal Tech Blogs
| Title | Homepage |
|---|---|
| Eugene Yan (Senior Applied Scientist at Amazon) | [homepage](https://eugeneyan.com/) |
| Latent Space | [homepage](https://www.latent.space/) | 
| Lycee AI | [homepage](https://www.lycee.ai/blog) |
| Zeta Alpha - Trends in AI blog | [homepage](https://www.zeta-alpha.com/blog-1) | 
| Han, Not Solo | [homepage](https://leehanchung.github.io/blogs/) | 
| Interconnects (Nathan Lambert) | [homepage](https://www.interconnects.ai/) |
| Towards AI Newsletter | [homepage](https://newsletter.towardsai.net/) | 
| AHead of AI by Sebastian Raschka | [homepage](https://magazine.sebastianraschka.com) |  
| Exploring Language Models | [homepage](https://newsletter.maartengrootendorst.com) |
| High Learning Rate | [homepage](https://highlearningrate.substack.com/)|
| Generative AI Newsletter | [homepage](https://newsletter.genai.works/) |
| Maxime Labonne | [homepage](https://maximelabonne.substack.com/p/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172) | 
| Decoding ML Newsletter | [homepage](https://decodingml.substack.com/) | 
| Distilled AI | [homepage](https://aman.ai/primers/ai/) | 
| The AiEdge Newletter | [homepage](https://newsletter.theaiedge.io/) | 

# Big Tech Blogs
| Company | Homepage |
|---|---|
| Meta AI | [homepage](https://ai.meta.com/blog/) |
| Pinterest Engineering | [homepage](https://medium.com/@Pinterest_Engineering) |
| Sionic AI | [homepage](https://blog.sionic.ai/) |
| Liquid AI | [homepage](https://www.liquid.ai/) |

# Founder's Blogs
| People | Homepage |
|---|---|
| Frontier (Doyeob Kim) | [homepage](https://frontierbydoyeob.substack.com/) |

# Community
| Community | Homepage |
|---|---|
| AI 언어보델 로컬 채널 | [homepage](https://arca.live/b/alpaca) |
