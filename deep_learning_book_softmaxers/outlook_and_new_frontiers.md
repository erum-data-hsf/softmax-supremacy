---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
--- 

# ATTENTION: Outlook and new frontiers on transformers and reformers

This is a researched summary, supported by Google Gemini.

## Efficiency and Scaling 🧠

  The Challenge: Standard Transformers have a computational cost that grows quadratically with the sequence length (O(n2)), making them very slow and memory-intensive for long documents, high-resolution images, or long videos.

  * Mixture of Experts (MoE):
    - [Hugging Face -  Mixture of Experts Explained](https://huggingface.co/blog/moe)
    - [Wikipedia - Mixture of experts](https://en.wikipedia.org/wiki/Mixture_of_experts)
    - [IBM -  What is mixture of experts? ](https://www.ibm.com/think/topics/mixture-of-experts)
    - [Arxiv - A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications](https://arxiv.org/abs/2503.07137)

  * Sparse Attention:
    - [DeepSpeed Sparse Attention](https://www.deepspeed.ai/tutorials/sparse-attention/)
    - [GitHub SparseAttention](https://github.com/kyegomez/SparseAttention)
    - [Arxiv](https://arxiv.org/abs/2502.11089)
    - [Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers](https://arxiv.org/abs/2406.16747)

  * Linear Attention: 
    - [Linear attention is (maybe) all you need (to understand transformer optimization)](https://arxiv.org/abs/2310.01082)
    - [Github](https://github.com/fla-org/flash-linear-attention)
    - [Hailey Scheolkopf](https://haileyschoelkopf.github.io/blog/2024/linear-attn/)
    - [Arxiv](https://arxiv.org/abs/2507.06457)

  * State Space Models (SSM):
    - [hugging face SSM info](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train)
    - [IBM SSM info](https://www.ibm.com/think/topics/state-space-model)

  * Mamba architecture:
    - [wikipedia mambe](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture))
    - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

## Multimodality 🖼️🗣️

  The Challenge: The original Transformer was designed for text. The goal now is to have single models that can seamlessly understand and process information from different sources at once—text, images, audio, and video.

  The Outlook: This is one of the most active areas of research. Models like Google's Gemini or OpenAI's GPT-4o are prime examples. They don't just understand text or images; they can reason across them (e.g., watch a video and answer questions about it). This involves creating "patches" of images or audio that can be treated as tokens, just like words.

  * Multimodal Transformers
    - [Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)
    - [Meta-Transformer: A Unified Framework for Multimodal Learning](https://huggingface.co/papers/2307.10802)

  * Vision Transformer (ViT)
    - [Arxiv: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
    - [Wikipedia](https://en.wikipedia.org/wiki/Vision_transformer)
    - [HuggingFace - Vision Transformer (ViT)](https://huggingface.co/docs/transformers/en/model_doc/vit)

  * Large Multimodal Models (LMMs)
    - [GitHub - Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)

  * CLIP
    CLIP is a is a multimodal vision and language model motivated by overcoming the fixed number of object categories when training a computer vision model.
    - [Hugging Face - CLIP 1](https://huggingface.co/transformers/v4.8.0/model_doc/clip.html)
    - [Hugging Face - CLIP 2](https://huggingface.co/docs/transformers/en/model_doc/clip)

  * Flamingo
    - [Arxiv - Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
    - [DeepMind Flamingo: A Visual Language Model for Few-Shot Learning](https://wandb.ai/gladiator/Flamingo%20VLM/reports/DeepMind-Flamingo-A-Visual-Language-Model-for-Few-Shot-Learning--VmlldzoyOTgzMDI2)

  * Audio Spectrogram Transformer
    - [Hugging Face - Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer)
    - [Arxiv - AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
    - [Arxiv - FAST: Fast Audio Spectrogram Transformer](https://arxiv.org/abs/2501.01104)
    - [Interspeech 2021 - AST: Audio Spectrogram Transformer](https://www.isca-archive.org/interspeech_2021/gong21b_interspeech.pdf)

## Retrieval-Augmented Generation (RAG) 📚

  The Challenge: Transformers can "hallucinate" or provide outdated information because their knowledge is locked into the data they were trained on. Instead of relying solely on internal knowledge, RAG systems connect the Transformer to an external, up-to-date knowledge base (like a search engine or a database). When you ask a question, the system first retrieves relevant documents and then uses the Transformer to generate an answer based on those documents. This makes models more accurate, verifiable, and current.

  * Retrieval-Augmented Generation (RAG)
  * Knowledge Grounding
  * Vector Databases
  * In-context Learning


