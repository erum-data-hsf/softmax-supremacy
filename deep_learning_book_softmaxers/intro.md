# Self-attention for newbies.

This is a small sample book to give you a feel for how book content is
structured.
It shows off a few of the major file types, as well as some sample content.
It does not go in-depth into any particular topic - check out [the Jupyter Book documentation](https://jupyterbook.org) for more information.


```{tableofcontents}
```
## Introduction

In order to apply neural networks to the analysis of text, speach, translation
and time series datasets initially Recurrent Neural Networks (RNN) were used.
They have in common that they process input in  one step at a time while
carrying a hidden state that remembers past information. It makes RNNs in the
end simple memory-based sequence models, which can for example output
predictions like the next word in a text. Limitations were connected to long
sequences which lead to vanishing gradients. They need to run also step by step
which leads to long training time. An improvement were Long Short-Term
Memory Networks (LSTM) which can remember information over long sequences.
They added gates that include information to keep, update or forget past
information. Each LSTM cell has 
- Forget gate → what old info to throw away.
- Input gate → what new info to add.
- Cell state → long-term memory.
- Output gate → what to output for this step.

This reduced the vanishing gradient problems of RNNs.
However LSTMs are still sequential so that it shows still long training times
on large sequences and it is hard to capture very long-range dependencies.

In 2017 a new type of neural network architecture was introduced with the
paper "Attention Is All You Need" so called transformer. 
It is the foundation of modern AI models like GPT (ChatGPT), BERT, and Vision Transformers (ViT).
The Core Idea is a mechanism called  "Attention". Instead of reading text word
by word like in RNNs or looking at small patches like in CNNs, Transformers use self-attention. Each word (or token) can directly “look at” all other words in
the input. The model decides which words are most important for understanding
the meaning. An example is shown in “The dog chased the ball because it was
fast”, the Transformer can figure out that “it” refers to “dog” by looking at
all words at once.
