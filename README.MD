## Mapping Censorship in a Chinese LLM through training a Sparse Autoencoder

### Model choice
I went for a Deepseek-based Qwen3-Distill ([deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)) for several reasons:
1. 8 billion params is a reasonable tradeoff where you get meaningful features (instead of everything collapsing into a kind of "feature-soup" with smaller models), but the model still fits into the VRAM of an RTX 4090 at full precision and token requirements remaind handleable.
2. Qwen3-architecture is supported out-of-the-box by TransformerLens/SAELens (important, so I don't spend all my time on doing the tedious bootstrapping)
3. This particular distill displays strong censorship behavior that is easy to elicit -> actually interesting to try & map this behavior to features.

### Methodology
#### 1. Collection
I collect a set of censorship-triggering prompts with the use of a helper model: o4-mini. I chose this one because it was reasonably affordable, sufficiently creative and (this one was weird and annoying) it actually complied with the request to generate these prompts. Other models, including o3(-mini) frequently refused on obscure "security" grounds. Ridiculous!
#### 2. 

   
