# Mapping Censorship in a Chinese LLM through training a Sparse Autoencoder
---
## Model choice

### Subject
I went for a Deepseek-based Qwen3-Distill ([deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)) for several reasons:
1. 8 billion params is a reasonable tradeoff where you get meaningful features (instead of everything collapsing into a kind of "feature-soup" with smaller models), but the model still fits into the VRAM of an RTX 4090 at full precision and token requirements remaind handleable.
2. Qwen3-architecture is supported out-of-the-box by TransformerLens/SAELens (important, so I don't spend all my time on doing the tedious bootstrapping)
3. This particular distill displays strong censorship behavior that is easy to elicit -> actually interesting to try & map this behavior to features.

### Generator & helper models
* Generator: I chose o4-mini because it was reasonably affordable, sufficiently creative and (this one was weird and annoying) it actually complied with the request to generate these prompts. Other models, including o3(-mini) frequently refused on obscure "security" grounds. Ridiculous!
* Evaluator: gpt-4.1-mini. Cheap (no o5 was available at the time), relatively fast and reliable.
---
### Methodology

### Prompts / training token aggregation
1. I collect a set of censorship-triggering prompts with the use of o4-mini
2. We deduplicate, then pass the prompt on to the subject model, then evaluate & categorize the response with gpt-4.1.-mini (is it censored and if so, which type of censorship does it exhibit?)
3. Afterwards, storage in chromadb. Cheap, easy to setup. Would probably use Qdrant in a more professional context and to retain speed with larger dataset sizes.
4. I use the cerebras/SlimPajama-627B as a negative-dataset (random text, essentially). I only sample from it, to get between 5 and 10 million tokens for training.
5. I built a simple dashboard to visualize prompt diversity as well as censorship category distribution. Not very performant ATM when the chromadb has a few thousand entries and isn't running locally, but that's primarily a lookup-latency/caching issue.

### Activation capture
I initially went ahead and captured the activations in a dedicated step

   
