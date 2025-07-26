| **Hook string** | **Captures** | **Typical interpretability use** |
| --- | --- | --- |
| blocks.L.attn.hook_q, .k, .v | query/key/value vectors | find which tokens attend during steering |
| blocks.L.attn.hook_z | attention scores before softmax | spot pattern-based circuits |
| blocks.L.mlp.hook_pre | MLP input | less common |
| blocks.L.mlp.hook_post | MLP output (after activation) | most used for feature extraction |