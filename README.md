# Phenaki
### Note: This is work in progress
## What is Phenaki?
[Phenaki](https://openreview.net/pdf?id=vOEXS39nOF) is a text-to-video model which is very similar to the normal text-to-image models that are learnt in a quantized & compressed latent space. Phenaki introduces a first-stage which spatially & temporally compresses the input videos (e.g. a video of shape 100 x 3 x 256 x 256 -> 20 x 32 x 32). This is achieved by temporal & spatial transformers. An interesting thing to note is that the temporal transformer is autoregressive, which eventually can be used to generate videos with variable length by a shifting context. After learning the first stage which can encode / compress & decode / uncompress videos well, the video-generation model is learned in the latent space. The paper uses [MaskGIT](https://arxiv.org/pdf/2202.04200) for that.

## Open Questions:
- [ ] does the first-stage use a pretrained ViT as proposed in ViViT?
    ANSWERED
- [ ] how to do positional encoding? [current approach](https://github.com/LAION-AI/phenaki/blob/main/vivq.py#L41)
- [ ] best way to construct dataloader for videos?
