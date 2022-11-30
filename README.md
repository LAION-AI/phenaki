# Phenaki
### Note: This is work in progress
## What is Phenaki?
[Phenaki](https://openreview.net/pdf?id=vOEXS39nOF) is a text-to-video model which is very similar to the normal text-to-image models that are learnt in a quantized & compressed latent space. Phenaki introduces a first-stage which spatially & temporally compresses the input videos (e.g. a video of shape 100 x 3 x 256 x 256 -> 20 x 32 x 32). This is achieved by temporal & spatial transformers. An interesting thing to note is that the temporal transformer is autoregressive, which eventually can be used to generate videos with variable length by a shifting context. After learning the first stage which can encode / compress & decode / uncompress videos well, the video-generation model is learned in the latent space. The paper uses [MaskGIT](https://arxiv.org/pdf/2202.04200) for that.

## First Stage Results
We trained a convolutional 3D VQGAN with a spatial compression of f8 and temporal compression of f2. Videos of **(10+1)x128x128** are encoded to a latent size of **(5+1)x16x16**. cViViT proposes to use a separate stem to encode the first frame. In our early experiments we saw that this stem would not receive a lot gradients and thus evolve very slowly, while the rest of the frames looked much better. As a result, we only use a single stem for all frames at once. To still enable image only training in the second stage, we learn an additional frame and prepend it to the start of the sequence, such that when downsampling temporally by 2, the learned and first frame would be encoded into one and the model could learn to ignore the learned embedding and only encode the information from the first frame. We trained the model (43M parameters) for 100k steps, with a batch size of 64 on 8 A100 for 1 day. In the following video the right one is the original and the left one is reconstructed, while in the table top rows represent the original frames and bottom are reconstructed.

https://user-images.githubusercontent.com/61938694/197097822-fa5127d4-281d-4c78-8a79-c9b980959c72.mp4

![108100](https://user-images.githubusercontent.com/61938694/197315310-169e981f-eb43-4f0d-ba9b-a069146f2585.jpg)|![103500](https://user-images.githubusercontent.com/61938694/197315356-56c53f28-1e14-405f-aaa2-61eecd6ac8fc.jpg)
:-------------------------:|:-------------------------:
![109000](https://user-images.githubusercontent.com/61938694/197315281-8bb5918b-b382-47a2-9e8f-52026230f63b.jpg)  |  ![109500](https://user-images.githubusercontent.com/61938694/197315259-aff9015c-13ec-41c1-b5b2-c58c810d3aba.jpg)


<hr>


## Current Progress:
- [x] Implement cViViT
- [x] Implement convolutional baseline for first stage
- [x] Implement Loss (without video perceptual loss)
- [x] Implement Training code
- [x] Download dataset
- [x] Data pipeline
- [ ] Efficient data pipeline
- [x] Test first stage
- [x] Small training first stage
- [X] Full training first stage
- [X] Implement MaskGIT
- [X] Adjust data pipeline for second stage training (include image-only training data)
- [X] Test second stage
- [X] Small training second stage
- [ ] Full training second stage
- [ ] ....

## TODO
- [ ] Activate KMeans in VQGAN training
- [ ] Move dataset to s3

## Open Questions:
- [ ] does the first-stage use a pretrained ViT as proposed in ViViT?
- [ ] how to do positional encoding? [current approach](https://github.com/LAION-AI/phenaki/blob/main/vivq.py#L41)
- [ ] best way to construct dataloader for videos?
- [ ] The dataset 'Moments in Time' does not have captions, and only contains labels. How are captions generated? "A video of {label}"?
