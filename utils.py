import math
import random
import torch
import torchvision


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, video_transform=None, clip_len=10):
        super(RandomDataset).__init__()

        # self.samples = get_samples(root)

        self.clip_len = clip_len
        self.video_transform = video_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # path = random.choice(self.samples)
        path = "./videos/test.mp4"
        vid, _, metadata = torchvision.io.read_video(path)
        vid = vid.permute(0, 3, 1, 2) / 255.
        max_seek = vid.shape[0] - (self.clip_len / vid.shape[0])
        start = math.floor(random.uniform(0., max_seek))
        vid = vid[start:start+self.clip_len]
        if self.video_transform:
            vid = self.video_transform(vid)
        return vid


# video_path = "./videos/test.mp4"
# video, _, _ = torchvision.io.read_video(video_path)
# video = video.permute(0, 3, 2, 1) / 255.
#
transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(128),
        ])
#
# video = transforms(video)

d = RandomDataset(video_transform=transforms)
sample_vid = d[0]
print(sample_vid.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
for i in range(10):
    plt.subplot(4, 4, i + 1)
    plt.imshow(sample_vid[i].permute(1, 2, 0))
    plt.axis("off")

plt.show()
