import math
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from webdataset.handlers import warn_and_continue


class VideoDataset(Dataset):
    def __init__(self, root=None, video_transform=None, clip_len=10):
        super(VideoDataset).__init__()

        # self.samples = get_samples(root)

        self.clip_len = clip_len
        self.video_transform = video_transform
        path = "./videos/test.mp4"
        video, _, _ = torchvision.io.read_video(path)
        self.video = video.permute(0, 3, 1, 2) / 255.

    def __len__(self):
        # return len(self.samples)
        return 1000000

    def __getitem__(self, item):
        # path = random.choice(self.samples)
        max_seek = self.video.shape[0] - self.clip_len
        start = math.floor(random.uniform(0., max_seek))
        video = self.video[start:start+self.clip_len+1]
        if self.video_transform:
            video = self.video_transform(video)
        image, video = video[0], video[1:]
        return image, video


transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(128),
        ])


def get_dataloader(args):
    if args.webdataset:
        dataset = wds.WebDataset(args.dataset_path, resampled=True, handler=warn_and_continue).decode(wds.torch_video,
                    handler=warn_and_continue).map(ProcessData(), handler=warn_and_continue).to_tuple("image", "video",
                    handler=warn_and_continue).shuffle(690, handler=warn_and_continue)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)  # TODO: num_workers=args.num_workers
    else:
        dataset = VideoDataset(video_transform=transforms)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)  # TODO: add num_workers=args.num_workers
    return dataloader


class ProcessData:
    def __init__(self, clip_len=10):
        self.video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(128),
        ])
        self.clip_len = clip_len

    def __call__(self, data):
        video = data["mp4"][0].permute(0, 3, 1, 2) / 255.
        max_seek = video.shape[0] - self.clip_len
        start = math.floor(random.uniform(0., max_seek))
        video = video[start:start+self.clip_len+1]
        if self.video_transform:
            video = self.video_transform(video)
        image, video = video[0], video[1:]
        data["video"] = video
        data["image"] = image
        return data



# video_path = "./videos/test.mp4"
# video, _, _ = torchvision.io.read_video(video_path)
# video = video.permute(0, 3, 2, 1) / 255.
# video = transforms(video)
if __name__ == '__main__':
    # d = VideoDataset(video_transform=transforms)
    # sample_vid = d[0]
    # print(sample_vid.shape)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(12, 12))
    # for i in range(10):
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(sample_vid[i].permute(1, 2, 0))
    #     plt.axis("off")
    #
    # plt.show()

    dataset = wds.WebDataset("file:./data/6.tar", resampled=True, handler=warn_and_continue).decode(wds.torch_video,
                handler=warn_and_continue).map(ProcessData(), handler=warn_and_continue).to_tuple("video", "image",
                handler=warn_and_continue).shuffle(690, handler=warn_and_continue)
    dataloader = DataLoader(dataset, batch_size=2)


    for sample in dataloader:
        print(sample)

    print(sample)
