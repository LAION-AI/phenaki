import math
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from webdataset.handlers import warn_and_continue


class VideoDataset(Dataset):
    def __init__(self, root=None, video_transform=None, clip_len=10, skip_frames=4):
        super(VideoDataset).__init__()

        # self.samples = get_samples(root)

        self.clip_len = clip_len
        self.skip_frames = skip_frames
        self.video_transform = video_transform
        path = "./videos/test.mp4"
        video, _, _ = torchvision.io.read_video(path)
        self.video = video.permute(0, 3, 1, 2) / 255.

    def __len__(self):
        # return len(self.samples)
        return 1000000

    def __getitem__(self, item):
        # path = random.choice(self.samples)
        max_seek = self.video.shape[0] - (self.clip_len * self.skip_frames)
        start = math.floor(random.uniform(0., max_seek))
        video = self.video[start:start+(self.clip_len*self.skip_frames)+1:self.skip_frames]
        if self.video_transform:
            video = self.video_transform(video)
        image, video = video[0], video[1:]
        return image, video


transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(128),
        ])


def collate(batch):
    images = torch.stack([i[0] for i in batch], dim=0)
    videos = torch.stack([i[1] for i in batch], dim=0)
    return [images, videos]


def get_dataloader(args):
    if args.webdataset:
        dataset = wds.WebDataset(args.dataset_path, resampled=True, handler=warn_and_continue).decode(wds.torch_video,
                    handler=warn_and_continue).map(ProcessData(clip_len=args.clip_len, skip_frames=args.skip_frames),
                    handler=warn_and_continue).to_tuple("image", "video", handler=warn_and_continue).shuffle(690, handler=warn_and_continue)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)  # TODO: num_workers=args.num_workers
    else:
        dataset = VideoDataset(video_transform=transforms)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)  # TODO: add num_workers=args.num_workers
    return dataloader


class ProcessData:
    def __init__(self, clip_len=10, skip_frames=4):
        self.video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(128)
        ])
        self.clip_len = clip_len
        self.skip_frames = skip_frames
        print(f"Using clip length of {clip_len} and {skip_frames} skip frames.")

    def __call__(self, data):
        video = data["mp4"][0]
        max_seek = video.shape[0] - (self.clip_len * self.skip_frames)
        if max_seek < 0:
            raise Exception(f"Video too short ({video.shape[0]} frames), skipping.")
        start = math.floor(random.uniform(0., max_seek))
        video = video[start:start+(self.clip_len*self.skip_frames)+1:self.skip_frames]
        video = video.permute(0, 3, 1, 2) / 255.
        if self.video_transform:
            video = self.video_transform(video)
        image, video = video[0], video[1:]
        data["image"] = image
        data["video"] = video
        if video.shape[0] != 10:
            raise Exception("Not 10 frames. But I should find the real cause lol for this happening.")
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

    dataset = wds.WebDataset("./data/6.tar", resampled=True).decode(wds.torch_video,
        ).map(ProcessData()).to_tuple("image", "video",
        ).shuffle(690)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    for sample in dataloader:
        break

    print(sample)
