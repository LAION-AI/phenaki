import random
import torch
import itertools
import torchvision


class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(RandomDataset).__init__()

        self.samples = get_samples(root)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            yield video


def example_read_video(video_object, start=0, end=None):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )
    video_frames = torch.empty(0)
    video_pts = []
    video_object.set_current_stream("video")
    frames = []
    for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
        frames.append(frame['data'])
        video_pts.append(frame['pts'])
    if len(frames) > 0:
        video_frames = torch.stack(frames, 0)

    return video_frames, video_pts, video_object.get_metadata()

video_path = "./WUzgd7C1pWA.mp4"
stream = "video"
video = torchvision.io.VideoReader(video_path, stream)
video.get_metadata()


# Total number of frames should be 327 for video and 523264 datapoints for audio
vf, af, info, meta = example_read_video(video)
print(vf.size(), af.size())