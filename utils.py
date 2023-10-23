from typing import Union
import random
import math
import os

from torch.utils.data import Dataset, DataLoader, IterableDataset
from webdataset.handlers import warn_and_continue
from einops import rearrange
import webdataset as wds
import torchvision
import numpy as np
import torch
import cv2

from maskgit import gumbel_sample


SEP = os.path.sep


@torch.no_grad()
def sample_maskgit(model, text_embeddings, steps=20, cond_scale=3., starting_temperature=0.9, base_shape=(6, 16, 16)):
    device = next(model.parameters()).device
    num_tokens = np.prod(base_shape)

    shape = (text_embeddings.shape[0], num_tokens)

    video_token_ids = torch.full(shape, model.mask_id, device=device)
    mask = torch.ones(shape, device=device, dtype=torch.bool)
    timesteps = torch.linspace(0, 1, steps+1)[:-1]

    for step in range(steps):
        is_first_step = step == 0
        is_last_step = step == (steps - 1)

        steps_til_x0 = steps - step

        if not is_first_step:
            num_tokens_mask = int(num_tokens * model.gamma(timesteps[step]))

            _, indices = scores.topk(num_tokens_mask, dim=-1)
            mask = torch.zeros(shape, device=device).scatter(1, indices, 1).bool()

        video_token_ids = torch.where(mask, model.mask_id, video_token_ids)

        logits = model(video_token_ids, text_embeddings)

        temperature = starting_temperature * (step / steps_til_x0)
        pred_video_ids = gumbel_sample(logits, temperature=temperature)

        video_token_ids = torch.where(mask, pred_video_ids, video_token_ids)

        if not is_last_step:
            scores = logits.gather(2, rearrange(pred_video_ids, '... -> ... 1'))
            scores = 1 - rearrange(scores, '... 1 -> ...')
            scores = torch.where(mask, scores, -1e4)
    return video_token_ids.view(-1, *base_shape)


def sample_paella(model, c, x=None, mask=None, T=12, size=(6, 16, 16), starting_t=0, temp_range=[1.0, 1.0], typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1, renoise_steps=11, renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        elif mask is not None:
            noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            x = noise * mask + (1-mask) * x
        init_x = x.clone()
        for i in range(starting_t, T):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]
            logits = model(x, c, r)
            if classifier_free_scale >= 0:
                logits_uncond = model(x, torch.zeros_like(c), r)
                logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, x.size(1))
            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            # x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            x_flat = gumbel_sample(x_flat, temperature=temp)
            x = x_flat.view(x.size(0), *x.shape[2:])
            if mask is not None:
                x = x * mask + (1-mask) * init_x
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                else:  # 'rand'
                    x, _ = model.add_noise(x, r_range[i+1])
    return x.detach()


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


def collate_first_stage(batch):
    images = torch.stack([i[0] for i in batch], dim=0)
    videos = torch.stack([i[1] for i in batch], dim=0)
    return [images, videos]


def collate_second_stage(batch):

    if len(batch[0]) == 2:
        images = torch.stack([i[0] for i in batch], dim=0)
        videos = None
        captions = [i[1] for i in batch]
    else:
        images = torch.stack([i[0] for i in batch], dim=0)
        videos = torch.stack([i[1] for i in batch], dim=0)
        captions = [i[2] for i in batch]
    return [images, videos, captions]


def get_dataloader(args):
    if args.dataset == "first_stage":
        dataset = wds.WebDataset(args.dataset_path, resampled=True, handler=warn_and_continue).decode(wds.torch_video,
                    handler=warn_and_continue).map(ProcessVideos(clip_len=args.clip_len, skip_frames=args.skip_frames),
                    handler=warn_and_continue).to_tuple("image", "video", handler=warn_and_continue).shuffle(690, handler=warn_and_continue)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_first_stage)  # TODO: num_workers=args.num_workers

    elif args.dataset == "second_stage":
        dataset = MixImageVideoDataset(args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_second_stage, num_workers=args.num_workers)  # TODO: num_workers=args.num_workers

    else:
        dataset = VideoDataset(video_transform=transforms)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)  # TODO: add num_workers=args.num_workers
    return dataloader


class ProcessImages:
    def __init__(self,):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(128),
            torchvision.transforms.RandomCrop(128),
        ])

    def __call__(self, data):
        data["jpg"] = self.transforms(data["jpg"])
        return data


class ProcessVideos:
    def __init__(self, clip_len=10, skip_frames=4):
        self.video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.RandomCrop(128)
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


class MixImageVideoDataset(IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size  # TODO: split this into image bs and video bs
        self.video_dataset, self.image_dataset = self.init_dataloaders(args)

    def init_dataloaders(self, args):
        video_dataset = wds.WebDataset(args.urls["videos"], resampled=True, handler=warn_and_continue).decode(wds.torch_video,
                    handler=warn_and_continue).map(ProcessVideos(clip_len=args.clip_len, skip_frames=args.skip_frames),
                    handler=warn_and_continue).to_tuple("image", "video", "txt", handler=warn_and_continue).shuffle(690, handler=warn_and_continue)
        image_dataset = wds.WebDataset(args.urls["images"], resampled=True, handler=warn_and_continue).decode("rgb").map(
            ProcessImages(), handler=warn_and_continue).to_tuple("jpg", "txt", handler=warn_and_continue).shuffle(6969, initial=10000)
        return video_dataset, image_dataset

    def __iter__(self):
        # sources = [iter(self.image_dataset), iter(self.video_dataset)]
        sources = [iter(self.video_dataset), iter(self.image_dataset)]
        # sources = [iter(self.video_dataset)]
        while True:
            for source in sources:
                for _ in range(self.batch_size):
                    try:
                        yield next(source)
                    except Exception as e:
                        print(f'\n\nerror:{e}\n\n')
                        return

# video_path = "./videos/test.mp4"
# video, _, _ = torchvision.io.read_video(video_path)
# video = video.permute(0, 3, 2, 1) / 255.
# video = transforms(video)


## MaskGIT, Paella 모델 학습시 결과물 저장을 위한 함수 (영상 저장용.)
def video_write(videos: Union[torch.tensor, list], save_path: str):

    ## save_path 인자값을 경로 구분자로 분할 후 폴더 생성.
    ## e.g.) /home/workspace/dove/sparrow -> /home/workspace/dove
    #! 경로 구분자로 분할한 리스트의 마지막 요소는 파일이름으로 사용할 예정이라 폴더 생성할 때에는 제외함.
    path = SEP.join(save_path.split(SEP)[:-1])
    os.makedirs(path, exist_ok = True)
    
    ## 저장할 영상이 여러개 있는 경우 반복문 돌며 저장.
    for idx, video in enumerate(videos):
        
        ## 영상 데이터 전처리 해주는 부분.
        video  = video.cpu().mul(255).add_(0.5).clamp_(0, 255).numpy()
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        ## mp4 포맷에 fps 30, 128 x 128 사이즈로 저장.
        output = cv2.VideoWriter(f'{save_path}_{idx}.mp4', fourcc, 30.0, (128, 128))

        for frame in video:
            
            ## 영상 데이터가 가지고 있는 프레임(이미지) 전처리하는 부분.
            frame = np.transpose(frame, (1, 2, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ## 전처리 된 이미지를 영상파일에 추가하는 부분.
            output.write(frame.astype(np.uint8))

        output.release()


## MaskGIT, Paella 모델 학습시 결과물 저장을 위한 함수 (이미지 저장용.)
def image_write(images: Union[torch.tensor, list], save_path: str):
    ## save_path 인자값을 경로 구분자로 분할 후 폴더 생성.
    ## e.g.) /home/workspace/dove/sparrow -> /home/workspace/dove
    #! 경로 구분자로 분할한 리스트의 마지막 요소는 파일이름으로 사용할 예정이라 폴더 생성할 때에는 제외함.
    path = SEP.join(save_path.split(SEP)[:-1])
    os.makedirs(path, exist_ok = True)
    
    ## 저장할 이미지 데이터가 여러개인 경우 반복문을 돌며 저장
    for idx, image in enumerate(images):

        ## 이미지 전처리 하는 부분.
        image = image.cpu().mul(255).add_(0.5).clamp_(0, 255).squeeze(0).numpy()
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## 이미지 저장하는 부분.
        cv2.imwrite(f'{save_path}_{idx}.jpg', image)


def calculate_psnr(img1: np.array, img2: np.array) -> float:
    # img1 and img2 have range [0, 255]

    img1 *= 255
    img2 *= 255
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1:np.array, img2:np.array) -> float:
    img1 *= 255
    img2 *= 255
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1 : np.array, img2 : np.array) -> Union[float, None]:
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
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

    # dataset = wds.WebDataset("./data/6.tar", resampled=True).decode(wds.torch_video,
    #     ).map(ProcessVideos()).to_tuple("image", "video",
    #     ).shuffle(690)
    # dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_first_stage)

    args.batch_size = 1
    args.clip_len = 10
    args.skip_frames = 3
    # args.urls = {"videos": "file:./data/6.tar"}
    args.urls = {
        "videos": "file:C:/Users/d6582/Documents/ml/phenaki/data/webvid/tar_files/0.tar",
        "images": "file:C:/Users/d6582/Documents/ml/paella/paella_unet/000069.tar"
    }
    dataset = MixImageVideoDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_second_stage)

    for sample in dataloader:
        break

    print(sample)