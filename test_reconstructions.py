import torch
from vivq import VIVQ
import torchvision.utils as vutils
from utils import transforms, VideoDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

device = "cuda"
num_frames = 20

ckpt_path = "./models/vivq_test_one_video_save_model/model.pt"
model = VIVQ().to(device)
model.load_state_dict(torch.load(ckpt_path))
model.eval().requires_grad_(False)

dataset = DataLoader(VideoDataset(video_transform=transforms, clip_len=num_frames), batch_size=1)

image, video = next(iter(dataset))

image, video = image.to(device), video.to(device)

reconstruction, _ = model(image, video)

orig = torch.cat([image.unsqueeze(1), video], dim=1)
orig = orig[0]
recon = reconstruction[0]
comp = vutils.make_grid(torch.cat([orig, recon]), nrow=len(orig)).detach().cpu()
plt.imshow(comp.permute(1, 2, 0))
plt.show()

vutils.save_image(comp, f"results/{num_frames}.jpg")

print(reconstruction)
