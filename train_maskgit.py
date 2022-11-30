import os
import time

import numpy as np
import open_clip
import torch
import torch.multiprocessing as mp
import wandb
from open_clip import tokenizer
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from distributed import init_distributed_device, is_primary, is_local_primary
from maskgit import MaskGit
from paella import DenoiseUNet
from utils import get_dataloader, sample_paella
from vivq import VIVQ, BASE_SHAPE
# from transformers import T5Tokenizer, T5Model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, default=None)
parser.add_argument('--model', type=str, default='paella')
parser.add_argument('--dataset', type=str, default='second_stage')

parser.add_argument('--total-steps', type=int, default=300_000)
parser.add_argument('-b', '--batch-size', type=int, default=4)
parser.add_argument('-j', '--num-workers', type=int, default=6)
parser.add_argument('--log-period', type=int, default=2000)
parser.add_argument('--extra-ckpt', type=int, default=10_000)
parser.add_argument('--accum-grad', type=int, default=2)

parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--no-set-device-rank", default=False, action="store_true",
                    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).")

# args.vq_path = "./models/server/vivq_8192_5_skipframes/model_100000.pt"
parser.add_argument('--vq-path', type=str, default='/fsx/phenaki/src/models/model_120000.pt')

parser.add_argument('--dim', type=int, default=1224)
parser.add_argument('--num-tokens', type=int, default=8192)
parser.add_argument('--max-seq-len', type=int, default=6 * 16 * 16)
parser.add_argument('--depth', type=int, default=22)
parser.add_argument('--dim-context', type=int, default=1024)
parser.add_argument('--heads', type=int, default=22)

parser.add_argument('--clip-len', type=int, default=10)
parser.add_argument('--skip-frames', type=int, default=10)


def main():
    args = parser.parse_args()

    # FIXME turn into arg(s)
    args.urls = {
        # "videos": "file:C:/Users/d6582/Documents/ml/phenaki/data/webvid/tar_files/0.tar",
        # "videos": "/fsx/mas/phenaki/data/webvid/tar_files/{0..249}.tar",
        "videos": "/fsx/phenaki/data/videos/tar_files/{0..1243}.tar",
        # "images": "file:C:/Users/d6582/Documents/ml/paella/evaluations/laion-30k/000069.tar"
        # "images": "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
        "images": "/fsx/phenaki/coyo-700m/coyo-data-2/{00000..20892}.tar"
    }

    # FIXME handle autogen run name if empty
    assert args.run_name is not None

    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True
    else:
        resume = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = init_distributed_device(args)

    vqmodel = VIVQ(codebook_size=args.num_tokens).to(device)
    vqmodel.load_state_dict(torch.load(args.vq_path, map_location=device))
    vqmodel.vqmodule.q_step_counter += int(1e9)
    vqmodel.eval().requires_grad_(False)

    # t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")  # change with "t5-b3" for the 10GB model LoL
    # t5_model = T5Model.from_pretrained("t5-small").to(device).requires_grad_(False)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir="/fsx/mas/.cache")
    del clip_model.visual
    clip_model = clip_model.to(device).eval().requires_grad_(False)

    if args.model == "maskgit":
        model = MaskGit(
            dim=args.dim,
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            depth=args.depth,
            dim_context=args.dim_context,
            heads=args.heads).to(device)
    elif args.model == "paella":
        model = DenoiseUNet(
            num_labels=args.num_tokens,
            down_levels=[4, 6, 8],
            up_levels=[8, 6, 4],
            c_clip=args.dim_context).to(device)
    else:
        raise NotImplementedError()

    if is_primary(args):
        print(f"Starting run '{args.run_name}'....")
        print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    lr = 3e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    grad_scaler = torch.cuda.amp.GradScaler()
    grad_norm = torch.tensor(0., device=device)

    grad_accum_steps = args.accum_grad
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=args.total_steps, pct_start=0.1, div_factor=25, anneal_strategy='cos')
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=args.total_steps,
        pct_start=0.1, div_factor=25, final_div_factor=1 / 25, anneal_strategy='linear')

    if resume:
        if is_primary(args):
            print("Loading last checkpoint....")
        logs = torch.load(f"results/{args.run_name}/log.pt")
        run_id = logs["wandb_run_id"]
        start_step = logs["step"] + 1
        losses = logs["losses"]
        accuracies = logs["accuracies"]
        total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        model.load_state_dict(torch.load(f"models/{args.run_name}/model.pt", map_location=device))
        if is_primary:
            print("Loaded model....")
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            while last_lr > optimizer.param_groups[0]["lr"]:
                scheduler.step()
        if is_primary:
            print(f"Initialized scheduler")
            print(f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} "
                  f"-> {last_lr == optimizer.param_groups[0]['lr']}")
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        run_id = wandb.util.generate_id()
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0

    if is_primary:
        wandb.init(project="DenoiseGIT", name=args.run_name, entity="wand-tech", config=vars(args), id=run_id,
                   resume="allow")
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)
        wandb.watch(model)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    model.train()
    # images, videos = next(iter(dataset))
    if is_primary(args):
        pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step)
    else:
        pbar = enumerate(dataset, start=start_step)
    # pbar = tqdm(range(1000000))
    for step, (images, videos, captions) in pbar:
    # for step in pbar:
    #     images = torch.randn(1, 3, 128, 128)
        # videos = None
        # videos = torch.randn(1, 10, 3, 128, 128)
        images = images.to(device)
        if videos is not None:
            videos = videos.to(device)

        with torch.no_grad():
            video_indices = vqmodel.encode(images, videos)[2]  # TODO: make this cleaner
            r = torch.rand(images.size(0), device=device)
            noised_indices, mask = model.module.add_noise(video_indices, r)  # add module back

            if np.random.rand() < 0.1:  # 10% of the times...
                # text_embeddings = images.new_zeros(images.size(0), 1, args.dim_context)
                text_embeddings = images.new_zeros(images.size(0), args.dim_context)
            else:
                # text_tokens = t5_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids
                # text_tokens = text_tokens.to(device)
                # text_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state

                text_tokens = tokenizer.tokenize(captions)
                text_tokens = text_tokens.to(device)
                text_embeddings = clip_model.encode_text(text_tokens).float()

                # text_embeddings = torch.randn(1, 768).to(device)

        with torch.cuda.amp.autocast():
            pred = model(noised_indices, text_embeddings, r)
            loss, acc = model.module.loss(pred, video_indices)
            loss_adjusted = loss / grad_accum_steps

        total_loss += loss.item()
        total_acc += acc.item()

        grad_scaler.scale(loss_adjusted).backward()
        # loss_adjusted.backward()

        if (step + 1) % grad_accum_steps == 0:
            # optimizer.step()
            grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if is_primary(args):
            log = {
                'loss': total_loss / (step + 1),
                'curr_loss': loss.item(),
                'acc': total_acc / (step + 1),
                'curr_acc': acc.item(),
                'ppx': np.exp(loss.item()),
                'lr': optimizer.param_groups[0]['lr'],
                'gn': grad_norm
            }
            pbar.set_postfix(log)
            wandb.log(log)

        if is_primary(args) and step % args.log_period == 0:
            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                n = 1
                captions = captions[:6]
                text_embeddings = text_embeddings[:6]
                sampled = sample_paella(model.module, text_embeddings)
                sampled = vqmodel.decode_indices(sampled)

                cool_captions_data = torch.load("cool_captions.pth")
                cool_captions_text = cool_captions_data["captions"]

                # text_tokens = t5_tokenizer(cool_captions_text, return_tensors="pt", padding=True, truncation=True).input_ids
                # text_tokens = text_tokens.to(device)
                # cool_captions_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state

                text_tokens = tokenizer.tokenize(cool_captions_text)
                text_tokens = text_tokens.to(device)
                cool_captions_embeddings = clip_model.encode_text(text_tokens).float()

                cool_captions = DataLoader(
                    TensorDataset(cool_captions_embeddings.repeat_interleave(n, dim=0)),
                    batch_size=1)
                cool_captions_sampled = []
                st = time.time()
                for caption_embedding in cool_captions:
                    caption_embedding = caption_embedding[0].float().to(device)
                    sampled_text = sample_paella(model.module, caption_embedding)
                    sampled_text = vqmodel.decode_indices(sampled_text)
                    for s in sampled_text:
                        cool_captions_sampled.append(s.cpu())
                print(f"Took {time.time() - st} seconds to sample {len(cool_captions_text)} captions.")

            model.train()

            images = images[:6]
            video_indices = video_indices[:6]
            if videos is not None:
                videos = videos[:6]
                videos = torch.cat([images.unsqueeze(1), videos], dim=1)
                recon_video = vqmodel.decode_indices(video_indices, BASE_SHAPE)

                log_data = [
                    [captions[i]] +
                    [wandb.Video(sampled[i].cpu().mul(255).add_(0.5).clamp_(0, 255).numpy())] +
                    [wandb.Video(videos[i].cpu().mul(255).add_(0.5).clamp_(0, 255).numpy())] +
                    [wandb.Video(recon_video[i].cpu().mul(255).add_(0.5).clamp_(0, 255).numpy())]
                    for i in range(len(captions))
                ]
            else:
                videos = images.unsqueeze(1)
                recon_video = vqmodel.decode_indices(video_indices, (1, 16, 16))
                log_data = [
                    [captions[i]] +
                    [wandb.Video(sampled[i].cpu().mul(255).add_(0.5).clamp_(0, 255).numpy())] +
                    [wandb.Image(videos[i])] +
                    [wandb.Image(recon_video[i])]
                    for i in range(len(captions))
                ]

            log_table = wandb.Table(data=log_data, columns=["Caption", "Video", "Orig", "Recon"])
            wandb.log({"Log": log_table})

            log_data_cool = [
                [cool_captions_text[i]] +
                [wandb.Video(cool_captions_sampled[i].cpu().mul(255).add_(0.5).clamp_(0, 255).numpy())]
                for i in range(len(cool_captions_text))]
            log_table_cool = wandb.Table(data=log_data_cool, columns=["Caption", "Video"])
            wandb.log({"Log Cool": log_table_cool})

            del videos, video_indices, images, r, text_embeddings, sampled, log_data, sampled_text, log_data_cool
            del noised_indices, mask, pred, loss, loss_adjusted, acc

            if step % args.extra_ckpt == 0:
                torch.save(model.module.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
                torch.save(grad_scaler.state_dict(), f"models/{args.run_name}/model_{step}_scaler.pt")
            torch.save(model.module.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save(grad_scaler.state_dict(), f"models/{args.run_name}/scaler.pt")
            torch.save({
                'step': step, 'losses': losses, 'accuracies': accuracies,
                'wandb_run_id': run_id}, f"results/{args.run_name}/log.pt")


if __name__ == '__main__':
    main()

    # device = "cuda"
    # model = MaskGit(dim=args.dim, num_tokens=args.num_tokens, max_seq_len=args.max_seq_len, depth=args.depth,
    #                 dim_context=args.dim_context).to(device)
    # t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")  # change with "t5-b3" for the 10GB model LoL
    # t5_model = T5Model.from_pretrained("t5-small").to(device).requires_grad_(False)
    # caption = "the weather is so beautiful today"
    # text_tokens = t5_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).input_ids
    # text_tokens = text_tokens.to(device)
    # text_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state
    # print(text_embeddings.shape)
    # text_embeddings = torch.randn(1, 10, 512).to(device)
    #
    # print(sample(model, text_embeddings).shape)
