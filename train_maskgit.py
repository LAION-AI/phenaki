import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from cvivit import VIVIT
from vivq import VIVQ
from maskgit import MaskGit, gumbel_sample
from utils import get_dataloader
from transformers import T5Tokenizer, T5Model
from einops import rearrange


@torch.no_grad()
def sample(model, text_embeddings, steps=20, cond_scale=3., starting_temperature=0.9, base_shape=(6, 16, 16)):
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


def train(proc_id, args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True  # TODO: change back
    else:
        resume = False
    if not proc_id and args.node_id == 0:
        if resume:
            wandb.init(project="Phenaki", name=args.run_name, entity="wand-tech", config=vars(args))
        else:
            wandb.init(project="Phenaki", name=args.run_name, entity="wand-tech", config=vars(args))
        print(f"Starting run '{args.run_name}'....")
        print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")
    parallel = len(args.devices) > 1
    device = torch.device(proc_id)

    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="file:///fsx/mas/phenaki/dist_file",
                                world_size=args.n_nodes * len(args.devices),
                                rank=proc_id + len(args.devices) * args.node_id)
        torch.set_num_threads(6)

    vqmodel = VIVQ().to(device)
    vqmodel.load_state_dict(torch.load(args.vq_path, map_location=device))
    vqmodel.vqmodule.q_step_counter += int(1e9)
    vqmodel.eval().requires_grad_(False)

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")  # change with "t5-b3" for the 10GB model LoL
    t5_model = T5Model.from_pretrained("t5-small").to(device).requires_grad_(False)

    if args.model == "maskgit":
        model = MaskGit(dim=args.dim, num_tokens=args.num_tokens, max_seq_len=args.max_seq_len, depth=args.depth, dim_context=args.dim_context).to(device)
    elif args.model == "":
        model = None.to(device)
    else:
        raise NotImplementedError()

    if not proc_id and args.node_id == 0:
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    lr = 3e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    if not proc_id and args.node_id == 0:
        # wandb.watch(model)
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)

    grad_accum_steps = args.accum_grad
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                              steps_per_epoch=math.ceil(1000 / grad_accum_steps),
                                              epochs=300, pct_start=30 / 300, div_factor=25,
                                              final_div_factor=1 / 25, anneal_strategy='linear')

    if resume:
        if not proc_id and args.node_id == 0:
            print("Loading last checkpoint....")
        logs = torch.load(f"results/{args.run_name}/log.pt")
        start_step = logs["step"] + 1
        losses = logs["losses"]
        accuracies = logs["accuracies"]
        total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        model.load_state_dict(torch.load(f"models/{args.run_name}/model.pt", map_location=device))
        if not proc_id and args.node_id == 0:
            print("Loaded model....")
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            while last_lr > optimizer.param_groups[0]["lr"]:
                scheduler.step()
        if not proc_id and args.node_id == 0:
            print(f"Initialized scheduler")
            print(f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}")
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0

    model.train()
    # images = torch.randn(1, 3, 128, 128)
    # videos = torch.randn(1, 10, 3, 128, 128)
    # images, videos = next(iter(dataset))
    pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step) if args.node_id == 0 and proc_id == 0 else enumerate(dataset, start=start_step)
    # pbar = tqdm(range(1000000))
    for step, (images, videos, captions) in pbar:
    # for step in pbar:
        images = images.to(device)
        videos = videos.to(device)

        with torch.no_grad():
            video_indices = vqmodel.encode(images, videos)[2]  # TODO: make this cleaner
            video_shape = video_indices.shape
            video_indices = video_indices.view(videos.shape[0], -1)
            r = torch.rand(images.size(0), device=device)
            noised_indices, mask = model.module.add_noise(video_indices, r)

            if np.random.rand() < 0.1:  # 10% of the times...
                text_embeddings = images.new_zeros(images.size(0), 1, args.dim_context)
            else:
                text_tokens = t5_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids
                text_tokens = text_tokens.to(device)
                text_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state

        pred = model(noised_indices, text_embeddings)
        loss = criterion(pred[mask], video_indices[mask])  # check if this works and not the other way around
        loss_adjusted = loss / grad_accum_steps

        loss_adjusted.backward()
        if (step + 1) % grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        acc = (pred.argmax(1) == video_indices).float()
        acc = acc.mean()

        if not proc_id and args.node_id == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': acc.item(),
                'ppx': np.exp(loss.item() / (step + 1)),
                'lr': optimizer.param_groups[0]['lr'],
                'gn': grad_norm
            })
            # wandb.log({
            #     "loss": loss,
            #     "d_loss": d_loss,
            #     "lr": optimizer.param_groups[0]['lr'],
            # })

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:
            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                n = 1
                images = images[:6]
                video_indices = video_indices[:6]
                captions = captions[:6]
                text_embeddings = text_embeddings[:6]
                sampled = sample(model.module, c=text_embeddings)[-1]
                sampled = vqmodel.decode(sampled, video_shape)
                recon_video = vqmodel.decode(video_indices, video_shape)

                if args.log_captions:
                    cool_captions_data = torch.load("cool_captions.pth")
                    cool_captions_text = cool_captions_data["captions"]

                    text_tokens = t5_tokenizer(cool_captions_text, return_tensors="pt", padding=True, truncation=True).input_ids
                    text_tokens = text_tokens.to(device)
                    cool_captions_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state

                    cool_captions = DataLoader(TensorDataset(cool_captions_embeddings.repeat_interleave(n, dim=0)), batch_size=6)
                    cool_captions_sampled = []
                    st = time.time()
                    for caption_embedding in cool_captions:
                        caption_embedding = caption_embedding[0].float().to(device)
                        sampled_text = sample(model.module, c=caption_embedding)[-1]
                        sampled_text = vqmodel.decode(sampled_text, video_shape)
                        for s, in sampled_text:
                            cool_captions_sampled.append(s.cpu())
                    print(f"Took {time.time() - st} seconds to sample {len(cool_captions_text)} captions.")

                    cool_captions_sampled = torch.stack(cool_captions_sampled)
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(cool_captions_sampled, nrow=11),
                        os.path.join(f"results/{args.run_name}", f"cool_captions_{step:03d}.png")
                    )

                log_images = torch.cat([
                    torch.cat([i for i in sampled.cpu()], dim=-1),
                ], dim=-2)

            model.train()

            torchvision.utils.save_image(log_images, os.path.join(f"results/{args.run_name}", f"{step:03d}.png"))

            log_data = [
                [captions[i]] + [wandb.Video(sampled[i])] + [wandb.Video(images[i])] + [
                    wandb.Video(recon_video[i])] for i in range(len(captions))]
            log_table = wandb.Table(data=log_data, columns=["Caption", "Video", "Orig", "Recon"])
            wandb.log({"Log": log_table})

            if args.log_captions:
                log_data_cool = [[cool_captions_text[i]] + [wandb.Video(cool_captions_sampled[i])] for i in range(len(cool_captions_text))]
                log_table_cool = wandb.Table(data=log_data_cool, columns=["Caption", "Video"])
                wandb.log({"Log Cool": log_table_cool})
                del sampled_text, log_data_cool

            del sampled, log_data

            if step % args.extra_ckpt == 0:
                torch.save(model.module.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
            torch.save(model.module.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save({'step': step, 'losses': losses, 'accuracies': accuracies}, f"results/{args.run_name}/log.pt")

    del images, videos, video_indices, r, text_embeddings
    del noised_indices, mask, pred, loss, loss_adjusted, acc


def launch(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in args.devices])
    if len(args.devices) == 1:
        train(0, args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "33751"
        p = mp.spawn(train, nprocs=len(args.devices), args=(args,))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "maskgit_1"
    args.model = "maskgit"
    args.webdataset = True
    # args.dataset_path = "file:./data/6.tar"
    args.dataset_path = "/fsx/mas/phenaki/data/raw_data/Moments_in_Time_Raw/tar_files/{0..363}.tar"
    args.total_steps = 5_000_000
    args.batch_size = 10
    args.num_workers = 10
    args.log_period = 100
    args.extra_ckpt = 10_000
    args.accum_grad = 1

    args.vq_path = ""
    args.dim = 128
    args.num_tokens = 1024
    args.max_seq_len = 6 * 16 * 16
    args.depth = 6
    args.dim_context = 512

    args.clip_len = 10
    args.skip_frames = 3

    # args.n_nodes = 1
    # args.node_id = int(os.environ["SLURM_PROCID"])
    # # args.node_id = 0
    # args.devices = [0, 1, 2, 3, 4, 5, 6, 7]
    # args.devices = [0]
    #
    # print("Launching with args: ", args)
    # launch(
    #     args
    # )

    device = "cuda"
    model = MaskGit(dim=args.dim, num_tokens=args.num_tokens, max_seq_len=args.max_seq_len, depth=args.depth,
                    dim_context=args.dim_context).to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")  # change with "t5-b3" for the 10GB model LoL
    t5_model = T5Model.from_pretrained("t5-small").to(device).requires_grad_(False)
    caption = "the weather is so beautiful"
    # text_tokens = t5_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).input_ids
    # text_tokens = text_tokens.to(device)
    # text_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state
    text_embeddings = torch.randn(1, 10, 512).to(device)

    print(sample(model, text_embeddings).shape)
