import time
import os

from distributed import init_distributed_device, is_primary
from torch.nn.parallel import DistributedDataParallel
# from transformers import T5Tokenizer, T5Model
from open_clip import tokenizer
from torch import optim
from tqdm import tqdm
import numpy as np
import open_clip
import torch

from utils import get_dataloader, sample_paella, sample_maskgit
from utils import image_write, video_write
from vivq import VIVQ, BASE_SHAPE
from paella import DenoiseUNet
from maskgit import MaskGit


def train(args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True
    else:
        resume = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = init_distributed_device(args)

    vqmodel = VIVQ(codebook_size=args.num_tokens, model = 'maskgit').to(device)
    vqmodel.load_state_dict(torch.load(args.vq_path, map_location=device))
    vqmodel.vqmodule.q_step_counter += int(1e9)
    vqmodel.eval().requires_grad_(False)

    # t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")  # change with "t5-b3" for the 10GB model LoL
    # t5_model = T5Model.from_pretrained("t5-small").to(device).requires_grad_(False)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir="/fsx/mas/.cache")
    del clip_model.visual
    clip_model = clip_model.to(device).eval().requires_grad_(False)

    if args.model == "maskgit":
        model = MaskGit(dim=args.dim, num_tokens=args.num_tokens, max_seq_len=args.max_seq_len, depth=args.depth, dim_context=args.dim_context, heads=args.heads).to(device)
    elif args.model == "paella":
        model = DenoiseUNet(num_labels=args.num_tokens, down_levels=[4, 6, 8], up_levels=[8, 6, 4], c_clip=args.dim_context).to(device)
    else:
        raise NotImplementedError()

    if is_primary(args):
        print(f"Starting run '{args.run_name}'....")
        print(f"Batch Size check: {args.world_size * args.batch_size * args.accum_grad}")
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    lr = 5e-3
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    grad_scaler = torch.cuda.amp.GradScaler()
    grad_norm = torch.tensor(0., device=device)

    grad_accum_steps = args.accum_grad
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=args.total_steps, pct_start=0.1, div_factor=25, anneal_strategy='cos')
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=args.total_steps, pct_start=0.1, div_factor=25, final_div_factor=1 / 25, anneal_strategy='linear')

    if resume:
        logs = torch.load(f"results/{args.run_name}/log.pt")
        start_step = logs["step"] + 1
        losses, curr_losses         = logs["losses"]    , logs['curr_losses']
        accuracies, curr_accuracies = logs["accuracies"], logs['curr_accuracies']
        total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        model.load_state_dict(torch.load(f"models/{args.run_name}/model.pt", map_location=device))
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            while last_lr > optimizer.param_groups[0]["lr"]:
                scheduler.step()
        if is_primary(args):
            print("Loaded last checkpoint....")
            print(f"Initialized scheduler")
            print(f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}")
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        # run_id = wandb.util.generate_id()
        losses    , curr_losses     = [], []
        accuracies, curr_accuracies = [], []
        start_step, total_loss, total_acc = 0, 0, 0

    if is_primary(args):
        # wandb.init(project="DenoiseGIT", name=args.run_name, entity="wand-tech", config=vars(args), id=run_id, resume="allow")
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)
        # wandb.watch(model)

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

        # videos = None
        # videos = torch.randn(1, 10, 3, 128, 128)
        images = images.to(device)
        if videos is not None:
            videos = videos.to(device)

        with torch.no_grad():
            
            video_indices = vqmodel.encode(images, videos)[2]  # TODO: make this cleaner
            r = torch.rand(images.size(0), device=device)

            noised_indices, mask = model.add_noise(video_indices, r)  # add module back

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

            loss, acc = model.loss(pred, video_indices)
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
            # wandb.log(log)

        if is_primary(args) and step % args.log_period == 0:
            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            curr_losses.append(loss.item())
            curr_accuracies.append(acc.item())

            model.eval()
            with torch.no_grad():
                captions = captions[:6]
                text_embeddings = text_embeddings[:6]
                sampled = sample_paella(model, text_embeddings)
                sampled = vqmodel.decode_indices(sampled)

                # text_tokens = t5_tokenizer(cool_captions_text, return_tensors="pt", padding=True, truncation=True).input_ids
                # text_tokens = text_tokens.to(device)
                # cool_captions_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state

                cool_captions_text = open("cool_captions.txt").read().splitlines()

                text_tokens = tokenizer.tokenize(cool_captions_text).to(device)
                cool_captions_embeddings = clip_model.encode_text(text_tokens).float()

                cool_captions_sampled = []
                st = time.time()
                for caption_embedding in cool_captions_embeddings.chunk(10):
                    caption_embedding = caption_embedding[0].float().to(device)
                    caption_embedding = caption_embedding.unsqueeze(0)
                    
                    sampled_text = sample_paella(model, caption_embedding)
                    sampled_text = vqmodel.decode_indices(sampled_text)
                    for s in sampled_text:
                        cool_captions_sampled.append(s.cpu())
                print(f"Took {time.time() - st} seconds to sample {len(cool_captions_text)} captions.")

            model.train()

            images = images[:6]
            video_indices = video_indices[:6]

            step_ = str(step).zfill(7)
            if videos is not None:
                videos = videos[:6]
                videos = torch.cat([images.unsqueeze(1), videos], dim=1)
                recon_video = vqmodel.decode_indices(video_indices, BASE_SHAPE)

                video_write(sampled    ,  f'results/{args.run_name}/sampled/sampled_{step_}')
                video_write(videos     ,   f'results/{args.run_name}/videos/videos_{step_}')
                video_write(recon_video,    f'results/{args.run_name}/recon/recon_{step_}')
                    
            else:
                videos = images.unsqueeze(1)
                recon_video = vqmodel.decode_indices(video_indices, (1, 16, 16))

                video_write(sampled    , f'results/{args.run_name}/sampled_{step_}')
                image_write(videos     ,  f'results/{args.run_name}/videos_{step_}')
                image_write(recon_video,   f'results/{args.run_name}/recon_{step_}')

            video_write(cool_captions_sampled, f'results/{args.run_name}/cool_sampled/{step_}')

            del videos, video_indices, images, r, text_embeddings, sampled, sampled_text
            del noised_indices, mask, pred, loss, loss_adjusted, acc

            if step % args.extra_ckpt == 0:
                torch.save(model.state_dict(), f"models/{args.run_name}/model.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
                torch.save(grad_scaler.state_dict(), f"models/{args.run_name}/scaler.pt")
                
                torch.save(model.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
                torch.save(grad_scaler.state_dict(), f"models/{args.run_name}/model_{step}_scaler.pt")
            torch.save({'step': step, 'curr_losses' : curr_losses, 'losses': losses, 
                         'curr_accuracies' : curr_accuracies,'accuracies': accuracies}, 
                       f"results/{args.run_name}/log.pt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Paella_Test_4"
    args.model = "paella"
    args.dataset = "second_stage"
    args.urls = {
        "videos": "/fsx/phenaki/data/videos/tar_files/{0..1243}.tar",
        "images": "/fsx/phenaki/coyo-700m/coyo-data-2/{00000..20892}.tar"
    }
    args.total_steps = 300_000
    args.batch_size = 4
    args.num_workers = 10
    args.log_period = 2000
    args.extra_ckpt = 10_000
    args.accum_grad = 2

    args.vq_path = "/fsx/phenaki/src/models/model_120000.pt"
    args.dim = 1224  # 1224
    args.num_tokens = 8192
    args.max_seq_len = 6 * 16 * 16
    args.depth = 22  # 22
    args.dim_context = 1024  # for clip, 512 for T5
    args.heads = 22  # 22

    args.clip_len = 10
    args.skip_frames = 5

    args.n_nodes = 1
    args.dist_url = "env://"
    args.dist_backend = "nccl"
    args.no_set_device_rank = False

    print("Launching with args: ", args)
    train(args)
