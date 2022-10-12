import math
import os
import torch
import wandb
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from loss.loss import FirstStageLoss


def train(proc_id, args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True
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
        dist.init_process_group(backend="nccl", init_method="file:///fsx/mas/paella_unet/dist_file",
                                world_size=args.n_nodes * len(args.devices),
                                rank=proc_id + len(args.devices) * args.node_id)
        torch.set_num_threads(6)

    if args.model == "vivit":
        print(f"Model: DenoiseGIC")
        model = None.to(device)
    elif args.model == "vivq":
        model = None.to(device)
    else:
        raise NotImplementedError()

    if not proc_id and args.node_id == 0:
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    criterion = FirstStageLoss()
    lr = 3e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer_discriminator = optim.AdamW(criterion.discriminator.parameters(), lr=lr*1e-2)
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    if not proc_id and args.node_id == 0:
        wandb.watch(model)
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
            print("Loaded model and EMA model.")
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

    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step) if args.node_id == 0 and proc_id == 0 else enumerate(dataset, start=start_step)
    model.train()
    for step, videos in pbar:
        videos = videos.to(device)

        recon = model(videos)
        loss, d_loss = criterion(videos, recon)
        loss_adjusted = loss / grad_accum_steps

        loss_adjusted.backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer_discriminator.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if not proc_id and args.node_id == 0:
            pbar.set_postfix({
                'loss': total_loss / (step + 1),
                'ppx': np.exp(total_loss / (step + 1)),
                'lr': optimizer.param_groups[0]['lr']
            })
            wandb.log({
                "loss": total_loss / (step + 1),
                "ppx": np.exp(total_loss / (step + 1)),
                "lr": optimizer.param_groups[0]['lr'],
            })
            logger.add_scalar("loss", total_loss / (step + 1), global_step=step)
            logger.add_scalar("ppx", np.exp(total_loss / (step + 1)), global_step=step)
            logger.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=step)

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:
            print(f"Step {step} - loss {total_loss / (step + 1)} - acc {total_acc / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}")

            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                pass

            if step % args.extra_ckpt == 0:
                torch.save(model.module.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
            torch.save(model.module.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save({'step': step, 'losses': losses, 'accuracies': accuracies}, f"results/{args.run_name}/log.pt")



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
    args.run_name = "vivit_test"
    args.model = "vivit"
    args.dataset_type = "webdataset"
    args.total_steps = 5_000_000
    args.batch_size = 22
    args.num_workers = 10
    args.log_period = 5000
    args.extra_ckpt = 50_000
    args.accum_grad = 1

    args.n_nodes = 8
    args.node_id = int(os.environ["SLURM_PROCID"])
    args.devices = [0, 1, 2, 3, 4, 5, 6, 7]

    args.dataset_path = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
    print("Launching with args: ", args)
    launch(
        args
    )
