# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionParticlePipeline, DDIMScheduler, EulerDiscreteScheduler
import torch
from coco_data_loader import text_image_pair
from PIL import Image
import os
import pandas as pd
import argparse
import torch.nn as nn
from torch_utils import distributed as dist
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=30, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--w', type=float, default=7.5)
parser.add_argument('--s_noise', type=float, default=1.)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default='DDIM')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--second', action='store_true', default=False, help='second order ODE')
parser.add_argument('--sigma', action='store_true', default=False, help='use sigma')
parser.add_argument('--coeff', type=float, default=0.)
parser.add_argument('--dino', action='store_true', default=False, help='use dino')
parser.add_argument('--csv_path', type=str, default='~/data/coco/subset.csv')

args = parser.parse_args()

dist.init()

if dist.get_rank() == 0:
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
torch.distributed.barrier()


dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))
# define dataset / data_loader

df = pd.read_csv(args.csv_path)
all_text = list(df['caption'])
all_text = all_text[: args.max_cnt]
num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]

##### load stable diffusion models #####
pipe = StableDiffusionParticlePipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
dist.print0("default scheduler config:")
dist.print0(pipe.scheduler.config)
pipe = pipe.to("cuda")

if args.dino:
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to("cuda")

if args.scheduler == 'DDIM':
    # recommend using DDIM with Restart
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == 'ODE':
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
else:
    raise NotImplementedError

generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)

##### setup save configuration #######
if args.name is None:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_dino_{args.dino}_coeff_{args.coeff}')
else:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_dino_{args.dino}_coeff_{args.coeff}_name_{args.name}')

dist.print0("save images to {}".format(save_dir))

if dist.get_rank() == 0 and not os.path.exists(save_dir):
    os.mkdir(save_dir)

## generate images ##
assert args.bs == 1
for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    torch.distributed.barrier()
    if len(mini_batch) == 0:
        continue
    text = list(mini_batch)
    # generate four images using the same text
    text = text * 4
    if args.dino:
        out, _ = pipe.dino(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, restart=args.restart,
                    second_order=args.second, dist=dist, S_noise=args.s_noise, coeff=args.coeff, dino=dino)
    else:
        out, _ = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, restart=args.restart,
                    second_order=args.second, dist=dist, S_noise=args.s_noise, coeff=args.coeff)
    image = out.images

    for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
        path = os.path.join(save_dir, f'{global_idx}')
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(len(image)):
            image[i].save(os.path.join(path, f'{i}.png'))

# Done.
torch.distributed.barrier()
if dist.get_rank() == 0:
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(save_dir, 'subset.csv'))