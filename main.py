import os
import torch
import argparse

from split.split_spatial import *
from split.split_tcga import *

from SEG.train_SEG import *
from GEP.train_GEP import *

parser = argparse.ArgumentParser()
parser.add_argument('--run_type', type=str, default='full', help='select if ST and TCGA')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--sp_dir', type=str, default='io_data/ST/')

## SEG and GEP parameters
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--adj_threshold', type=float, default=0.2)
parser.add_argument('--nb_heads', type=int, default=12)
parser.add_argument('--nb_embed', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_step', type=int, default=5)
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=16)


parser.add_argument('--tcga_dir', type=str, default='io_data/TCGA/')



args = parser.parse_args()
cuda = not args.no_cuda and torch.cuda.is_available()

if cuda: gpu='cuda'
else: gpu='cpu'

## Generating the spots
## Assumes that spatial WSIs are in jpeg/jpg/tiff//png format and TCGA data in SVS format 
## Assumess that the ST count matrix is in n_samplesxn_genes shape

print("----------Processing for ST data------------")
split_spatial(args.sp_dir)
print("--------Splitting done for ST data---------\n\n--------Training SEG-------\n\n")
run_SEG(args.sp_dir, args.patience, args.adj_threshold, args.nb_heads, args.nb_embed, 
        args.n_epochs, args.lr, args.lr_step, args.lr_gamma, args.batch_size)



print("---------SEG trained-----------\n\n--------------Processing for TCGA data----------")
split_tcga(args.tcga_dir)
print("--------Splitting done for TCGA data---------\n\n--------Training GEP-------\n\n")
run_GEP(args.tcga_dir, args.patience, args.adj_threshold, args.nb_heads, args.nb_embed, 
        args.n_epochs, args.lr, args.lr_step, args.lr_gamma, args.batch_size)


