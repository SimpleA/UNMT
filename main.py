import sys
import argparse
from train import Test
from test import Test
from utils import USE_CUDA


parser = argparse.ArgumentParser(prog='main.py', description='Unsupervised Neural Maching Translation')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--load', default=None)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--print_every', type=str, default=1)
parser.add_argument('--save_every', type=str, default=1)
parser.add_argument('--postfix', type=str, default='None')
parser.add_argument('--save_dir', type=str, default='save/')
args = parser.parse_args()

def main(args):
    if USE_CUDA:
        print("Using Cuda device...")
    if args.train:
       if args.load:
           print("Loading saved model {} ...".format(args.load))
       train(args.epoch, args.lr, args.batch_size, args.hidden_size, args.print_every, args.save_every, args.postfix,args.save_dir) 
    elif args.test:
        test()
    else:
        print ('Error: Please use --train or --test flag')
        sys.exit()
    
