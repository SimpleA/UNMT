import sys
import argparse
from train import Train
from test import Test
from utils import USE_CUDA


parser = argparse.ArgumentParser(prog='main.py', description='Unsupervised Neural Maching Translation')
parser.add_argument('--verbose', action = 'store_true', default = False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--l1', type=str, default = 'en')
parser.add_argument('--l2', type=str, default = 'fr')
parser.add_argument('--load', default=None)
parser.add_argument('--iteration', type=int, default=150000)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--vocab_size', type = int, default=49000)
parser.add_argument('--print_every', type=str, default=5)
parser.add_argument('--save_every', type=str, default=100)
parser.add_argument('--postfix', type=str, default='None')
parser.add_argument('--save_dir', type=str, default='save/')
parser.add_argument('--model_path', type=str, default='./1.pkl')
args = parser.parse_args()

def main(args):
    if USE_CUDA:
        print("Using Cuda device...")
    if args.train:
       if args.load:
           print("Loading saved model {} ...".format(args.load))
       Train(args.verbose, args.l1, args.l2, args.iteration, args.lr, args.batch_size, args.hidden_size,args.vocab_size, \
		args.print_every, args.save_every, args.postfix,args.save_dir) 
    elif args.test:
        Test(args.l1, args.l2, args.batch_size, args.hidden_size, args.vocab_size, args.model_path)
    else:
        print ('Error: Please use --train or --test flag')
        sys.exit()

if __name__ == '__main__':
	main(args)