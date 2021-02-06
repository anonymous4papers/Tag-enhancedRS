import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from TSML import TSML
from SML import SML
from dataset import TrainDataset, TestDataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--model', type=str, default='TSML')
    parser.add_argument('--dataset', type=str, default='ciao', help='Choose a dataset.')
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--rand_seed', type=int, default=34567)
    parser.add_argument('--top_k', nargs='?', default=[5, 10])
    parser.add_argument('--tag_pair', type=float, default=0.001)
    parser.add_argument('--nuc_reg', type=float, default=0.01)
    parser.add_argument('--margin', type=float, default=1.0)
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()
    np.random.seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)


    args.save_path = "./restore/{}_{}.pt".format(args.model, args.dataset)

    negatives_filename = 'data/' + args.dataset + '/negatives.dat'
    train_filename = 'data/' + args.dataset + '/train.dat'
    val_filename = 'data/' + args.dataset + '/val.dat'
    test_filename = 'data/' + args.dataset + '/test.dat'

    train = TrainDataset(train_filename, negatives_filename)
    valid = TestDataset(val_filename)
    test = TestDataset(test_filename)

    n_user = max(train.n_user, valid.n_user, test.n_user)
    n_item = max(train.n_item, valid.n_item, test.n_item)

    train_pos, valid_pos, test_pos = train.get_pos(n_item), valid.get_pos(n_item), test.get_pos(n_item)
    train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid = DataLoader(valid, batch_size=args.batch_size_test)
    test = DataLoader(test, batch_size=args.batch_size_test)

    model_class = TSML
    if args.model == 'SML':
        model_class = SML
    model = model_class(
        device=args.device, n_users=n_user, n_items=n_item,
        embed_dim=128,
        margin=args.margin,
        rand_seed=args.rand_seed,
        args=args
    )
    model.train_test(model.loss, model.test_rank,
                     train, (valid, train_pos, valid_pos), (test, train_pos, test_pos),
                     n_epochs=args.num_epoch, lr=args.lr, n_metric=3, savepath=args.save_path, topk=args.top_k,
                     small_better=[False] * 3)


