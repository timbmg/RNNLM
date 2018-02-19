import argparse
import torch
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from utils import to_var
from RNNLM import RNNLM
from TextDataset import TextDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",               type=str,   default="data")
    parser.add_argument("--batch_size",             type=int,   default=64)
    parser.add_argument("--epochs",                 type=int,   default=30)
    parser.add_argument("--learning_rate",          type=float, default=0.001)
    parser.add_argument("--print_every",            type=int,   default=100)
    parser.add_argument("--max_sequence_length",    type=int,   default=40)
    parser.add_argument("--min_occ",                type=int,   default=10)
    parser.add_argument("--pre",                    type=str,   default='wap')

    args = parser.parse_args()

    splits = ['train', 'valid', 'test']

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = TextDataset(root=args.data_dir, split=split, pre=args.pre,
                                      max_sequence_length=args.max_sequence_length,
                                      min_occ=args.min_occ)

    model = RNNLM(vocab_size=datasets['train'].vocab_size, padding_idx=datasets['train'].pad_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=datasets['train'].pad_token)



    for epoch in range(args.epochs):

        tracker = defaultdict(torch.Tensor)

        for split in splits[:2]:

            data = DataLoader(dataset=datasets[split], batch_size=args.batch_size)

            for iteration, batch in enumerate(data):

                for key, item in batch.items():
                    if torch.is_tensor(item):
                        batch[key] = to_var(item)

                inputs = batch['inputs']
                lengths = batch['len']
                targets = batch['targets']

                # sort inputs and targets based on sequence length
                lengths, idx = torch.sort(lengths, descending=True)
                inputs = inputs[idx]
                targets = targets[idx]

                # remove padding from targets and flatten
                targets = pack_padded_sequence(targets, lengths.data.tolist(), batch_first=True)[0]

                # get predictions for words
                predictions = model(inputs, lengths)

                # compute loss
                loss = criterion(predictions, targets)


                # if training, optimize
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # bookkeeping
                if iteration % args.print_every == 0 or iteration+1 == len(data):
                    print("%s Batch %04d/%i, Loss %.4f"%(split.upper(), iteration, len(data), loss.data[0]))

                tracker['loss'] = torch.cat((tracker['loss'], loss.data))

            print("%s Epoch %02d/%i, Mean Loss: %.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['loss'])))
