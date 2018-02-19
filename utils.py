import torch
import random
import io
import os
import re
from collections import Counter, OrderedDict, defaultdict
from torch.nn import functional
from torch.autograd import Variable

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def txt2splits(root, textfile, sizes, pre=''):
    """Given a text file creates random into train, validation and test splits along the lines in the file.

    Parameters
    ----------
    root : str
        Root directory of the original text file.
    textfile : str
        The text file.
    sizes : list
        Sizes of the splits, does not have to be normalized.
    pre : str
        Will be prepended to every output file.

    """

    splits = defaultdict(str)

    norm_sizes = [s/sum(sizes) for s in sizes]

    with open(os.path.join(root, textfile), 'r') as file:

        for line in file.readlines():

            if re.match(r'^\s*$', line):
                # skip empty lines
                continue

            line = line.strip()

            p = random.random()

            if p < norm_sizes[0]:
                splits['train'] += "\n"+line
            elif norm_sizes[0] <= p < norm_sizes[0]+norm_sizes[1]:
                splits['valid'] += "\n"+line
            else:
                splits['test'] += "\n"+line

    for split, text in splits.items():
        with open(os.path.join(root, pre + '.' + split + '.txt'), 'w') as file:
            file.write(text)
