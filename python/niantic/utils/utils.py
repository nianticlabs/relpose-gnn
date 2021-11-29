import os

import torch
from torchvision.datasets.folder import default_loader


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print
        'Could not load image {:s}, IOError: {:s}'.format(filename, e)
        return None
    except:
        print
        'Could not load image {:s}, unexpected error'.format(filename)
        return None

    return img


def save_checkpoint(logdir, epoch, model, optimizer, train_criterion):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    filename = os.path.join(logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
    checkpoint_dict = \
        {'epoch': epoch, 'model_state_dict': model.state_dict(),
         'optim_state_dict': optimizer.state_dict(),
         'criterion_state_dict': train_criterion.state_dict()}
    torch.save(checkpoint_dict, filename)
