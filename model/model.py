import torch.nn as nn
import torchvision.models as models

import logging
import os

import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.manual_seed(0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ConvModel(nn.Module):
    def __init__(self, out_dim, dataset: str = ''):
        super(ConvModel, self).__init__()
        self.resnet_model = models.resnet18(pretrained=False, num_classes=out_dim)

        self.backbone = self.resnet_model
        if dataset.upper() == 'MNIST':
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.backbone(x)


class SimCLR(object):

    def __init__(self, **kwargs):
        self.model = kwargs['model'].to(kwargs.get('device'))
        self.args = kwargs
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.get('device'))

    def train(self, train_loader, shuffle: bool = True):

        scaler = GradScaler(enabled=self.args.get('fp16_precision'))

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.get('epochs')} epochs.")
        logging.info(f"Training with gpu: {self.args.get('disable_cuda')}.")
        for epoch_counter in range(self.args.get('epochs')):
            g = torch.Generator()
            g.manual_seed(0)
            for images, labels in tqdm((train_loader)):
                images = torch.cat((images, images), dim=0)
                if shuffle:
                    labels = labels[torch.randperm(labels.size()[0], generator=g)]
                labels = torch.cat((labels, labels), dim=0)
#czy shuffle na pewno dziala?
#jakie bedzie acc validacyjne jesli shufflawanie bedzie wylaczone w pretreningu. Bylo bardzo wysokie z tego co pamietam
#
                images = images.to(self.args.get('device'))
                labels = labels.to(self.args.get('device'))
                output = self.model(images)
                #output - podejrzane
                loss = self.criterion(output, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                if n_iter % self.args.get('log_every_n_steps') == 0:
                    top1, top5 = accuracy(output, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tBest ACC: {top1[0]}")

#%%
