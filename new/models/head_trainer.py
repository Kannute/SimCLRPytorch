import logging
import os
import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import utils


class HeadTrainer(object):

    def __init__(self, **kwargs):
        self.model = kwargs['model'].to(kwargs.get('device'))
        self.args = kwargs
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=kwargs['log_dir'])
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.get('device'))

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.get('fp16_precision'))

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.get('epochs')} epochs.")
        logging.info(f"Training with gpu: {self.args.get('disable_cuda')}.")
        for epoch_counter in range(self.args.get('epochs')):

            g = torch.Generator()
            g.manual_seed(0)
            for images, labels in tqdm((train_loader)):
                images = torch.cat((images, images), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                images = images.to(self.args.get('device'))
                labels = labels.to(self.args.get('device'))
                output = self.model(images)
                loss = self.criterion(output, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.get('log_every_n_steps') == 0:
                    top1, top5 = utils.accuracy(output, labels, top_k=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tBest ACC: {top1[0]}")