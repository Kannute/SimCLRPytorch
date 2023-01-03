#%%

#%%
import torch.nn as nn
import torchvision.models as models

import logging
import os

import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt

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


class PreTrainer(object):

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
                images = images.to(self.args.get('device'))
                labels = labels.to(self.args.get('device'))
                output = self.model(images)
                loss = self.criterion(output, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # top1kacc = accuracy(output, labels)
                #
                # if (top1kacc[0]>95 or (epoch_counter>50 and top1kacc[0]>80)):
                #     print("Early stopping. Stopped on epoch: ", epoch_counter, "\naccuracy: ", top1kacc)
                #     break  # koniec uczenia

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


class HeadTrainerNew(object):

    def __init__(self, **kwargs):
        self.model = kwargs['model'].to(kwargs.get('device'))
        self.args = kwargs
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
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
class HeadTrainer:
    def __init__(self, train_dataset, test_dataset, batch_size=256):
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, model, optimizer, loss_fn=torch.nn.CrossEntropyLoss, n_epochs=100):
        self.logs = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
        model = model.to(self.device)
        correct, numel = 0, 0
        for e in range(1, n_epochs + 1):
            model.train()
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                y_pred = torch.argmax(output, dim=1)
                correct += torch.sum(y_pred == y).item()
                numel += self.batch_size
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

            self.logs['train_loss'].append(loss.item())
            self.logs['train_accuracy'].append(correct / numel)
            correct, numel = 0, 0

            model.eval()
            with torch.no_grad():
                for x_test, y_test in self.test_loader:
                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)
                    output = model(x_test)
                    y_pred = torch.argmax(output, dim=1)
                    correct += torch.sum(y_pred == y_test).item()
                    numel += self.batch_size
                loss = loss_fn(output, y_test)

            self.logs['test_loss'].append(loss.item())
            self.logs['test_accuracy'].append(correct / numel)
            correct, numel = 0, 0

        return self.logs


def show_results(orientation='horizontal', accuracy_bottom=None, loss_top=None, **histories):
    if orientation == 'horizontal':
        f, ax = plt.subplots(1, 2, figsize=(16, 5))
    else:
        f, ax = plt.subplots(2, 1, figsize=(16, 16))
    for i, (name, h) in enumerate(histories.items()):
        if len(histories) == 1:
            ax[0].set_title("Best test accuracy: {:.2f}% (train: {:.2f}%)".format(
                max(h['test_accuracy']) * 100,
                max(h['train_accuracy']) * 100
            ))
        else:
            ax[0].set_title("Accuracy")
        ax[0].plot(h['train_accuracy'], color='C%s' % i, linestyle='--', label='%s train' % name)
        ax[0].plot(h['test_accuracy'], color='C%s' % i, label='%s test' % name)
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('accuracy')
        if accuracy_bottom:
            ax[0].set_ylim(bottom=accuracy_bottom)
        ax[0].legend()

        if len(histories) == 1:
            ax[1].set_title("Minimal train loss: {:.4f} (test: {:.4f})".format(
                min(h['train_loss']),
                min(h['test_loss'])
            ))
        else:
            ax[1].set_title("Loss")
        ax[1].plot(h['train_loss'], color='C%s' % i, linestyle='--', label='%s train' % name)
        ax[1].plot(h['test_loss'], color='C%s' % i, label='%s test' % name)
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('loss')
        if loss_top:
            ax[1].set_ylim(top=loss_top)
        ax[1].legend()

    plt.show()
