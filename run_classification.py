import os
import shutil
import argparse
from collections import defaultdict

import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from metrics import BinaryClassificationMeter
from utils.early_stopping import EarlyStopping

from models import make_model
from datasets import make_dataset, make_data_loader
from losses import make_loss
from optims import make_optimizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(phase, metrics):
    loss = metrics.pop('loss')
    conf_mat = metrics.pop('conf_mat')

    loss_log_str = '{:6}loss: {:.6f}'.format(phase, loss)
    other_metrics_log_str = ' '.join([
        '{}: {:.6f}'.format(k, v)
        for k, v in metrics.items()
    ])

    metrics['loss'] = loss
    metrics['conf_mat'] = conf_mat
    print(f'{loss_log_str} {other_metrics_log_str}\nConfusion matrix:\n{conf_mat}')


def write_metrics(epoch, metrics, writer):
    conf_mat = metrics.pop('conf_mat')
    for k, v in metrics.items():
        writer.add_scalar(f'metrics/{k}', v, epoch)

    metrics['conf_mat'] = conf_mat


def init_experiment(config):
    if os.path.exists(config.experiment_dir):
        def ask():
            return input(f'Experiment "{config.experiment_name}" already exists. Delete (y/n)?')

        answer = ask()
        while answer not in ('y', 'n'):
            answer = ask()

        delete = answer == 'y'
        if not delete:
            exit(1)

        shutil.rmtree(config.experiment_dir)

    os.makedirs(config.experiment_dir)
    with open(config.config_save_path, 'w') as dest_file:
        config.dump(stream=dest_file)


def train(model, optimizer, train_loader, loss_f, device, threshold=0.5):
    model.train()

    meter = BinaryClassificationMeter(device)
    metrics = defaultdict(lambda: 0)

    for data, target in tqdm(train_loader):
        data = data.to(device).float()
        target = target.to(device).float()

        output = model(data)
        loss = loss_f(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        metrics['loss'] += loss.item() * batch_size

        output = (output >= threshold).type(torch.uint8)
        target = target.type(torch.uint8)
        meter.update(target, output)

    dataset_length = len(train_loader.dataset)
    metrics['loss'] /= dataset_length
    metrics.update(meter.get_metrics())

    return metrics


def val(model, val_loader, loss_f, device, threshold=0.5):
    model.eval()

    meter = BinaryClassificationMeter(device)
    metrics = defaultdict(lambda: 0)

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data = data.to(device).float()
            target = target.to(device).float()

            output = model(data)
            loss = loss_f(output, target)

            batch_size = target.shape[0]
            metrics['loss'] += loss.item() * batch_size

            output = (output >= threshold).type(torch.uint8)
            target = target.type(torch.uint8)
            meter.update(target, output)

    dataset_length = len(val_loader.dataset)
    metrics['loss'] /= dataset_length
    metrics.update(meter.get_metrics())

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file-path',
        required=True,
        type=str,
        help='path to the configuration file'
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_path = args.config_file_path

    config = get_config(config_path, new_keys_allowed=True)

    config.defrost()
    config.experiment_dir = os.path.join(config.log_dir, config.experiment_name)
    config.tb_dir = os.path.join(config.experiment_dir, 'tb')
    config.model.best_checkpoint_path = os.path.join(config.experiment_dir, 'best_checkpoint.pt')
    config.model.last_checkpoint_path = os.path.join(config.experiment_dir, 'last_checkpoint.pt')
    config.config_save_path = os.path.join(config.experiment_dir, 'segmentation_config.yaml')
    config.freeze()

    init_experiment(config)
    set_random_seed(config.seed)

    train_dataset = make_dataset(config.train.dataset)
    train_loader = make_data_loader(config.train.loader, train_dataset)

    val_dataset = make_dataset(config.val.dataset)
    val_loader = make_data_loader(config.val.loader, val_dataset)

    device = torch.device(config.device)
    model = make_model(config.model).to(device)

    optimizer = make_optimizer(config.optim, model.parameters())
    scheduler = None

    loss_f = make_loss(config.loss)

    early_stopping = EarlyStopping(
        **config.stopper.params
    )

    train_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))

    for epoch in range(1, config.epochs + 1):
        print(f'Epoch {epoch}')
        train_metrics = train(model, optimizer, train_loader, loss_f, device)
        write_metrics(epoch, train_metrics, train_writer)
        print_metrics('Train', train_metrics)

        val_metrics = val(model, val_loader, loss_f, device)
        write_metrics(epoch, val_metrics, val_writer)
        print_metrics('Val', val_metrics)

        early_stopping(val_metrics['loss'])
        if config.model.save and early_stopping.counter == 0:
            torch.save(model.state_dict(), config.model.best_checkpoint_path)
            print('Saved best model checkpoint to disk.')
        if early_stopping.early_stop:
            print(f'Early stopping after {epoch} epochs.')
            break

        if scheduler:
            scheduler.step()

    train_writer.close()
    val_writer.close()

    if config.model.save:
        torch.save(model.state_dict(), config.model.last_checkpoint_path)
        print('Saved last model checkpoint to disk.')


if __name__ == '__main__':
    main()
