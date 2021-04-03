import os
import shutil
import argparse
from collections import defaultdict

import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from metrics import make_metrics, AverageMetricsMeter, calculate_iou, calculate_iou6
from utils.early_stopping import EarlyStopping

from models import make_model
from datasets import make_dataset, make_data_loader, make_sampler
from losses import make_loss
from optims import make_optimizer

import torch.nn as nn


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(phase, metrics):
    loss = metrics.pop('loss')

    loss_log_str = '{:6}loss: {:.6f}'.format(phase, loss)
    other_metrics_log_str = ' '.join([
        '{}: {:.6f}'.format(k, v)
        for k, v in metrics.items()
    ])

    metrics['loss'] = loss
    print(f'{loss_log_str} {other_metrics_log_str}')


def write_metrics(epoch, metrics, writer):
    for k, v in metrics.items():
        writer.add_scalar(f'metrics/{k}', v, epoch)


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


def train(model, optimizer, train_loader, loss_f, metric_fns, use_valid_masks, device):
    model.train()
    www = 0
    meter = [AverageMetricsMeter(metric_fns, device) for _ in range(7)]
    metrics = defaultdict(lambda: 0)
    intersection = np.zeros(7)
    union = np.zeros(7)
    dataset_length = 0
    confusion_matrix = np.zeros((7, 7))

    for data, target, meta in tqdm(train_loader):
        data = data.to(device).float()
        target = target.to(device).float()
        output = model(data)
        if (www % 100 == 5):
            out = nn.Sigmoid()(output).detach().cpu().numpy()
            out = out > 0.7
            tar = target.cpu().numpy()
            background = 1 - np.max(tar, axis=1)
            background = background.reshape((tar.shape[0], 1, tar.shape[2], tar.shape[3]))
            tar = np.concatenate((background, tar), axis=1)
            background = 1 - np.max(out, axis=1)
            background = background.reshape((out.shape[0], 1, out.shape[2], out.shape[3]))
            out = np.concatenate((background, out), axis=1)
            for i in range(7):
                if np.unique(tar[0][i]).shape[0] == 2:
                    cv2.imwrite(f'train2/{i}/target_not_empty{www}.png', tar[0][i] * 255)
                    cv2.imwrite(f'train2/{i}/output_not_empty{www}.png', out[0][i] * 255)
                else:
                    cv2.imwrite(f'train2/{i}/target{www}.png', tar[0][i] * 255)
                    cv2.imwrite(f'train2/{i}/output{www}.png', out[0][i] * 255)

        www += 1
        w = calculate_iou6(target, output)
        intersection += w[0]
        union += w[1]
        if use_valid_masks:
            valid_mask = meta["valid_pixels_mask"].to(device)
        else:
            valid_mask = torch.ones_like(meta["valid_pixels_mask"]).to(device)
        loss = loss_f(output, target, valid_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = target.shape[0]
        dataset_length += batch_size
        metrics['loss'] += loss.item() * batch_size
    metrics['loss'] /= dataset_length
    metrics['average_iou'] = 0
    for i in range(7):
        x = meter[i].get_metrics()
        metrics['iou_score ' + str(i)] = intersection[i] / union[i]
        metrics['average_iou'] += intersection[i] / union[i] / 7

    return metrics


def val(model, val_loader, loss_f, metric_fns, use_valid_masks, device):
    model.eval()

    meter = [AverageMetricsMeter(metric_fns, device) for _ in range(7)]
    metrics = defaultdict(lambda: 0)
    www = 0
    intersection = np.zeros(7)
    union = np.zeros(7)
    with torch.no_grad():
        for data, target, meta in tqdm(val_loader):

            data = data.to(device).float()
            target = target.to(device).float()

            output = model(data)
            if www % 100 == 4:
                out = nn.Sigmoid()(output).detach().cpu().numpy()
                out = (out > 0.7) * out
                tar = target.cpu().numpy()
                background = 1 - np.max(tar, axis=1)
                background = background.reshape((tar.shape[0], 1, tar.shape[2], tar.shape[3]))
                tar = np.concatenate((background, tar), axis=1)
                background = 1 - np.max(out, axis=1)
                background = background.reshape((out.shape[0], 1, out.shape[2], out.shape[3]))
                out = np.concatenate((background, out), axis=1)
                for i in range(7):
                    if np.unique(tar[0][i]).shape[0] == 2:
                        cv2.imwrite(f'test2/{i}/target_not_empty{www}.png', tar[0][i] * 255)
                        cv2.imwrite(f'test2/{i}/output_not_empty{www}.png', out[0][i] * 255)
                    else:
                        cv2.imwrite(f'test2/{i}/target{www}.png', tar[0][i] * 255)
                        cv2.imwrite(f'test2/{i}/output{www}.png', out[0][i] * 255)

            www += 1

            if use_valid_masks:
                valid_mask = meta["valid_pixels_mask"].to(device)
            else:
                valid_mask = torch.ones_like(meta["valid_pixels_mask"]).to(device)
            loss = loss_f(output, target, valid_mask)
            batch_size = target.shape[0]
            metrics['loss'] += loss.item() * batch_size
            w = calculate_iou6(target, output)
            intersection += w[0]
            union += w[1]

            '''
            for i in range(6):
                for j in range(target.shape[0]):
                    if torch.unique(target[0][i]).shape[0] == 2:
                        x = meter[i].update(target[j, i, :, :], output[j, i, :, :])
            '''
    dataset_length = len(val_loader.dataset)
    metrics['loss'] /= dataset_length
    metrics['average_iou'] = 0
    for i in range(7):
        x = meter[i].get_metrics()
        metrics['iou_score ' + str(i)] = intersection[i] / union[i]
        metrics['average_iou'] += intersection[i] / union[i] / 7

    return metrics


def test(model, test_loader, use_valid_masks, device, save_to, threshold=0.5):
    model.eval()

    with torch.no_grad():
        for data, meta in tqdm(test_loader):
            data = data.to(device).float()
            output = model(data)
            if use_valid_masks:
                valid_mask = meta["valid_pixels_mask"].to(device)
            else:
                valid_mask = torch.ones_like(meta["valid_pixels_mask"]).to(device)
            batch_size = output.shape[0]
            size = output.shape
            answer = np.zeros((size[0], 1, size[2], size[3]))
            output = nn.Sigmoid()(output)
            for i in range(batch_size):
                answer[i, 0, :, :] = (np.argmax(output[i, :, :, :].cpu().numpy(), axis=0) + 1) * (
                        np.ndarray.max(output[i, :, :, :].cpu().numpy(), axis=0) > threshold)
                cv2.imwrite(os.path.join(save_to, meta['image_id'][i].split('.')[0] + '.png'), answer[i, 0])
    return None


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
    train_sampler = make_sampler(config.train.loader.params.sampler, config.train.dataset)
    train_loader = make_data_loader(config.train.loader, train_dataset, train_sampler)

    val_dataset = make_dataset(config.val.dataset)
    val_loader = make_data_loader(config.val.loader, val_dataset)
    test_dataset = make_dataset(config.test.dataset)
    test_loader = make_data_loader(config.test.loader, test_dataset)

    device = torch.device(config.device)
    model = make_model(config.model).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    scheduler = None

    loss_f = make_loss(config.loss)
    metrics = make_metrics(config.metrics)

    early_stopping = EarlyStopping(
        **config.stopper.params
    )

    train_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))
    use_valid_masks_train = config.train.use_valid_masks
    use_valid_masks_val = config.train.use_valid_masks
    save_to = config.test.save_to
    best_loss = 1000000
    if len(config.optim.params.lr) != 1:
        step = config.epochs / (len(config.optim.params.lr) - 1)
    else:
        step = config.epochs * 2
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch + 1}")

        optimizer = make_optimizer(config.optim, model.parameters(), int(epoch // step))
        if config.to_train:
            try:
                train_sampler.randomize()
            except:
                print("ERROR TO RANDOMIZE")
            train_metrics = train(model, optimizer, train_loader, loss_f, metrics, use_valid_masks_train, device)
            write_metrics(epoch, train_metrics, train_writer)
            print_metrics('Train', train_metrics)

        val_metrics = val(model, val_loader, loss_f, metrics, use_valid_masks_val, device, config.to_train)
        loss = val_metrics['loss']
        write_metrics(epoch, val_metrics, val_writer)
        print_metrics('Val', val_metrics)
        early_stopping(val_metrics['loss'])
        test(model, test_loader, True, device, save_to, config.to_train)
        if not config.to_train:
            # ! If we just test we need only one epoch
            return
        if config.model.save and early_stopping.counter == 0:
            if loss < best_loss:
                print("Saved")
                best_loss = min(loss, best_loss)
                print(loss, best_loss)

                torch.save(model.state_dict(), config.model.best_checkpoint_path)
                print('Saved best model checkpoint to disk.')
        else:
            print("Loss start growing")
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
