import os
import shutil
import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn

from config import get_config
from metrics import make_metrics, AverageMetricsMeter, calculate_iou, calculate_confusion_matrix
from utils.early_stopping import EarlyStopping

from models import make_model
from datasets import make_dataset, make_data_loader, make_sampler
from losses import make_loss
from optims import make_optimizer

import torch.nn as nn
import matplotlib.pyplot as plt

import ttach as tta


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tta_transform():
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
        ]
    )


def print_metrics(phase, metrics):
    loss = metrics.pop('loss')
    loss_log_str = '{:6}loss: {:.6f}'.format(phase, loss)
    try:
        confusion_matrix = metrics.pop('confusion_matrix')
        print("Confusion matrix:\n", confusion_matrix, '\n\n')
    except:
        pass
    other_metrics_log_str = ' '.join([
        '{}: {:.6f}'.format(k, v)
        for k, v in metrics.items()
    ])

    metrics['loss'] = loss
    print(f'{loss_log_str} {other_metrics_log_str}')


def write_confusion_matrix(writer, conf_mat, step):
    fig = plt.figure()
    conf_mat = conf_mat / conf_mat.max() * 100
    sn.heatmap(conf_mat, annot=True)

    writer.add_figure('confusion_matrix', fig, step)


def write_metrics(epoch, metrics, writer):
    confusion_matrix = metrics['confusion_matrix']
    write_confusion_matrix(writer, confusion_matrix, step=epoch)
    for k, v in metrics.items():
        try:
            writer.add_scalar(f'metrics/{k}', v, epoch)
        except:
            pass


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


def save_files(to, target, output, idx):
    for i in range(7):
        if torch.unique(target[0][i]).shape[0] == 2:
            cv2.imwrite(f'{to}/{i}/target_not_empty{idx}.png', target[0][i].cpu().numpy() * 255)
            cv2.imwrite(f'{to}/{i}/output_not_empty{idx}.png', output[0][i].detach().cpu().numpy() * 255)
        else:
            cv2.imwrite(f'{to}/{i}/target{idx}.png', target[0][i].cpu().numpy() * 255)
            cv2.imwrite(f'{to}/{i}/output{idx}.png', output[0][i].detach().cpu().numpy() * 255)


def crop_image(data, strip=128):
    k = 256 // strip + 1
    data1 = torch.zeros(size=(data.shape[0] * k * k, data.shape[1], 256, 256))
    for i in range(data.shape[0]):
        for d in range(k * k):
            data1[k * k * i + d, :, :, :] = data[i, :, 128 * (d % k):256 + 128 * (d % k),
                                            128 * (d // k):256 + 128 * (d // k)]

    return data1


def uncrop_image(data, strip=128):
    k = 256 // strip + 1
    data1 = torch.zeros(size=(data.shape[0] // (k * k), data.shape[1], 512, 512)).to(data.device)
    for i in range(0, data.shape[0], k * k):
        to = i // (k * k)
        for d in range(k * k):
            data1[to, :, 128 * (d % k):256 + 128 * (d % k),
            128 * (d // k):256 + 128 * (d // k)], _ = torch.max(torch.cat((data[i + d].unsqueeze(0),
                                                                           data1[to, :,
                                                                           128 * (d % k):256 + 128 * (d % k),
                                                                           128 * (d // k):256 + 128 * (
                                                                                   d // k)].unsqueeze(0)), dim=0),
                                                                dim=0)
    return data1


def train(model, optimizer, train_loader, loss_f, metric_fns, use_valid_masks, device, n=10):
    model.train()
    www = 0
    meter = [AverageMetricsMeter(metric_fns, device) for _ in range(n)]
    metrics = defaultdict(lambda: 0)
    intersection = torch.zeros(n)
    union = torch.zeros(n)
    metrics['confusion_matrix'] = torch.zeros((n, n))
    for data, target, meta in tqdm(train_loader):
        data = crop_image(data, 256)
        data = data.to(device).float()
        target = target.to(device).float()
        output = model(data)
        output = uncrop_image(output, 256)
        # metrics['confusion_matrix'] += calculate_confusion_matrix(target, output)
        if (www % 100 == 2):
            save_files('train', target, output, www)
        www += 1
        w = calculate_iou(target, output)
        intersection += w[0]
        union += w[1]
        valid_mask = meta["valid_pixels_mask"].to(device)
        loss = loss_f(output, target, valid_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = target.shape[0]
        metrics['loss'] += loss.item() * batch_size

    dataset_length = len(train_loader.dataset)
    metrics['loss'] /= dataset_length
    metrics['average_iou'] = 0
    for i in range(n):
        x = meter[i].get_metrics()
        metrics['iou_score ' + str(i)] = intersection[i] / union[i]
        metrics['average_iou'] += intersection[i] / union[i] / n
    print(intersection / union)

    return metrics


def val(model, val_loader, loss_f, metric_fns, use_valid_masks, device, to_train, strip=256, n=10):
    model.eval()
    meter = [AverageMetricsMeter(metric_fns, device) for _ in range(n)]
    metrics = defaultdict(lambda: 0)
    www = 0
    intersection = torch.zeros(n)
    union = torch.zeros(n)
    metrics['confusion_matrix'] = torch.zeros((n,n))

    if not to_train:
        print("HERE")
        tta_model = tta.SegmentationTTAWrapper(model, tta_transform(), merge_mode='max')
    else:
        tta_model = model

    with torch.no_grad():
        for data, target, meta in tqdm(val_loader):
            data = crop_image(data, strip)
            data = data.to(device).float()
            target = target.to(device).float()
            output = tta_model(data)
            output = uncrop_image(output, strip)
            # metrics['confusion_matrix'] += calculate_confusion_matrix(target, output)
            if (www % 100 == 2):
                save_files('test', target, output, www)
            www += 1
            if use_valid_masks:
                valid_mask = meta["valid_pixels_mask"]
            else:
                valid_mask = torch.ones_like(meta["valid_pixels_mask"])
            valid_mask = valid_mask.to(device)
            loss = loss_f(output, target, valid_mask)
            batch_size = target.shape[0]
            metrics['loss'] += loss.item() * batch_size
            w = calculate_iou(target, output)
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
    for i in range(n):
        x = meter[i].get_metrics()
        metrics['iou_score' + str(i)] = intersection[i] / union[i]
        metrics['average_iou'] += intersection[i] / union[i] / 10
    return metrics


def test(model, test_loader, use_valid_masks, device, save_to, to_train, strip):
    model.eval()
    if not to_train:
        tta_model = tta.SegmentationTTAWrapper(model, tta_transform(), merge_mode='max')
    else:
        tta_model = model
    with torch.no_grad():
        for data, meta in tqdm(test_loader):
            data = crop_image(data, strip)
            data = data.to(device).float()
            output = tta_model(data)
            output = uncrop_image(output, strip)
            batch_size = output.shape[0]
            size = output.shape
            answer = np.zeros((size[0], 1, size[2], size[3]))
            output = nn.Sigmoid()(output)
            for i in range(batch_size):
                answer[i, 0, :, :] = np.argmax(output[i, :, :, :].cpu().numpy(), axis=0)
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


def clear_files():
    for i in range(0, 10):
        folder = f'test/{i}'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        folder = f'train/{i}'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def main():
    args = parse_args()
    config_path = args.config_file_path
    clear_files()
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

    val_dataset = make_dataset(config.val.dataset)
    val_sampler = make_sampler(config.val.loader.params.sampler, config.val.dataset)

    val_loader = make_data_loader(config.val.loader, val_dataset, val_sampler)
    test_dataset = make_dataset(config.test.dataset)
    test_loader = make_data_loader(config.test.loader, test_dataset)

    device = torch.device(config.device)
    model = make_model(config.model).to(device)
    if config.parallel:
        model = nn.DataParallel(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", pytorch_total_params)

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
    if config.optim.params.get('lr', False) and len(config.optim.params.lr) != 1:
        step = config.epochs / (len(config.optim.params.lr))
    else:
        step = config.epochs * 2
    for epoch in range(1, config.epochs + 1):
        train_dataset = make_dataset(config.train.dataset, epoch)
        train_sampler = make_sampler(config.train.loader.params.sampler, config.train.dataset)
        train_loader = make_data_loader(config.train.loader, train_dataset, train_sampler)
        print(f"Epoch {epoch}", int(epoch // step), step, epoch)
        optimizer = make_optimizer(config.optim, model.parameters(), epoch, step)
        if config.to_train:
            train_metrics = train(model, optimizer, train_loader, loss_f, metrics, use_valid_masks_train, device,
                                  config.model.params.out_channels)
            write_metrics(epoch, train_metrics, train_writer)
            print_metrics('Train', train_metrics)
        val_metrics = val(model, val_loader, loss_f, metrics, use_valid_masks_val, device,
                          config.to_train, config.val.strip, config.model.params.out_channels)
        loss = 1 - val_metrics['average_iou']
        write_metrics(epoch, val_metrics, val_writer)
        print_metrics('Val', val_metrics)
        early_stopping(val_metrics['loss'])
        if not config.to_train:
            test(model, test_loader, True, device, save_to, config.to_train, config.test.strip)
        torch.save(model.state_dict(), config.model.last_checkpoint_path)
        if not config.to_train:
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
