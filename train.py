import torch
from torchvision.datasets import UCF101
import wandb
import logging
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate
from models.final_model import BARNET
from utils.utils import build_transforms
import os
import sys


class Logger():
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.log.write(message)

    def flush(self):
        pass


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def train_net(net,
              device,
              epochs: int = 100,
              batch_size: int = 96,
              learning_rate: float = 1e-5,
              val_percent: float = 0.2,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # video_path = r'../UCF101/UCF-101'
    video_path = r'./UCF-101'
    annotation_path = r'./ucfTrainTestlist'
    dir_checkpoint = r'./result'
    frames_per_clip = 27
    # 1. Create dataset
    try:
        dataset = UCF101(root=video_path, annotation_path=annotation_path, frames_per_clip=frames_per_clip, transform=build_transforms())
        print(1)
    except (AssertionError, RuntimeError):
        print('Error: failed to load the dataset')
        pass
    # 2. Split into train / validation partitions
    print(2)
    train_size = int(len(dataset) * 0.3)
    val_size = int(len(dataset) * 0.1)
    empty_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size, empty_size])
    # 3. Create data loaders
    print(3)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=custom_collate,
                                                   num_workers=0,
                                                   pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 collate_fn=custom_collate,
                                                 num_workers=0,
                                                 pin_memory=True)


    # (Initialize logging)
    # experiment = wandb.init(project='BARNet', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))
    #
    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}
    #     Batch size:      {batch_size}
    #     Learning rate:   {learning_rate}
    #     Training size:   {train_size}
    #     Validation size: {val_size}
    #     Checkpoints:     {save_checkpoint}
    #     Device:          {device.type}
    #     Images scaling:  {img_scale}
    #     Mixed Precision: {amp}
    # ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    print(4)
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    print(5)
    sys.stdout_1 = Logger('loss.txt')
    sys.stdout_2 = Logger('acc.txt')
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=train_size, desc=f'Epoch {epoch}/{epochs}', unit='video') as pbar:
            for i, (video, label) in enumerate(train_loader):
                # assert video.shape == net.input_shape, \
                #     f'Network has been defined with {net.input_shape} input shapes, ' \
                #     f'but loaded video have {video.shape} shapes. Please check that ' \
                #     'the videos are loaded correctly.'

                video = video.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    pred = net(video)
                    loss = criterion(pred, label)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(video.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                if i % 100 == 0:
                    print('{}_loss:'.format(i))
                    print("{}\n".format(loss.item()))
                    torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'train_{}.pth'.format(i)))

            sys.stdout_1.write('{}\n'.format(epoch_loss))
            print("loss%f\n" % epoch_loss)
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                # pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (train_size // (10 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
                #         histograms = {}
                #         for tag, value in net.named_parameters():
                #             tag = tag.replace('/', '.')
                #             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                #             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_acc, val_loss = evaluate(net, val_loader, device)
            sys.stdout_2.write('{}\n'.format(val_acc))
            print("acc%f\n" % val_acc)
            scheduler.step(val_acc)

                        # logging.info('Validation Dice score: {}'.format(val_acc))
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation accuracy': val_acc,
                        #     'validation loss': val_loss.item(),
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })

        if save_checkpoint:
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BARNET().to(device=device)
    train_net(net, device=device)


