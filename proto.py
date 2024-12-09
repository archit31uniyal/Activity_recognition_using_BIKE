from datasets.video_dataset import PoseDataset
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from utils.logger import setup_logger
import torch.distributed as dist
from datetime import datetime

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Define the layers for the residual block
        self.norm1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size // 2)
        
        self.norm2 = nn.LayerNorm(input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, output_size)
        
        self.fc3 = nn.Linear(input_size, output_size)

        self.act = nn.ELU()

    def forward(self, x):
        # Forward pass of the residual block
        x = self.act(self.norm1(x))
        skip = self.fc3(x)
        
        x = self.act(self.norm2(self.fc1(x)))
        x = self.fc2(x)
        
        return x + skip

class PoseDetector(nn.Module):
    def __init__(self, num_frames, num_points, num_coor, output_size, num_layers=1, num_blocks=1):
        super(PoseDetector, self).__init__()
        seq_len = num_points*num_coor
        self.num_layers = num_layers
        # flatten
        # print(seq_len, seq_len*4)
        self.input_mlp = nn.Sequential(
            nn.Linear(seq_len, 4*seq_len),
            nn.ReLU(),
            nn.Linear(4*seq_len, 128))
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=128, 
            num_layers=self.num_layers, 
            batch_first=True)

        blocks = [ResBlockMLP(128, 128) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        self.fc_out = nn.Sequential(
            nn.Linear((num_frames*128), (num_frames*128)),
            nn.Linear((num_frames*128), (num_frames*128)//2),
            nn.Linear((num_frames*128)//2, output_size)
        )
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # https://www.youtube.com/watch?v=lyUT6dOARGs&ab_channel=LukeDitria
    def forward(self, x, lengths):
        # Sort sequences by length in descending order
        lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]

        # want it to be 32x(dimension size)
        # print(f'input     {x.shape}', flush=True)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print(f'reshape   {x.shape}', flush=True)
        x = self.input_mlp(x)
        # print(f'input_mlp {x.shape}', flush=True)
        # Pack the padded sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        # Pass through LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)
        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Restore the original order
        _, original_idx = sorted_idx.sort(0)
        output = output[original_idx]
        hn = hn[:, original_idx, :]
        cn = cn[:, original_idx, :]

        # print(f'lstm      {x.shape}', flush=True)
        x = self.act(self.res_blocks(x)).squeeze(0)
        # flatten
        # print(f'resblock  {x.shape}', flush=True)
        x = x.reshape(x.shape[0], -1)
        # print(f'reshape   {x.shape}', flush=True)
        x = self.fc_out(x)
        # print(f'fc        {x.shape}', flush=True)
        return self.softmax(x)

def train(pd, train_loader, optimizer, criterion, epoch, device, logger, batch_size=32):
  cnt = 0
  training_loss_logger = []
  loss = 0
  hidden_size = 128
  for i, (video, pose, class_id, sample_fname) in enumerate(train_loader):
    # curr_len = batch_size
    # if pose.shape[0] < batch_size:
    #   curr_len = pose.shape[0]
    #   pose = pose.repeat(2, 1, 1, 1)
    #   pose = pose[:batch_size, :, :, : ]

    # if class_id.shape[0] < batch_size:
    #   curr_len = pose.shape[0]
    #   class_id = class_id.repeat(2)[:batch_size]
    # hidden = torch.zeros(1, pose.shape[0], hidden_size, device=device)
    # mem = torch.zeros(1, pose.shape[0], hidden_size, device=device)

    pose = pose.type(torch.FloatTensor)

    # move to device
    pose = pose.to(device)
    class_id = class_id.to(device)

    lengths = torch.zeros(batch_size) + 1
    lengths = lengths.to(device)
    out = pd(pose, lengths)

    gt = class_id.argmax(axis=1)
    loss = criterion(out, gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info(f'Epoch [{epoch}]: iteration {i+1}: loss {loss.item()}')
    training_loss_logger.append(loss.item())
    
    cnt += 1
    if cnt > 10:
      break

  return training_loss_logger

def main():
    working_dir = os.path.join('./', 'wlasl', 'new', '001')

    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          name=f'BIKE')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    # pp = pprint.PrettyPrinter(indent=4)
    # logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))

    num_classes = 10
    batch_size = 8
    num_segments = 16
    modality = 'video'
    num_epoch = 40
    freq = 4

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_ids=[0]
        gpu_ids = list(map(int, device_ids))
        cuda='cuda:'+ str(gpu_ids[0])
        # cudnn.benchmark = True

    device= torch.device(cuda if use_cuda else 'cpu')

    TIME = 0
    HEIGHT = 1
    WIDTH = 2
    CHANNEL = 3
    tfs = transforms.Compose([
        # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
        # scale in [0, 1] of type float
        transforms.Lambda(lambda x: x / 255.),
        # reshape into (C, W, T, H) for easier convolutions
        transforms.Lambda(lambda x: x.permute(TIME,CHANNEL,  HEIGHT, WIDTH)),
        # rescale to the most common size
        transforms.Lambda(lambda x: nn.functional.interpolate(x, (256, 320))),
    ])

    train_data = PoseDataset(
        '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/resized_wlasl_10/', 
        './lists/wlasl/',
        num_segments=num_segments,
        frame_rate=15,
        modality=modality,
        transform=tfs,
        train=True)

    val_data = PoseDataset(
        '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/resized_wlasl_10/', 
        './lists/wlasl/',
        num_segments=num_segments,
        frame_rate=15,
        modality=modality,
        transform=tfs,
        train=False)
    logger.info(f'Train Size')

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        drop_last=True, 
        shuffle=True)         
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        drop_last=True, 
        shuffle=True)

    model = PoseDetector(16, 21, 3, num_classes, num_layers=5, num_blocks=1)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # import pdb; pdb.set_trace()
    total_loss = []
    for epoch in range(num_epoch):
        # Train the model
        model.train()
        loss = train(model, train_loader, optimizer, criterion, epoch, device, logger, batch_size=batch_size)
        logger.info(f'Epoch {epoch}: total {np.mean(loss)}')
        total_loss.append(loss)

        if (epoch+1) >= 10 and (epoch+1) % freq == 0:
            model.eval()
            with torch.no_grad():
                num_samples = 0
                num_correct = 0
                logger.info('Starting validation....')
                for i, (video, pose, class_id, sample_fname) in enumerate(val_loader):
                    
                    pose = pose.type(torch.FloatTensor)

                    # move to device
                    pose = pose.to(device)
                    class_id = class_id.to(device)
                    lengths = torch.zeros(batch_size) + 1
                    lengths = lengths.to(device)

                    out = model(pose, lengths)

                    # logger.debug(f'{out.dtype}, {out.shape}, {out}')
                    one_hot = out.argmax(axis=1)
                    gt = class_id.argmax(axis=1)
                    # logger.debug(f'{one_hot.dtype}, {one_hot.shape}, {one_hot}')
                    # logger.debug(f'{gt.dtype}, {gt.shape}, {gt}')

                    # logger.info(f'{one_hot.shape} {one_hot}')
                    # logger.info(f'{gt.shape} {gt}')
                    num_correct += int(torch.sum(gt == one_hot))
                    num_samples += batch_size
                    logger.info(f'Current Accuracy: {num_correct/num_samples}%')
                
                logger.info(f'Accuracy: {num_correct/num_samples}%')
                    

            # Save the model
            current_date = datetime.now()
            formatted_date = current_date.strftime('%Y-%m-%d')
            filename = f"mode_{formatted_date}.txt"
            torch.save(model.state_dict(), working_dir + '/' + filename)

if __name__ == '__main__':
    main()