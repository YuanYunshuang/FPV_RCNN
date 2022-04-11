import numpy as np

from utils.logger import LogMeter
from pathlib import Path
import wandb, shutil
from utils.train_utils import *
from models.utils import load_model_dict
import torch, random, numpy
from glob import glob


def train(cfgs):
    # set random seeds
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)

    dcfg, mcfg, cfg = cfgs
    cfg.TRAIN['pc_range'] = dcfg.pc_range
    # make checkpoints and log path
    log_path = Path(cfg.PATHS['run'])
    log_path.mkdir(exist_ok=True)
    # copy cfg file to log path
    cur_dir = os.path.abspath(__file__).rsplit('/', 2)[0]
    cfg_file = mcfg.name + '_' + dcfg.name + '.py'
    shutil.copy(os.path.join(cur_dir, 'cfg', cfg_file),
                os.path.join(cfg.PATHS['run'], cfg_file))

    train_dataloader = build_dataloader(dcfg, cfg)

    # get logger
    total_iterations = len(train_dataloader.dataset) // cfg.TRAIN['batch_size']
    logger = LogMeter(total_iterations, log_path, log_every=cfg.TRAIN['log_every'],
                      wandb_project=cfg.TRAIN['project_name'])

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = build_model(mcfg, cfg, dcfg).to(device)
    optimizer, lr_scheduler = build_optmizer(model, cfg)

    num_epochs = cfg.TRAIN['total_epochs']

    # load checkpoint if resume training
    if cfg.TRAIN['resume']:
        if cfg.TRAIN['epoch']==0:
            if os.path.exists(str(log_path).rsplit('/', 1)[0] + '/cia_ssd.pth'):
                filename = str(log_path).rsplit('/', 1)[0] + '/cia_ssd.pth'
            elif os.path.exists(str(log_path).rsplit('/', 1)[0] + '/latest.pth'):
                filename = str(log_path).rsplit('/', 1)[0] + '/latest.pth'
            else:
                filenames = glob(str(log_path).rsplit('/', 1)[0] + '/*.pth')
                filename = filenames[0]

            pretrained_dict = torch.load(filename)
        else:
            pretrained_dict = torch.load(str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['epoch'])))
        model = load_model_dict(model, pretrained_dict)

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(cfg.TRAIN['epoch'], num_epochs):
            train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch + 1, cfg,
                            device, logger=logger)

            lr_scheduler.step()

            # save checkpoint
            if (epoch + 1) % cfg.TRAIN['save_ckpt_every']==0:
                torch.save(model.state_dict(), str(log_path / 'epoch{:03}.pth'.format((epoch + 1))))
            else:
                torch.save(model.state_dict(), str(log_path / 'latest.pth'.format((epoch + 1))))
    print("Train finished.")
    if cfg.TRAIN['project_name'] is not None:
        wandb.finish()


def train_one_epoch(train_dataloader, model,optimizer, lr_scheduler, epoch, cfg, device, logger=None):
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    iteration = 1
    len_data = len(train_dataloader)
    model.train()
    for batch_data in train_dataloader:
        # if iteration==63:
        #     print("debug")
        # points = batch_data['points']
        # cloud_fused = batch_data.pop('points_fused')
        #points = points[points[: ,0]==0, 1:4]
        # bev = batch_data['bev_input'][0]
        gt_boxes = batch_data['gt_boxes'][0]
        if len(batch_data['gt_boxes_fused'])==0 or np.array([len(boxes)<=1 for boxes in batch_data['gt_boxes']]).any():
            continue
        load_data_to_device(batch_data)

        optimizer.zero_grad()
        # Forward pass
        batch_data = model(batch_data)

        if (len_data * (epoch-1) + iteration) % 40== 0 and cfg.TRAIN['project_name'] is not None:
            with torch.no_grad():
                if epoch>2 and iteration % 40 == 0:
                    wandb.log({'test': wandb.Image(np.zeros([100, 100]))})
                    predictions_dicts = model.post_processing(batch_data, cfg.TEST)
                    n_boxes = len(predictions_dicts[0]['box_lidar'])
                    pred_boxes = predictions_dicts[0]['box_lidar'] if n_boxes>0 else None
                    cloud_fused = torch.cat(batch_data.pop('points_fused'), dim=0)
                    cfg.TRAIN['visualization_func'](cloud_fused, pred_boxes, gt_boxes, cfg.TRAIN['pc_range'])
                    # cfg.TRAIN['visualization_func'](points, pred_boxes=None, gt_boxes=None,
                    #                                 pc_range=cfg.TRAIN['pc_range'])
                    # pass

        loss_dict = model.loss(batch_data)
        loss = loss_dict['loss']
        # Getting gradients
        loss.backward()

        # clip gradients
        max_norm = 0.1
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # Updating parameters
        optimizer.step()

        iteration += 1

        # Log training
        loss_dict = {k: v.item() for k, v in loss_dict.items() if v}
        if logger is not None:
            logger.log(epoch, iteration, lr_scheduler.get_last_lr()[0], **loss_dict)


if __name__=="__main__":
    cfgs = cfg_from_py('fusion_pvrcnn_comap')
    train((cfg() for cfg in cfgs))