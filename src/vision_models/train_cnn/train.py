"""
train.py

This script trains the CNN perceptual model (ResNet/ConvNeXt) on triplet data.
It builds a backbone + optional MLP head, prepares the chosen dataset
(NIGHTS / FashionTriplets / NIGHTS+Fashion), and logs metrics to W&B.

Main tasks:
- Parse config/CLI args and set seeds
- Construct datasets/dataloaders for train/val
- Initialize CNNPerceptualModel (frozen backbone, optional MLP head)
- Train with hinge loss on 2AFC triplets and log to WandB/Lightning
- Checkpoint best model by val_loss and early-stop on plateau
"""

import configargparse
import logging
import os
import time
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vision_models.train_cnn.model import create_backbone_cnn
from vision_models.dreamsim.util.train_utils import Mean, HingeLoss, seed_worker


def parse_args():
  """
  Build and parse command-line/config arguments for CNN training.

  Returns:
    argparse.Namespace with run options (logging/W&B), model options
    (backbone type, MLP, hidden size, normalization), dataset options
    (root paths, dataset name, workers), and training options
    (lr, weight decay, batch size, epochs, margin, early stopping).
  """
  parser = configargparse.ArgumentParser()
  parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

  ## Run options
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--tag', type=str, default='', help='tag for experiments (ex. experiment name)')
  parser.add_argument('--project_root', type=str, default='/content/drive/MyDrive/perceptual-vits-fashion-forecasting')
  parser.add_argument('--log_dir', type=str, default='/vision_models/cnn_training', help='path to save model checkpoints and logs')
  parser.add_argument('--wandb_project', type=str, default='vision_model_training', help='Weights & Biases project name')
  parser.add_argument('--vision_model_training_name', type=str, default='resnet18_run', help='Human-friendly name for grouping/naming W&B runs')
  parser.add_argument('--wandb_notes', type=str, default='', help='Optional W&B notes for the run')

  ## Model options
  parser.add_argument('--model_type', type=str, default='resnet50', help='Which CNN model to finetune.'
                      'list of models. Accepted models: [resnet18, resnet50, convnext_tiny]')
  parser.add_argument('--mlp', action='store_true', help='Enable MLP finetuning.')
  parser.add_argument('--hidden_size', type=int, default=512, help='Size of the MLP hidden layer.')
  parser.add_argument('--normalize_embeds', action='store_true', help='Normalizes the embeddings')
  parser.add_argument('--load_size', type=int, default=224, help='Height and Width of the images')

  ## Dataset options
  parser.add_argument('--dataset_root', type=str, default='/content/datasets/nights', help='path to training dataset.')
  parser.add_argument('--second_dataset_root', type=str, default=None, help='path for training on both nights and fashion triplets')
  parser.add_argument('--dataset_name', type=str, default='nights',
                      choices=['nights', 'fashion_triplets', 'nights_fashion_triplets',
                      'synthetic_fashion_1', 'synthetic_fashion_2'],
                      help='Which dataset class to use for training/validation.')
  parser.add_argument('--num_workers', type=int, default=2)

  ## Training options
  parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for training.')
  parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for training.')
  parser.add_argument('--batch_size', type=int, default=16, help='Dataset batch size.')
  parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
  parser.add_argument('--margin', default=0.05, type=float, help='Margin for hinge loss')
  parser.add_argument('--patience', default=3, type=int, help='Early stopping patience')
  parser.add_argument('--min_delta', default=0.0, type=float, help='Early stopping min delta')

  # Safety measures
  parser.add_argument('--auto_save', type=int, default=0, help='Save extra checkpoints every N epochs (0=off).')
  parser.add_argument('--load_path', type=str, default=None, help='Path to a Lightning .ckpt to initialize CNN weights from.')

  return parser.parse_args()

class LightningCNNModel(pl.LightningModule):
  def __init__(
      self,
      model_type: str,
      mlp: bool,
      hidden_size: int,
      normalize_embeds: bool,
      lr: float,
      weight_decay: float,
      margin: float,
      train_data_len: int,
      project_root: str,
      device: str,
      **kwargs
  ):
    super().__init__()
    self.save_hyperparameters()

    self.model_type = model_type
    self.mlp = mlp
    self.hidden_size = hidden_size
    self.normalize_embeds = normalize_embeds
    self.lr = lr
    self.weight_decay = weight_decay
    self.margin = margin
    self.train_data_len = train_data_len
    self._target_device = torch.device(device)

    self.val_metrics = {'loss': Mean().to(device), 'score': Mean().to(device)}
    self.__reset_val_metrics()

    self.cnn_model, _ = create_backbone_cnn(
      model_type=self.model_type,
      hidden_size=self.hidden_size,
      mlp=self.mlp,
      normalize_embeds=self.normalize_embeds,
      device=device,
      project_root=project_root,
    )
    self.__prep_linear_model()

    pytorch_total_params = sum(p.numel() for p in self.cnn_model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in self.cnn_model.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params} | Trainable params: {pytorch_total_trainable_params} "
          f"| % Trainable: {pytorch_total_trainable_params / pytorch_total_params * 100}")

    self.criterion = HingeLoss(margin=self.margin, device=device)

    self.epoch_loss_train = 0.0
    self.train_num_correct = 0.0

    # track best epoch by both metrics so run() can choose later
    self.best_val_loss = float('inf')
    self.best_loss_epoch = -1
    self.best_val_acc = float('-inf')
    self.best_acc_epoch = -1

  def forward(self, img_ref, img_0, img_1):
    # Get raw embeddings once
    ref   = self.cnn_model.embed(img_ref)
    left  = self.cnn_model.embed(img_0)
    right = self.cnn_model.embed(img_1)

    dist_0 = 1 - F.cosine_similarity(ref,  left,  dim=-1)
    dist_1 = 1 - F.cosine_similarity(ref,  right, dim=-1)
    return dist_0, dist_1

  def training_step(self, batch, batch_idx):
    img_ref, img_0, img_1, target, idx = batch
    dist_0, dist_1 = self.forward(img_ref, img_0, img_1)

    logit = dist_0 - dist_1
    loss = self.criterion(logit.view(-1), target.view(-1))
    loss /= target.shape[0]

    self.epoch_loss_train  += float(loss.detach())
    correct = ((target >= 0.5) == (dist_1 < dist_0)).sum()
    self.train_num_correct += int(correct.detach().item())
    return loss

  def validation_step(self, batch, batch_idx):
    img_ref, img_0, img_1, target, idx = batch
    dist_0, dist_1 = self.forward(img_ref, img_0, img_1)

    decisions = torch.lt(dist_1, dist_0)
    logit = dist_0 - dist_1
    loss = self.criterion(logit.view(-1), target.view(-1))

    val_num_correct = ((target >= 0.5) == decisions).sum()

    bs       = target.shape[0]
    loss_det = loss.detach()
    ok_det   = val_num_correct.detach()
    denom    = torch.tensor(bs, device=loss_det.device)

    self.val_metrics['loss'].update(loss_det, denom)
    self.val_metrics['score'].update(ok_det,   denom)
    return loss

  def on_train_epoch_start(self):
    self.epoch_loss_train = 0.0
    self.train_num_correct = 0.0
    self._epoch_wall_start = time.time()
    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()

  def on_train_epoch_end(self):
    num_batches = max(1, self.trainer.num_training_batches)
    train_loss = self.epoch_loss_train / num_batches
    train_2afc_acc = self.train_num_correct / self.train_data_len

    epoch_sec = time.time() - getattr(self, '_epoch_wall_start', time.time())
    max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else self.lr

    self.log('train_loss', train_loss, prog_bar=True, on_epoch=True, logger=True)
    self.log('train_2afc_acc', train_2afc_acc, prog_bar=True, on_epoch=True, logger=True)
    self.log('time/epoch_sec', epoch_sec, on_epoch=True, logger=True)
    self.log('system/max_gpu_mem_mb', max_mem_mb, on_epoch=True, logger=True)
    self.log('optim/lr', lr, on_epoch=True, logger=True)

  def on_validation_start(self):
    self.cnn_model.backbone.eval()
    self.__reset_val_metrics()

  def on_validation_epoch_end(self):
    loss = self.val_metrics['loss'].compute()
    score = self.val_metrics['score'].compute()

    loss_float  = float(loss.detach().cpu().item())
    score_float = float(score.detach().cpu().item())
    self.log('val_loss', loss_float, prog_bar=True, on_epoch=True, logger=True)
    self.log('val_2afc_acc', score_float, prog_bar=True, on_epoch=True, logger=True)

    try:
      if loss_float < self.best_val_loss:
        self.best_val_loss  = loss_float
        self.best_loss_epoch = int(self.current_epoch)
        # write tiny summary file in exp_dir
        self.__write_best_summary(loss_float)
    except Exception:
      pass

    try:
      if score_float > self.best_val_acc:
        self.best_val_acc  = score_float
        self.best_acc_epoch = int(self.current_epoch)
    except Exception:
      pass
    self.log('epoch', int(self.current_epoch)+1, prog_bar=False, on_epoch=True, logger=True)

  def configure_optimizers(self):
    params = [p for p in self.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
      params, lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay
    )
    return [optimizer]

  def __reset_val_metrics(self):
    for k, v in self.val_metrics.items():
      v.reset()

  def __prep_linear_model(self):
    self.cnn_model.backbone.requires_grad_(False)
    self.cnn_model.head.requires_grad_(self.mlp)

  def on_after_backward(self):
    # Log a global grad-norm every N optimizer steps
    N = 100
    if (self.global_step % N) != 0:
      return

    total_sq = None
    for p in self.parameters():
      if p.grad is not None:
        v = p.grad.norm(2).pow(2)
        total_sq = v if total_sq is None else total_sq + v

    global_norm = float(total_sq.sqrt().item()) if total_sq is not None else 0.0
    self.log('grads/global_norm', global_norm, prog_bar=False, on_step=True, on_epoch=False, logger=True)

  def __write_best_summary(self, loss_value: float):
    try:
      base_dir = getattr(self.trainer, 'default_root_dir', os.getcwd())
      summary_path = os.path.join(base_dir, 'best_val_loss.txt')
      with open(summary_path, 'w') as f:
        f.write(f"best_epoch: {int(self.current_epoch)+1}\n")
        f.write(f"best_val_loss: {loss_value:.6f}\n")
    except Exception:
      pass


class SaveEveryNEpochs(pl.Callback):
  def __init__(self, n: int, out_dir: str):
    super().__init__()
    self.n = int(n)
    self.out_dir = out_dir
    os.makedirs(self.out_dir, exist_ok=True)

  def on_train_epoch_end(self, trainer, pl_module):
    # Save at epochs N, 2N, 3N, ...
    if self.n and ((trainer.current_epoch + 1) % self.n == 0):
      fname = f"epoch_{trainer.current_epoch + 1}_run.ckpt"
      path = os.path.join(self.out_dir, fname)
      trainer.save_checkpoint(path)

  # Optional: dump an emergency checkpoint if something crashes mid-run
  def on_exception(self, trainer, pl_module, err):
    try:
      trainer.save_checkpoint(os.path.join(self.out_dir, "_emergency_last.ckpt"))
    except Exception:
      pass


def run(args):
  """
  Orchestrate one training run.

  Steps:
    1) Create experiment directory and save run_config.yaml
    2) Seed RNGs and set DataLoader worker generator
    3) Select dataset class from args.dataset_name and build train/val loaders
    4) Configure W&B logger (group/tags/metrics) and PL callbacks
    5) Instantiate LightningCNNModel (backbone+MLP head) on the target device
    6) Launch PyTorch Lightning Trainer.fit() and log epoch metrics

  Args:
    args: parsed arguments from parse_args()
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  tag = args.tag if len(args.tag) > 0 else ""
  training_method = 'mlp' if args.mlp else 'vanilla'
  os.makedirs(os.path.join(args.project_root, 'vision_models', 'cnn_backbone_models'), exist_ok=True)

  # Build a readable run stub
  run_stub = (
    f'{tag}_{str(args.model_type)}_{training_method}_{args.dataset_name}'
  )

  base_dir = os.path.join(args.project_root, args.log_dir.lstrip('/'))
  os.makedirs(base_dir, exist_ok=True)
  exp_dir = os.path.join(base_dir, run_stub)
  os.makedirs(exp_dir, exist_ok=True)

  # Save configs
  cfg_path = os.path.join(exp_dir, 'run_config.yaml')
  with open(cfg_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

  # Seed and perf knobs
  seed_everything(args.seed)
  torch.backends.cudnn.benchmark = True
  g = torch.Generator()
  g.manual_seed(args.seed)

  # Pick dataset class
  if args.dataset_name == 'nights':
    from vision_models.train_cnn.dataset import NightsDataset as DatasetCls
  elif args.dataset_name == 'fashion_triplets':
    from vision_models.train_cnn.dataset import FashionTripletsDataset as DatasetCls
  elif args.dataset_name == 'nights_fashion_triplets':
    from vision_models.train_cnn.dataset import NightsFashionTripletsDataset as DatasetCls
  elif args.dataset_name == 'synthetic_fashion_1':
    from vision_models.train_cnn.dataset import SyntheticFashionDataset1 as DatasetCls
  elif args.dataset_name == 'synthetic_fashion_2':
    from vision_models.train_cnn.dataset import SyntheticFashionDataset2 as DatasetCls
  else:
    raise ValueError(f"Unknown dataset_name: {args.dataset_name}")

  first_root_dir  = args.dataset_root
  second_root_dir = args.second_dataset_root

  # Prepare Datasets
  train_dataset = DatasetCls(
    root_dir=first_root_dir,
    random_seed=args.seed,
    second_root_dir=second_root_dir,
    split='train'
  )
  val_dataset = DatasetCls(
    root_dir=first_root_dir,
    random_seed=args.seed,
    second_root_dir=second_root_dir,
    split='val'
  )

  # DataLoaders
  pin_mem = torch.cuda.is_available()
  use_workers = args.num_workers > 0
  common_dl = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    worker_init_fn=seed_worker,
    generator=g,
  )
  if use_workers:
    common_dl.update(pin_memory=pin_mem, persistent_workers=True, prefetch_factor=2)
  else:
    common_dl.update(pin_memory=False)

  train_loader = DataLoader(train_dataset, shuffle=True, **common_dl)
  val_loader = DataLoader(val_dataset, shuffle=False, **common_dl)

  # Console logging only
  for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
  logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
  logging.info('Arguments: %s', vars(args))
  logging.info(f"[RUN] exp_dir = {exp_dir}")

  # W&B logger
  run_name = f"{args.vision_model_training_name}_{args.model_type}_{training_method}_{args.dataset_name}"
  logger = WandbLogger(
    save_dir=exp_dir,
    project=args.wandb_project,
    name=run_name,
    group=args.vision_model_training_name,
    job_type='train cnn',
    tags=[args.vision_model_training_name, 'train cnn', args.model_type, training_method, args.dataset_name],
    notes=args.wandb_notes,
    log_model=False
  )

  # Make W&B use 'epoch' as the x-axis for epoch-level metrics
  logger.experiment.define_metric('epoch')  # declare step metric
  for m in [
    'train_loss', 'train_2afc_acc',
    'val_loss', 'val_2afc_acc',
    'time/epoch_sec', 'system/max_gpu_mem_mb', 'optim/lr'
  ]:
    logger.experiment.define_metric(m, step_metric='epoch')

  # Trainer
  ckpt = ModelCheckpoint(
      dirpath=exp_dir,
      filename='best',
      monitor='val_loss',
      mode='min',
      save_top_k=1,
      save_last=True,  # also writes last.ckpt
  )

  early_stop = EarlyStopping(
      monitor='val_loss',
      mode='min',
      patience=args.patience,
      min_delta=args.min_delta,
      check_on_train_epoch_end=False
  )

  callbacks = [ckpt, early_stop]

  if args.auto_save and args.auto_save > 0:
    logging.info(f"Periodic checkpointing every {args.auto_save} epochs → {exp_dir}")
    callbacks.append(SaveEveryNEpochs(args.auto_save, exp_dir))

  accel = 'gpu' if torch.cuda.is_available() else 'cpu'
  precision = '16-mixed' if torch.cuda.is_available() else '32-true'

  trainer = Trainer(
    devices=1,
    accelerator=accel,
    precision=precision,
    log_every_n_steps=10,
    logger=logger,
    max_epochs=args.epochs,
    default_root_dir=exp_dir,
    callbacks=callbacks,
    num_sanity_val_steps=0
  )
  root_dev = trainer.strategy.root_device
  logging.info(f"[PL] accelerator={accel} precision={precision} root_device={root_dev} | cuda={torch.cuda.is_available()} count={torch.cuda.device_count()}")
  if torch.cuda.is_available():
    print(f"[CUDA] {torch.cuda.get_device_name(0)}")

  # Model and train
  model = LightningCNNModel(
    device=device, train_data_len=len(train_dataset), **vars(args)
  )

  # load CNN weights from a Lightning .ckpt (if provided)
  if args.load_path is not None:
    if not os.path.exists(args.load_path):
      raise FileNotFoundError(f"--load_path not found: {args.load_path}")
    logging.info(f"[Warm start] Loading CNN weights from {args.load_path}")
    ckpt_blob = torch.load(args.load_path, map_location=device)
    sd = ckpt_blob.get('state_dict', {})
    # Pull only the CNN submodule weights (prefixed with 'cnn_model.')
    stripped = {
      k.replace('cnn_model.', '', 1): v
      for k, v in sd.items() if k.startswith('cnn_model.')
    }
    missing, unexpected = model.cnn_model.load_state_dict(stripped, strict=False)
    if missing:
      logging.warning(f"[Warm start] Missing keys (showing up to 8): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
      logging.warning(f"[Warm start] Unexpected keys (up to 8): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")


  # Compute once
  total_params = sum(p.numel() for p in model.cnn_model.parameters())
  trainable_params = sum(p.numel() for p in model.cnn_model.parameters() if p.requires_grad)

  # Build one dict and update config in a single call
  cfg = dict(vars(args))
  cfg.update({
    'params_total': total_params,
    'params_trainable': trainable_params,
    'params_percent_trainable': 100.0 * trainable_params / max(1, total_params),
  })
  logger.experiment.config.update(cfg, allow_val_change=True)

  logging.info('Training')
  trainer.fit(model, train_loader, val_loader)
  print('Done :)')
