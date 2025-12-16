"""
train.py (DreamSim ViT)

Finetunes a DreamSim-style ViT perceptual model on 2AFC triplets using
PyTorch Lightning. Supports two modes:
- LoRA: freeze the backbones and train injected LoRA adapters
- MLP:  freeze the backbones and train a small residual MLP head

Main tasks:
- Parse CLI/config file (via -c/--config) and set seeds
- Build train/val datasets and DataLoaders (NIGHTS / FashionTriplets / mixed / synthetic)
- Initialize PerceptualModel (ViT/CLIP/MAE variants; optional ensemble)
- Train with hinge loss on (ref, left, right) triplets and log metrics to W&B
- Checkpoint best model by val_loss and early-stop on plateau

Outputs:
- Run folder: <project_root>/<log_dir>/<tag>_<model_type>_<mode>_<feat>_<stride>_<dataset>[_lorar]
- If use_lora and save_mode != 'entire_model':
    adapters under exp_dir/lora_adapters/epoch_<N>/, plus best/ and last/
- If save_mode != 'adapter_only':
    Lightning checkpoints best.ckpt and last.ckpt in exp_dir
- Small summary file best_val_loss.txt with the best epoch/loss
"""

import logging
import yaml
import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from vision_models.dreamsim.util.train_utils import Mean, HingeLoss, seed_worker
from vision_models.dreamsim.util.utils import get_preprocess
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from peft import get_peft_model, LoraConfig, PeftModel
from vision_models.dreamsim.dreamsim.model import PerceptualModel
import os
import configargparse


def parse_args():
    """
    Build and parse command-line/config arguments for DreamSim ViT training.

    Groups:
      Run options:
        seed, tag, project_root, log_dir, wandb settings, save_mode
      Model options:
        model_type (single or comma-separated ensemble),
        feat_type (cls/embedding/last_layer/cls_patch),
        stride (single or comma-separated),
        use_lora (LoRA mode), hidden_size (MLP), normalize_embeds, load_size
      Dataset options:
        dataset_root, second_dataset_root (for mixed datasets),
        dataset_name, num_workers
      Training options:
        lr, weight_decay, batch_size, epochs, margin (hinge), patience, min_delta
      LoRA options:
        lora_r, lora_alpha, lora_dropout

    Returns:
      argparse.Namespace with all runtime options (supports -c/--config file).
    """
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    ## Run options
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--tag', type=str, default='', help='tag for experiments (ex. experiment name)')
    parser.add_argument('--project_root', type=str, default="/content/drive/MyDrive/perceptual-vits-fashion-forecasting")
    parser.add_argument('--log_dir', type=str, default="/vision_models/vits_training", help='path to save model checkpoints and logs')
    parser.add_argument('--save_mode', type=str, default="all", help='whether to save only LoRA adapter weights, '
                                                                     'entire model, or both. Accepted '
                                                                     'options: [adapter_only, entire_model, all]')
    parser.add_argument('--wandb_project', type=str, default='vision_model_training', help='Weights & Biases project name')
    parser.add_argument('--vision_model_training_name', type=str, default='dino_run', help='Human-friendly name for grouping/naming W&B runs')
    parser.add_argument('--wandb_notes', type=str, default='', help='Optional W&B notes for the run')

    ## Model options
    parser.add_argument('--model_type', type=str, default='dino_vitb16',
                        help='Which ViT model to finetune. To finetune an ensemble of models, pass a comma-separated'
                             'list of models. Accepted models: [dino_vits8, dino_vits16, dino_vitb8, dino_vitb16, '
                             'clip_vitb16, clip_vitb32, clip_vitl14, mae_vitb16, mae_vitl16, mae_vith14, '
                             'open_clip_vitb16, open_clip_vitb32, open_clip_vitl14]')
    parser.add_argument('--feat_type', type=str, default='cls',
                        help='What type of feature to extract from the model. If finetuning an ensemble, pass a '
                             'comma-separated list of features (same length as model_type). Accepted feature types: '
                             '[cls, embedding, last_layer, cls_patch].')
    parser.add_argument('--stride', type=str, default='16',
                        help='Stride of first convolution layer the model (should match patch size). If finetuning'
                             'an ensemble, pass a comma-separated list (same length as model_type).')
    parser.add_argument('--use_lora', action='store_true',
                        help='Enable LoRA finetuning (omit this flag to train the MLP head instead).')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of the MLP hidden layer.')
    parser.add_argument('--normalize_embeds', type=int, default=1, help='1 to normalize, 0 to not')
    parser.add_argument('--load_size', type=int, default=224, help='Height and Width of the images')

    ## Dataset options
    parser.add_argument('--dataset_root', type=str, default="/content/datasets/nights", help='path to training dataset.')
    parser.add_argument('--second_dataset_root', type=str, default=None, help='path for training on both nights and fashion triplets')
    parser.add_argument('--dataset_name', type=str, default="nights",
                        choices=["nights", "fashion_triplets", "nights_fashion_triplets",
                        "synthetic_fashion_1", "synthetic_fashion_2"],
                        help="Which dataset class to use for training/validation.")
    parser.add_argument('--num_workers', type=int, default=2)

    ## Training options
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Dataset batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--margin', default=0.05, type=float, help='Margin for hinge loss')
    parser.add_argument('--patience', default=3, type=int, help='Early stopping patience')
    parser.add_argument('--min_delta', default=0.0, type=float, help='Early stopping min delta')

    ## LoRA-specific options
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=float, default=16, help='Alpha for attention scaling')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')

    # Safety measures
    parser.add_argument('--auto_save', type=int, default=0,
                        help='Save extra checkpoints/adapters every N epochs (0=off).')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Warm start: if use_lora=False, path to a Lightning .ckpt; '
                             'if use_lora=True, path to a LoRA adapter dir (e.g., lora_adapters/best).')
    parser.add_argument('--load_lora_epoch', type=int, default=None,
                        help='Optional: if load_path points to lora_adapters/, pick a specific epoch_N subdir.')

    return parser.parse_args()


class LightningPerceptualModel(pl.LightningModule):
    def __init__(self,
                 feat_type: str = "cls",
                 model_type: str = "dino_vitb16",
                 stride: str = "16",
                 hidden_size: int = 512,
                 normalize_embeds: bool = True,
                 lr: float = 0.0003,
                 use_lora: bool = False,
                 margin: float = 0.05,
                 lora_r: int = 16,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.1,
                 weight_decay: float = 0.0001,
                 train_data_len: int = 1,
                 project_root: str = "/content/drive/MyDrive/perceptual-vits-fashion-forecasting",
                 device: str = "cuda",
                 save_mode: str = "all",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feat_type = feat_type
        self.model_type = model_type
        self.stride = stride
        self.hidden_size = hidden_size
        self.normalize_embeds = bool(normalize_embeds)
        self.lr = lr
        self.use_lora = use_lora
        self.margin = margin
        self.weight_decay = weight_decay
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.train_data_len = train_data_len
        self.save_mode = save_mode
        self._target_device = torch.device(device)

        self.__validate_save_mode()

        self.started = False
        self.val_metrics = {'loss': Mean().to(device), 'score': Mean().to(device)}
        self.__reset_val_metrics()

        self.perceptual_model = PerceptualModel(
                                              model_type=self.model_type,
                                              feat_type=self.feat_type,
                                              stride=self.stride,
                                              hidden_size=self.hidden_size,
                                              lora=self.use_lora,
                                              baseline=False,
                                              project_root=project_root,
                                              normalize_embeds = self.normalize_embeds,
                                              device=device
                                              )
        if self.use_lora:
            self.__prep_lora_model()
        else:
            self.__prep_linear_model()

        pytorch_total_params = sum(p.numel() for p in self.perceptual_model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in self.perceptual_model.parameters() if p.requires_grad)
        print(f'Total params: {pytorch_total_params} | Trainable params: {pytorch_total_trainable_params} '
              f'| % Trainable: {pytorch_total_trainable_params / pytorch_total_params * 100}')

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
        ref_raw   = self.perceptual_model.embed(img_ref)
        left_raw  = self.perceptual_model.embed(img_0)
        right_raw = self.perceptual_model.embed(img_1)

        # Apply the same post-processing as PerceptualModel.forward for cls_patch
        def finalize(e):
            if self.perceptual_model.feat_type_list[0] == 'cls_patch':
                cls = e[:, 0]        # [B, D]
                patches = e[:, 1:]   # [B, S^2, D]
                n = patches.shape[0]
                s = int(patches.shape[1] ** 0.5)
                pooled = F.adaptive_avg_pool2d(
                    patches.reshape(n, s, s, -1).permute(0, 3, 1, 2), (1, 1)
                ).squeeze()
                if pooled.dim() == 1:  # handle batch size = 1
                    pooled = pooled.unsqueeze(0)
                e = torch.cat((cls, pooled), dim=-1)  # [B, D + D]
            return e

        ref   = finalize(ref_raw)
        left  = finalize(left_raw)
        right = finalize(right_raw)

        dist_0 = 1 - F.cosine_similarity(ref,  left,  dim=-1)
        dist_1 = 1 - F.cosine_similarity(ref,  right, dim=-1)
        return dist_0, dist_1

    def training_step(self, batch, batch_idx):
        img_ref, img_0, img_1, target, idx = batch
        dist_0, dist_1 = self.forward(img_ref, img_0, img_1)
        decisions = torch.lt(dist_1, dist_0)
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
        self.started = True
        self._epoch_wall_start = time.time()
        if torch.cuda.is_available():
          torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self):
        if self.use_lora:
            self.__save_lora_weights()
        train_loss = self.epoch_loss_train / self.trainer.num_training_batches
        train_2afc_acc = self.train_num_correct / self.train_data_len
        epoch_sec = time.time() - getattr(self, "_epoch_wall_start", time.time())
        max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else self.lr
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_2afc_acc", train_2afc_acc, prog_bar=True, on_epoch=True, logger=True)
        self.log("time/epoch_sec", epoch_sec, on_epoch=True, logger=True)
        self.log("system/max_gpu_mem_mb", max_mem_mb, on_epoch=True, logger=True)
        self.log("optim/lr", lr, on_epoch=True, logger=True)

    def on_validation_start(self):
        for extractor in self.perceptual_model.extractor_list:
            extractor.model.eval()

    def on_validation_epoch_start(self):
        self.__reset_val_metrics()

    def on_validation_epoch_end(self):
        loss = self.val_metrics['loss'].compute()
        score = self.val_metrics['score'].compute()

        # cast to Python numbers for W&B
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

                # Save "best" LoRA adapter snapshot if applicable
                if self.use_lora and self.save_mode != 'entire_model':
                    base_dir = getattr(self.trainer, "default_root_dir", os.getcwd())
                    best_dir = os.path.join(base_dir, "lora_adapters", "best")
                    os.makedirs(best_dir, exist_ok=True)
                    self.perceptual_model.save_pretrained(best_dir)  # overwrite on improvement
        except Exception:
            pass

        # ---- Track best accuracy too (no save here) ----
        try:
            if score_float > self.best_val_acc:
                self.best_val_acc  = score_float
                self.best_acc_epoch = int(self.current_epoch)
        except Exception:
            pass

        self.log("epoch", int(self.current_epoch)+1, prog_bar=False, on_epoch=True, logger=True)
        return score

    def configure_optimizers(self):
        # Grab only parameters that are actually trainable (requires_grad=True).
        params = [p for p in self.parameters() if p.requires_grad]

        # Adam with standard betas for ViT-style training.
        optimizer = torch.optim.Adam(
            params, lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay
        )
        return [optimizer]

    def load_lora_weights(self, checkpoint_root, epoch_load=None):
        """
        Load LoRA adapters (adapter_only/all) or a full Lightning checkpoint (entire_model).
        """
        if self.save_mode in {'adapter_only', 'all'}:
            # Resolve epoch subfolder if provided
            if epoch_load is not None:
                checkpoint_root = os.path.join(checkpoint_root, f'epoch_{epoch_load}')
            else:
                raise ValueError("epoch_load must be provided when loading LoRA adapters.")

            logging.info(f'Loading adapter weights from {checkpoint_root}')
            target_device = getattr(self, "_target_device", next(self.parameters()).device)

            # Case A: already LoRA-wrapped → attach adapter into this PeftModel
            if isinstance(self.perceptual_model, PeftModel):
                self.perceptual_model.load_adapter(
                    checkpoint_root, adapter_name="default", is_trainable=True
                )
                self.perceptual_model.set_adapter("default")
                self.perceptual_model.to(target_device)  # in-place move

            # Case B: not wrapped yet → wrap base model from saved adapter dir
            else:
                self.perceptual_model = PeftModel.from_pretrained(
                    self.perceptual_model, checkpoint_root
                ).to(target_device)

        else:
            if epoch_load is None:
                raise ValueError("epoch_load must be provided when loading an entire model checkpoint.")
            # Handle both hyphen and equals styles
            ckpt_hyphen = os.path.join(checkpoint_root, f'epoch-{epoch_load:02d}.ckpt')
            ckpt_equal  = os.path.join(checkpoint_root, f'epoch={epoch_load:02d}.ckpt')
            ckpt_path = ckpt_hyphen if os.path.exists(ckpt_hyphen) else ckpt_equal

            logging.info(f'Loading entire model from {ckpt_path}')
            sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
            self.load_state_dict(sd, strict=True)
            # put the LightningModule on the intended device
            self.to(getattr(self, "_target_device", next(self.parameters()).device))

    def __reset_val_metrics(self):
        for k, v in self.val_metrics.items():
            v.reset()

    def __prep_lora_model(self):
        """
        Wrap self.perceptual_model with LoRA, auto-detecting Linear layers across ViT/CLIP variants.
        """
        candidates = ["qkv", "q_proj", "k_proj", "v_proj", "out_proj"]
        present = set()
        for name, module in self.perceptual_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                for tok in candidates:
                    if tok in name:
                        present.add(tok)
        target_modules = sorted(present)
        if not target_modules:
            raise RuntimeError("LoRA injection found no matching Linear modules in the backbone.")

        cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        self.perceptual_model = get_peft_model(self.perceptual_model, cfg)

        # sanity: make sure some params are trainable
        if sum(p.numel() for p in self.perceptual_model.parameters() if p.requires_grad) == 0:
            raise RuntimeError("LoRA wrapped the model but no parameters are trainable.")

    def __prep_linear_model(self):
        """
        Non-LoRA path: freeze backbones, train only the small MLP head.
        """
        for extractor in self.perceptual_model.extractor_list:
            if hasattr(extractor, "model"):
                extractor.model.requires_grad_(False)
            if self.feat_type == "embedding" and hasattr(extractor, "proj"):
                if hasattr(extractor.proj, "requires_grad_"):
                    extractor.proj.requires_grad_(False)

        if hasattr(self.perceptual_model, "mlp"):
            self.perceptual_model.mlp.requires_grad_(True)

    def __save_lora_weights(self):
        """Save PEFT LoRA adapters to a stable location inside the run folder.

        We avoid relying on callbacks' dirpath (which may be missing when save_mode='adapter_only').
        Adapters are stored under: <default_root_dir>/lora_adapters/epoch_<N>/
        """
        if self.save_mode == 'entire_model':
            return

        # default_root_dir is set to exp_dir in run(), so this is a stable base
        base_dir = getattr(self.trainer, "default_root_dir", None) or os.getcwd()
        adapters_root = os.path.join(base_dir, "lora_adapters")
        epoch_dir = os.path.join(adapters_root, f"epoch_{self.trainer.current_epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)

        # PEFT: writes adapter_config.json and adapter_model.bin into epoch_dir
        self.perceptual_model.save_pretrained(epoch_dir)

    def __validate_save_mode(self):
        save_options = {'adapter_only', 'entire_model', 'all'}
        assert self.save_mode in save_options, f'save_mode must be one of {save_options}, got {self.save_mode}'
        logging.info(f'Using save mode: {self.save_mode}')

    def on_after_backward(self):
        # Log a simple global grad-norm every 100 optimizer steps
        if (self.global_step % 100) != 0:
            return

        # Accumulate on-device to avoid per-parameter CPU syncs
        total_sq = None
        for p in self.parameters():
            if p.grad is not None:
                v = p.grad.norm(2).pow(2)          # ||g||^2 as a tensor
                total_sq = v if total_sq is None else total_sq + v

        global_norm = float(total_sq.sqrt().item()) if total_sq is not None else 0.0
        self.log("global_norm", global_norm, prog_bar=False, on_step=True, on_epoch=False, logger=True)

    def __write_best_summary(self, loss_value: float):
        try:
            base_dir = getattr(self.trainer, "default_root_dir", os.getcwd())
            summary_path = os.path.join(base_dir, "best_val_loss.txt")
            with open(summary_path, "w") as f:
                f.write(f"best_epoch: {int(self.current_epoch)+1}\n")
                f.write(f"best_val_loss: {loss_value:.6f}\n")
        except Exception:
            pass

    def on_fit_end(self):
        # save final LoRA adapter snapshot as "last"
        if self.use_lora and self.save_mode != 'entire_model':
            base_dir = getattr(self.trainer, "default_root_dir", None) or os.getcwd()
            out_dir = os.path.join(base_dir, "lora_adapters", "last")
            os.makedirs(out_dir, exist_ok=True)
            self.perceptual_model.save_pretrained(out_dir)


# Periodic saver for MLP runs only (full Lightning .ckpt); LoRA saves every epoch elsewhere
class SaveEveryNEpochsMLP(pl.Callback):
    def __init__(self, n: int, out_dir: str):
        super().__init__()
        self.n = int(n)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Only act for MLP runs (use_lora=False)
        if not self.n or getattr(pl_module.hparams, 'use_lora', False):
            return
        if ((trainer.current_epoch + 1) % self.n) != 0:
            return
        fname = f"epoch_{trainer.current_epoch + 1}_run.ckpt"
        trainer.save_checkpoint(os.path.join(self.out_dir, fname))

    def on_exception(self, trainer, pl_module, err):
        # Emergency save for MLP runs
        if getattr(pl_module.hparams, 'use_lora', False):
            return
        try:
            trainer.save_checkpoint(os.path.join(self.out_dir, "_emergency_last.ckpt"))
        except Exception:
            pass


def run(args):
    """
    Execute one training run end-to-end.

    Steps:
      1) Resolve device ('cuda' if available else 'cpu').
      2) Create experiment directory:
           exp_dir = <project_root>/<log_dir>/<tag>_<model_type>_<mode>_<feat>_<stride>_<dataset>[_lorar]
         and write run_config.yaml there.
      3) Seed RNGs, enable cuDNN benchmark, and seed DataLoader workers.
      4) Select dataset class from args.dataset_name and build train/val DataLoaders
         using roots joined to project_root; enable pin_memory/persistent_workers when appropriate.
      5) Initialize a W&B logger and declare 'epoch' as the step metric.
      6) Set up callbacks:
           - EarlyStopping on val_loss
           - ModelCheckpoint (best/last) unless save_mode == 'adapter_only'
      7) Instantiate LightningPerceptualModel (LoRA or MLP path) and report parameter counts.
      8) Push parameter counts into W&B config and start training with trainer.fit().

    Outputs:
      - Metrics to W&B
      - Checkpoints to exp_dir (best.ckpt/last.ckpt) when save_mode != 'adapter_only'
      - LoRA adapters to exp_dir/lora_adapters/epoch_<N>/, plus best/ and last/ (when use_lora and save_mode != 'entire_model')
      - best_val_loss.txt with best epoch and loss.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.tag if len(args.tag) > 0 else ""
    training_method = "lora" if args.use_lora else "mlp"
    os.makedirs(os.path.join(args.project_root, "vision_models", "vits_backbone_models"), exist_ok=True)

    # Build a readable run stub (used for a local folder name only)
    run_stub = (
        f'{tag}_{str(args.model_type)}_{str(training_method)}_{str(args.feat_type)}_{str(args.stride)}_{str(args.dataset_name)}'
    )

    base_dir = os.path.join(args.project_root, args.log_dir.lstrip('/'))
    os.makedirs(base_dir, exist_ok=True)
    exp_dir = os.path.join(base_dir, run_stub)
    os.makedirs(exp_dir, exist_ok=True)

    # Save run config once for reproducibility
    cfg_path = os.path.join(exp_dir, "run_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Seed & perf knobs
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Pick dataset class
    if args.dataset_name == "nights":
        from vision_models.dreamsim.dataset.dataset import NightsDataset as DatasetCls
    elif args.dataset_name == "fashion_triplets":
        from vision_models.dreamsim.dataset.dataset import FashionTripletsDataset as DatasetCls
    elif args.dataset_name == "nights_fashion_triplets":
        from vision_models.dreamsim.dataset.dataset import NightsFashionTripletsDataset as DatasetCls
    elif args.dataset_name == "synthetic_fashion_1":
        from vision_models.dreamsim.dataset.dataset import SyntheticFashionDataset1 as DatasetCls
    elif args.dataset_name == "synthetic_fashion_2":
        from vision_models.dreamsim.dataset.dataset import SyntheticFashionDataset2 as DatasetCls
    else:
        raise ValueError(f"Unknown dataset_name: {args.dataset_name}")

    first_root_dir  = args.dataset_root
    second_root_dir = args.second_dataset_root

    train_dataset = DatasetCls(
        root_dir=first_root_dir,
        random_seed=args.seed,
        second_root_dir=second_root_dir,
        split="train",
        preprocess=get_preprocess(args.model_type),
    )
    val_dataset = DatasetCls(
        root_dir=first_root_dir,
        random_seed=args.seed,
        second_root_dir=second_root_dir,
        split="val",
        preprocess=get_preprocess(args.model_type),
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

    # Console logging only (no exp.log file)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("Arguments: %s", vars(args))
    logging.info(f"[RUN] exp_dir = {exp_dir}")

    # W&B logger
    model_token = '+'.join(
    m.split('_vit')[0].replace('open_clip', 'openclip')
    for m in args.model_type.split(',')
    )
    run_name = f"{args.vision_model_training_name}_{model_token}_{training_method}_{args.dataset_name}"
    logger = WandbLogger(
        save_dir=exp_dir,
        project=args.wandb_project,
        name=run_name,
        group=args.vision_model_training_name,
        job_type='train vit',
        tags=[args.vision_model_training_name, 'vision_transformer', model_token, training_method, args.dataset_name],
        notes=args.wandb_notes,
        log_model=False
    )

    # Make W&B use 'epoch' as the x-axis for epoch-level metrics
    logger.experiment.define_metric("epoch")  # declare step metric
    for m in [
        "train_loss", "train_2afc_acc",
        "val_loss", "val_2afc_acc",
        "time/epoch_sec", "system/max_gpu_mem_mb", "optim/lr"
    ]:
        logger.experiment.define_metric(m, step_metric="epoch")

    # Trainer
    callbacks = []
    if args.save_mode != 'adapter_only':
        ckpt_best = ModelCheckpoint(dirpath=exp_dir, filename='best',
                                    monitor='val_loss', mode='min', save_top_k=1,
                                    save_last=True)  # writes/updates last.ckpt too
        callbacks = [ckpt_best]

    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.patience,
        min_delta=args.min_delta,
        check_on_train_epoch_end=False
    )
    callbacks.append(early_stop)

    # Periodic saver for MLP runs (LoRA already saves every epoch)
    if (not args.use_lora) and args.auto_save and args.auto_save > 0 and args.save_mode != 'adapter_only':
        callbacks.append(SaveEveryNEpochsMLP(args.auto_save, exp_dir))
        logging.info(
            f"Periodic saving every {args.auto_save} epochs → {exp_dir} (full .ckpt, MLP only)"
        )

    accel = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = Trainer(
        devices=1,
        accelerator=accel,
        precision=precision,
        log_every_n_steps=10,
        logger=logger,
        max_epochs=args.epochs,
        default_root_dir=exp_dir,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    root_dev = trainer.strategy.root_device
    logging.info(f"[PL] accelerator={accel} precision={precision} root_device={root_dev} | cuda={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")

    # Model + train
    model = LightningPerceptualModel(
        device=device, train_data_len=len(train_dataset), **vars(args)
    )

    # NEW — warm start
    if args.load_path is not None:
        if not os.path.exists(args.load_path):
            raise FileNotFoundError(f"--load_path not found: {args.load_path}")

        if not args.use_lora:
            # MLP path: load weights for perceptual_model.* from a Lightning .ckpt
            logging.info(f"[Warm start][MLP] Loading weights from {args.load_path}")
            ckpt = torch.load(args.load_path, map_location=device)
            sd = ckpt.get('state_dict', {})
            stripped = {
                k.replace('perceptual_model.', '', 1): v
                for k, v in sd.items() if k.startswith('perceptual_model.')
            }
            missing, unexpected = model.perceptual_model.load_state_dict(stripped, strict=False)
            if missing:
                logging.warning(f"[Warm start][MLP] Missing keys (up to 8): {missing[:8]}{' ...' if len(missing)>8 else ''}")
            if unexpected:
                logging.warning(f"[Warm start][MLP] Unexpected keys (up to 8): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
        else:
            # LoRA path: attach adapter from a directory (e.g., .../lora_adapters/best OR .../epoch_N)
            adapter_root = args.load_path
            if args.load_lora_epoch is not None:
                adapter_root = os.path.join(adapter_root, f"epoch_{args.load_lora_epoch}")
            logging.info(f"[Warm start][LoRA] Loading adapter from {adapter_root}")

            if not os.path.exists(adapter_root):
                raise FileNotFoundError(f"LoRA adapter path not found: {adapter_root}")

            target_device = getattr(model, "_target_device", device)
            if isinstance(model.perceptual_model, PeftModel):
                # Already wrapped: load as an additional adapter and select it
                model.perceptual_model.load_adapter(adapter_root, adapter_name="default", is_trainable=True)
                model.perceptual_model.set_adapter("default")
                model.perceptual_model.to(target_device)
            else:
                # Wrap base model with adapter from disk
                model.perceptual_model = PeftModel.from_pretrained(model.perceptual_model, adapter_root).to(target_device)


    # Compute once
    total_params = sum(p.numel() for p in model.perceptual_model.parameters())
    trainable_params = sum(p.numel() for p in model.perceptual_model.parameters() if p.requires_grad)

    # Build one dict and update config in a single call
    cfg = dict(vars(args))
    cfg.update({
        'params_total': total_params,
        'params_trainable': trainable_params,
        'params_percent_trainable': 100.0 * trainable_params / max(1, total_params),
    })
    logger.experiment.config.update(cfg, allow_val_change=True)

    logging.info("Training")
    trainer.fit(model, train_loader, val_loader)
    print("Done :)")
