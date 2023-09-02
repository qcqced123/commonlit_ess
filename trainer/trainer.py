import torch

import dataset_class.dataclass as dataset_class
import model.loss as model_loss
import model.model as model_arch
from torch.utils.data import DataLoader
from dataset_class.preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import f_beta


class OneToOneTrainer:
    """
    Trainer class for OneToOne DataSchema Pipeline
    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: configuration.CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.p_df = load_data('./dataset_class/data_folder/prompts_train.csv')
        self.s_df = load_data('./dataset_class/data_folder/summaries_train.csv')
        self.tokenizer = self.cfg.tokenizer

    def make_batch(self, fold: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
        """ function for making batch instance """
        train = self.s_df[self.s_df['fold'] != fold].reset_index(drop=True)
        valid = self.s_df[self.s_df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, self.p_df, train
        )
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, self.p_df, valid
        )

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ function for init backbone's configuration & train utils setting """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )
        return model, criterion, val_criterion, optimizer, lr_scheduler, awp

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp: None) -> tuple[Tensor, Tensor, Tensor]:
        """ function for train loop """
        # torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        model.train()

        for step, (inputs, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)

            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # Two target values to GPU
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = model(inputs)
                loss = criterion(preds, labels)
                losses.update(loss, batch_size)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        train_loss = losses.avg.detach().cpu().numpy()
        grad_norm = grad_norm.detach().cpu().numpy()
        return train_loss, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, val_criterion) -> Tensor:
        """ function for validation loop """
        valid_losses = AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                batch_size = labels.size(0)
                preds = model(inputs)
                valid_loss = val_criterion(preds, labels)
                valid_losses.update(valid_loss, batch_size)
        valid_loss = valid_losses.avg.detach().cpu().numpy()
        return valid_loss
