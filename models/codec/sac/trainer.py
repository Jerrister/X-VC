from typing import Dict, Literal, TypedDict

import torch
import torch.nn as nn
import torch.optim as optim
from models.codec.base.base_codec_trainer import BaseCodecTrainer
from omegaconf import DictConfig
from utils.train_utils import (
    toggle_grad,
)


class Batch(TypedDict):
    source_wav: torch.Tensor
    target_wav: torch.Tensor
    semantic_tokens: torch.Tensor
    index: str


class VCCodecTrainer(BaseCodecTrainer):
    """Trainer for wav codec"""

    def __init__(self, cfg: DictConfig = None):
        super().__init__(cfg)

    def update_batch(self, batch):
        batch.update(
            {
                "source_wav": batch["source_wav"].unsqueeze(1),
                "target_wav": batch["target_wav"].unsqueeze(1),
                "semantic_tokens": batch["semantic_tokens"],
                "step": self.step,
            }
        )
        return batch

    def batch_forward(
        self,
        models: Dict[str, nn.Module],
        batch: Batch,
        optimizers: Dict[str, optim.Optimizer] = None,
        schedulers: Dict[str, optim.lr_scheduler._LRScheduler] = None,
        mode: Literal["train", "val"] = "train",
    ):
        batch = self.batch_to_cuda(batch)
        batch = self.update_batch(batch)
        adv_loss_on = self.step > self.config["generator_warmup_steps"]
        loss_dict = dict()
        if self.train_engine == "deepspeed":
            # deepspeed
            # perform amp autocast when dtype != fp32
            with torch.cuda.amp.autocast(
                enabled=self.dtype is not None, dtype=self.dtype, cache_enabled=False
            ):
                outputs = models["generator"](batch)
                L = outputs["recons"].shape[-1]
                outputs["audios"] = batch["target_wav"][:, :, :L]

                # discriminator step
                if adv_loss_on:
                    # NOTE(xinsheng): In the GAN scenario, we have to freeze the state for the parameters being \
                    # handled by other optimizers during an update. Otherwise, it raises a set of issues like:
                    # "runtime error: no attribute 'ipg_index' and 'ipg_buffer' is None ". Please refer to:
                    # https://github.com/microsoft/DeepSpeed/issues/430#issuecomment-698227960
                    toggle_grad(models["discriminator"], True)
                    dis_loss = models["discriminator"].discriminative_loss(outputs)
                    if mode == "train":
                        self.optimizer_zero_grad(optimizers["discriminator"])
                        self.batch_backward_discriminator(
                            models["discriminator"], dis_loss["loss"]
                        )
                        self.update_parameter_and_lr(
                            models["discriminator"],
                            optimizers["discriminator"],
                            schedulers["discriminator"],
                        )
                    loss_dict["d_loss"] = dis_loss["d_loss"]
                    toggle_grad(models["discriminator"], False)

                # generator step
                gen_loss = models["generator"].generative_loss(outputs)
                if adv_loss_on:
                    loss_dict_adv = models["discriminator"].adversarial_loss(outputs)
                    gen_loss["loss"] += loss_dict_adv["loss"]
                    loss_dict_adv.pop("loss")
                    gen_loss.update(loss_dict_adv)

                loss_dict.update(gen_loss)

        else:
            # torch_ddp
            # autocast context
            # The more details about amp can be found in
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            with torch.cuda.amp.autocast(self.scaler is not None):
                outputs = models["generator"](batch)
                L = outputs["recons"].shape[-1]
                outputs["audios"] = batch["target_wav"][:, :, :L]
                # discriminator step
                if adv_loss_on:
                    dis_loss = models["discriminator"].discriminative_loss(outputs)
                    if mode == "train":
                        self.optimizer_zero_grad(optimizers["discriminator"])
                        self.batch_backward_discriminator(
                            models["discriminator"], dis_loss["loss"]
                        )
                        self.update_parameter_and_lr(
                            models["discriminator"],
                            optimizers["discriminator"],
                            schedulers["discriminator"],
                        )
                    loss_dict["d_loss"] = dis_loss["d_loss"]

                # generator step
                gen_loss = models["generator"].generative_loss(outputs)
                if adv_loss_on:
                    loss_dict_adv = models["discriminator"].adversarial_loss(outputs)
                    gen_loss["loss"] += loss_dict_adv["loss"]
                    loss_dict_adv.pop("loss")
                    gen_loss.update(loss_dict_adv)

                loss_dict.update(gen_loss)
        return loss_dict, outputs
