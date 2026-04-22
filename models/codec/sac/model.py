from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.log as log
from audiotools import AudioSignal
from omegaconf import DictConfig
from utils.checkpoint import strip_prefix
from utils.file import load_config


class XVC(nn.Module):
    def __init__(
        self,
        loss_config: DictConfig = None,
        semantic_encoder: nn.Module = None,
        semantic_adapter: nn.Module = None,
        acoustic_encoder: nn.Module = None,
        acoustic_converter: nn.Module = None,
        acoustic_quantizer: nn.Module = None,
        speaker_predictor: nn.Module = None,
        prenet: nn.Module = None,
        semantic_decoder: nn.Module = None,
        acoustic_decoder: nn.Module = None,
        speaker_encoder: nn.Module = None,
        mel_extractor: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        self.loss_config = loss_config

        self.semantic_encoder = semantic_encoder
        self.semantic_adapter = semantic_adapter
        self.acoustic_encoder = acoustic_encoder
        self.acoustic_converter = acoustic_converter
        self.acoustic_quantizer = acoustic_quantizer
        self.prenet = prenet
        self.speaker_encoder = speaker_encoder
        self.speaker_predictor = speaker_predictor
        self.semantic_decoder = semantic_decoder
        self.acoustic_decoder = acoustic_decoder
        self.mel_extractor = mel_extractor

        if loss_config is not None:
            self.init_loss_function(loss_config)

        if self.semantic_encoder and self.semantic_encoder.from_pretrained is not None:
            semantic_encoder = self.init_semantic_encoder(
                cfg=self.semantic_encoder.from_pretrained
            )
            if hasattr(self.semantic_encoder, "encoder"):
                self.semantic_encoder.encoder = semantic_encoder
            else:
                self.semantic_encoder = semantic_encoder

    def init_semantic_encoder(self, cfg: dict = None):
        from models.codec.sac.modules.semantic_encoder import WhisperVQEncoder

        model_path = cfg.get("local_ckpt") or cfg.get("hf_repo")
        encoder = WhisperVQEncoder.from_pretrained(model_path)

        load_codebook_only = cfg.get("load_codebook_only", False)
        if load_codebook_only:
            encoder._prune_to_codebook_only()

        if cfg.get("freeze", False):
            encoder._freeze_parameters()
            encoder.eval()
        return encoder

    @classmethod
    def load_from_checkpoint(
        cls, 
        config_path: Path, 
        ckpt_path: Path, 
        device: torch.device, **kwargs
    ):
        """
        Load pre-trained model

        Args:
            config_path (Path): path to the model model configuration.
            ckpt_path (Path): path of model checkpoint.
            device (torch.device): The device to load the model onto.

        Kwargs:
            ema_load (bool): If True and EMA weights are present, prefer 'ema_generator'.
            
        Returns:
            model (nn.Module): The loaded model instance.
        """
        cfg = load_config(config_path)
        if "config" in cfg.keys():
            cfg = cfg["config"]

        cls.device = device
        gen_cfg = cfg["model"]["generator"]
        semantic_encoder = hydra.utils.instantiate(gen_cfg["semantic_encoder"]) if gen_cfg.get("semantic_encoder") else False
        semantic_adapter = hydra.utils.instantiate(gen_cfg["semantic_adapter"]) if gen_cfg.get("semantic_adapter") else False
        acoustic_encoder = hydra.utils.instantiate(gen_cfg["acoustic_encoder"]) if gen_cfg.get("acoustic_encoder") else False
        acoustic_converter_cfg = gen_cfg.get("acoustic_converter")
        acoustic_converter = hydra.utils.instantiate(acoustic_converter_cfg) if acoustic_converter_cfg else False
        acoustic_quantizer = hydra.utils.instantiate(gen_cfg["acoustic_quantizer"]) if gen_cfg.get("acoustic_quantizer") else False
        speaker_predictor = hydra.utils.instantiate(gen_cfg["speaker_predictor"]) if gen_cfg.get("speaker_predictor") else False
        speaker_encoder = hydra.utils.instantiate(gen_cfg["speaker_encoder"]) if gen_cfg.get("speaker_encoder") else False
        prenet = hydra.utils.instantiate(gen_cfg["prenet"]) if gen_cfg.get("prenet") else False
        semantic_decoder = hydra.utils.instantiate(gen_cfg["semantic_decoder"]) if gen_cfg.get("semantic_decoder") else False
        acoustic_decoder = hydra.utils.instantiate(gen_cfg["acoustic_decoder"]) if gen_cfg.get("acoustic_decoder") else False
        mel_extractor = hydra.utils.instantiate(gen_cfg["mel_extractor"]) if gen_cfg.get("mel_extractor") else False

        model = cls(
            loss_config=None,
            semantic_encoder=semantic_encoder,
            semantic_adapter=semantic_adapter,
            acoustic_encoder=acoustic_encoder,
            acoustic_converter=acoustic_converter,
            acoustic_quantizer=acoustic_quantizer,
            speaker_predictor=speaker_predictor,
            prenet=prenet,
            semantic_decoder=semantic_decoder,
            acoustic_decoder=acoustic_decoder,
            speaker_encoder=speaker_encoder,
            mel_extractor=mel_extractor,
        )

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        want_ema = bool(cfg.get("ema_update", False)) and bool(kwargs.get("ema_load", False))
        key = "ema_generator" if (want_ema and "ema_generator" in state_dict) else "generator"
        sd = state_dict[key]

        if key == "ema_generator":
            sd = strip_prefix(sd, "ema_model.")
            log.info(f"[x-vc] Loading EMA weights from: {ckpt_path}")
        else:
            log.info(f"[x-vc] Loading weights from: {ckpt_path}")

        missing_keys, unexpected_keys = model.load_state_dict(
            sd, strict=False
        )

        for key in missing_keys:
            log.info("[x-vc] Missing tensor: {}".format(key))
        for key in unexpected_keys:
            log.info("[x-vc] Unexpected tensor: {}".format(key))

        model.to(device).eval()
        return model

    def forward(self, inputs: dict):
        """
        Forward pass of x-vc.
        """
        semantic_tokens = inputs.get("semantic_tokens", None)
        source_wav = inputs["source_wav"]
        target_wav = inputs["target_wav"]
        target_wav_cond = inputs.get("target_wav_cond", target_wav)
        ssl_feat = inputs.get("ssl_feat", None)

        # Note: online feature extraction is not supported yet.
        feat = semantic_tokens

        if feat is None:
            feat = self.semantic_encoder.extract_and_encode(source_wav.squeeze(1))["speech_tokens"]
            ssl_feat = self.semantic_encoder.extract_and_encode(target_wav.squeeze(1))["whisper_hidden_states_50hz"]

        # xvc simplified condition path:
        # - utterance-level condition: speaker embedding from speaker encoder
        # - frame-level condition: target mel
        if not self.speaker_encoder:
            raise RuntimeError("x-vc requires `speaker_encoder`.")
        if not self.mel_extractor:
            raise RuntimeError("x-vc requires `mel_extractor`.")
        with torch.no_grad():
            gt_sim_feat, _ = self.speaker_encoder(target_wav)
            speaker_condition = gt_sim_feat

        frame_condition = self.mel_extractor(target_wav_cond)

        sem_emb = self.semantic_encoder.embed_ids(feat)  # B x T_s x D_s

        if self.semantic_adapter:
            sem_emb = self.semantic_adapter(sem_emb.transpose(1, 2)).transpose(1, 2)
        else:
            raise RuntimeError("x-vc requires `semantic_adapter`.")

        if self.acoustic_encoder:
            acoustic_encoder_out = self.acoustic_encoder(source_wav)   # B x 1 x T_a -> B x D_a x T_a
        else:
            raise RuntimeError("x-vc requires `acoustic_encoder`.")

        if self.acoustic_quantizer:
            aq_outputs = self.acoustic_quantizer(acoustic_encoder_out)
            if len(aq_outputs) == 7:
                zq_a, a_indices, a_commit_loss, a_codebook_loss, _, a_perplexity, a_cluster_size = aq_outputs
                vq_loss = (a_commit_loss + a_codebook_loss).mean()
            elif len(aq_outputs) == 5:
                zq_a, a_indices, vq_loss, a_perplexity, a_cluster_size = aq_outputs
                if isinstance(vq_loss, tuple):
                    vq_loss = vq_loss[0]
            elif len(aq_outputs) == 2:
                zq_a, a_indices = aq_outputs
                vq_loss, a_perplexity, a_cluster_size = 0.0, 0.0, 0
            else:
                raise RuntimeError("Unexpected acoustic_quantizer output format.")
        else:
            raise RuntimeError("x-vc requires `acoustic_quantizer`.")

        acu_emb = zq_a.transpose(1, 2)  # B x T_a x D_a
        combined_emb = torch.cat([sem_emb, acu_emb], dim=2)  # B x T x (D_s + D_a)

        prenet_in = combined_emb.transpose(1, 2)
        if self.prenet:
            x = self.prenet(prenet_in, speaker_condition)
        else:
            raise RuntimeError("x-vc requires `prenet`.")

        if self.acoustic_converter:
            x = self.acoustic_converter(x, frame_condition, speaker_condition)
        else:
            raise RuntimeError("x-vc requires `acoustic_converter`.")

        if self.acoustic_decoder:
            y = self.acoustic_decoder(x)
        else:
            raise RuntimeError("x-vc requires `acoustic_decoder`.")

        if self.semantic_decoder:
            pred_feat = self.semantic_decoder(x)
        else:
            raise RuntimeError("x-vc requires `semantic_decoder`.")

        if self.speaker_predictor:
            proj_sim_feat = self.speaker_predictor(x)
        else:
            raise RuntimeError("x-vc requires `speaker_predictor`.")

        return {
            "recons": y,
            "pred": pred_feat,
            "zq": zq_a,
            "z": acoustic_encoder_out,
            "vqloss": vq_loss,
            "perplexity": a_perplexity,
            "cluster_size": a_cluster_size,
            "ssl_feat": ssl_feat,
            "step": inputs.get("step", 0),
            "quantized_tokens": a_indices,
            "pred_sim_feat": proj_sim_feat,
            "sim_feat": gt_sim_feat,
        }

    @torch.no_grad()
    def inference(self, inputs: dict):
        """
        Forward pass of x-vc (inference mode).
        """
        semantic_tokens = inputs.get("semantic_tokens", None)
        source_wav = inputs["source_wav"]
        target_wav = inputs["target_wav"]
        target_wav_cond = inputs.get("target_wav_cond", target_wav)
        ssl_feat = inputs.get("ssl_feat", None)

        # Note: online feature extraction is not supported yet.
        feat = semantic_tokens

        if feat is None:
            feat = self.semantic_encoder.extract_and_encode(source_wav.squeeze(1))["speech_tokens"]
            ssl_feat = self.semantic_encoder.extract_and_encode(target_wav.squeeze(1))["whisper_hidden_states_50hz"]

        # xvc simplified condition path:
        # - utterance-level condition: speaker embedding from speaker encoder
        # - frame-level condition: target mel
        if not self.speaker_encoder:
            raise RuntimeError("x-vc requires `speaker_encoder`.")
        if not self.mel_extractor:
            raise RuntimeError("x-vc requires `mel_extractor`.")
        with torch.no_grad():
            gt_sim_feat, _ = self.speaker_encoder(target_wav)
            speaker_condition = gt_sim_feat

        frame_condition = self.mel_extractor(target_wav_cond)

        sem_emb = self.semantic_encoder.embed_ids(feat)  # B x T_s x D_s

        if self.semantic_adapter:
            sem_emb = self.semantic_adapter(sem_emb.transpose(1, 2)).transpose(1, 2)
        else:
            raise RuntimeError("x-vc requires `semantic_adapter`.")

        if self.acoustic_encoder:
            acoustic_encoder_out = self.acoustic_encoder(source_wav)   # B x 1 x T_a -> B x D_a x T_a
        else:
            raise RuntimeError("x-vc requires `acoustic_encoder`.")

        if self.acoustic_quantizer:
            aq_outputs = self.acoustic_quantizer(acoustic_encoder_out)
            if len(aq_outputs) == 7:
                zq_a, a_indices, a_commit_loss, a_codebook_loss, _, a_perplexity, a_cluster_size = aq_outputs
                vq_loss = (a_commit_loss + a_codebook_loss).mean()
            elif len(aq_outputs) == 5:
                zq_a, a_indices, vq_loss, a_perplexity, a_cluster_size = aq_outputs
                if isinstance(vq_loss, tuple):
                    vq_loss = vq_loss[0]
            elif len(aq_outputs) == 2:
                zq_a, a_indices = aq_outputs
                vq_loss, a_perplexity, a_cluster_size = 0.0, 0.0, 0
            else:
                raise RuntimeError("Unexpected acoustic_quantizer output format.")
        else:
            raise RuntimeError("x-vc requires `acoustic_quantizer`.")

        acu_emb = zq_a.transpose(1, 2)  # B x T_a x D_a
        combined_emb = torch.cat([sem_emb, acu_emb], dim=2)  # B x T x (D_s + D_a)

        prenet_in = combined_emb.transpose(1, 2)
        if self.prenet:
            x = self.prenet(prenet_in, speaker_condition)
        else:
            raise RuntimeError("x-vc requires `prenet`.")

        if self.acoustic_converter:
            x = self.acoustic_converter(x, frame_condition, speaker_condition)
        else:
            raise RuntimeError("x-vc requires `acoustic_converter`.")

        if self.acoustic_decoder:
            y = self.acoustic_decoder(x)
        else:
            raise RuntimeError("x-vc requires `acoustic_decoder`.")

        return {
            "recons": y,
            "zq": zq_a,
            "z": acoustic_encoder_out,
            "vqloss": vq_loss,
            "perplexity": a_perplexity,
            "cluster_size": a_cluster_size,
            "ssl_feat": ssl_feat,
            "step": inputs.get("step", 0),
            "quantized_tokens": a_indices,
            "sim_feat": gt_sim_feat,
        }

    def generative_loss(
        self,
        inputs: dict,
    ):
        """
        Compute the generator-side composite loss.

        Args:
            inputs (dict): A dictionary that should contain the following elements:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'loss'
        """
        loss_dict = {}

        # vq loss
        loss_dict["vq_loss"] = inputs["vqloss"]
        loss_dict["perplexity"] = inputs["perplexity"]
        loss_dict["cluster_size"] = inputs["cluster_size"]
        # reconstruction loss of ssl feature
        loss_dict["mse_loss"] = self.compute_mse_loss(inputs["pred"], inputs["ssl_feat"])

        # reconstruction loss of speaker feature
        if (
            "pred_sim_feat" in inputs
            and inputs["pred_sim_feat"] is not None
            and "sim_feat" in inputs
            and inputs["sim_feat"] is not None
        ):
            loss_dict["sim_mse_loss"] = self.compute_mse_loss(inputs["pred_sim_feat"], inputs["sim_feat"])

        # mel reconstruction loss
        signal = AudioSignal(inputs["audios"], self.loss_config["sample_rate"])
        recons = AudioSignal(inputs["recons"], signal.sample_rate)
        loss_dict["mel_loss"] = self.compute_mel_loss(recons, signal)

        loss = sum(
            [
                v * loss_dict[k]
                for k, v in self.loss_config["loss_weights"].items()
                if k in loss_dict
            ]
        )
        loss_dict = {
            k: v.item() for k, v in loss_dict.items() if not isinstance(v, int)
        }
        loss_dict["loss"] = loss
        loss_dict["gen_loss"] = loss.item()

        return loss_dict

    def init_loss_function(self, loss_config):
        from models.codec.sac.blocks import loss as losses

        # In the inference process, initialization of this function can be skipped.
        if loss_config is None:
            return
        # Init loss function for training process.
        self.compute_mse_loss = nn.MSELoss()
        self.compute_mel_loss = losses.MelSpectrogramLoss(**loss_config["mel_loss"])
    
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class WavDiscriminator(nn.Module):
    """
    Patch-discriminator on waveform.
    """
    def __init__(
        self, loss_config: dict = None, discriminator: nn.Module = None, **kwargs
    ):
        super().__init__()
        self.loss_config = loss_config
        self.discriminator = discriminator

    def forward(self, fake: AudioSignal, real: AudioSignal):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminative_loss(self, inputs: dict):
        """
        Compute D loss:
            E[(D(fake))^2] + E[(1 - D(real))^2]

        Args:
            inputs (dict): A dictionary that should contain the following elements:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'd_loss'
        """
        loss_dict = dict()
        signal = AudioSignal(inputs["audios"], self.loss_config["sample_rate"])
        recons = AudioSignal(inputs["recons"], signal.sample_rate)
        d_loss = self.compute_discriminator_loss(recons, signal)
        loss_dict["loss"] = d_loss
        loss_dict["d_loss"] = d_loss.item()

        return loss_dict

    def adversarial_loss(self, inputs: dict):
        """
        Compute G-side GAN losses:
            - adv_gen_loss: sum over scales of (1 - D(fake_last))^2
            - adv_feat_loss: feature matching L1 between D(fake) and D(real)

        Args:
            inputs (dict): A dictionary that should contain the following elements:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'loss'
        """
        loss_dict = dict()
        signal = AudioSignal(inputs["audios"], self.loss_config["sample_rate"])
        recons = AudioSignal(inputs["recons"], signal.sample_rate)

        d_fake, d_real = self.forward(recons, signal)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

        loss_dict["adv_gen_loss"] = loss_g
        loss_dict["adv_feat_loss"] = loss_feature

        loss = sum(
            [
                v * loss_dict[k]
                for k, v in self.loss_config["loss_weights"].items()
                if k in loss_dict
            ]
        )

        loss_dict = {
            k: v.item() for k, v in loss_dict.items() if not isinstance(v, int)
        }
        loss_dict["loss"] = loss

        return loss_dict

    def compute_discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)

        return loss_d
