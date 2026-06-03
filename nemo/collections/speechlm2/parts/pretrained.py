# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule

from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging

def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(
    model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32, trust_remote_code: bool = True
):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.

    If user config specifies encoder parameters, they will override the pretrained model's config.
    """
    user_encoder_config = {}
    if "encoder" in model.cfg.perception:
        user_encoder_config = OmegaConf.to_container(model.cfg.perception.encoder, resolve=True)

    asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
    with open_dict(model.cfg):
        model.cfg.perception.preprocessor = asr.cfg.preprocessor
        model.cfg.perception.encoder = asr.cfg.encoder
        model.cfg.perception.output_dim = model.llm.config.hidden_size
        if user_encoder_config:
            for key, value in user_encoder_config.items():
                if value is not None:
                    model.cfg.perception.encoder[key] = value
            # Pretrained ASR encoder may set att_context_probs for multiple att_context_size modes.
            if user_encoder_config.get("att_context_size") is not None:
                enc_cfg = model.cfg.perception.encoder
                if OmegaConf.select(enc_cfg, "att_context_probs") is not None:
                    with open_dict(model.cfg.perception):
                        del model.cfg.perception.encoder["att_context_probs"]
    model.perception = AudioPerceptionModule(model.cfg.perception).train()
    model.perception.load_state_dict(asr.state_dict(), strict=False)


def setup_rnnt_decoder_joint(model: torch.nn.Module, model_path_or_name: str = None):
    """
    Load RNNT decoder and joint from a pretrained NeMo ASR checkpoint (e.g. ``.nemo``).
    The checkpoint must have ``decoder`` and ``joint`` attributes (e.g. RNNT-based ASR).
    The result is assigned to ``model.rnnt_decoder`` and ``model.rnnt_joint``.

    Call this after ``setup_speech_encoder`` if you want to use the RNNT head (e.g. for
    decoding). The path can be the same as ``pretrained_asr`` or a different checkpoint.

    Args:
        model: The DuplexSTTModel (or any module with a ``cfg`` attribute).
        model_path_or_name: Path to a ``.nemo`` ASR checkpoint. If ``None``, uses
            ``model.cfg.get("pretrained_rnnt_asr")``. If that is not set, sets
            ``model.rnnt_decoder`` and ``model.rnnt_joint`` to ``None`` and returns.
    """
    path = model_path_or_name
    if path is None:
        path = model.cfg.get("pretrained_rnnt_asr", None)
    if not path:
        model.rnnt_decoder = None
        model.rnnt_joint = None
        model.rnnt_tokenizer = None
        return

    asr = load_pretrained_nemo(ASRModel, path).eval()
    if not (hasattr(asr, "decoder") and hasattr(asr, "joint")):
        logging.warning(
            "Pretrained ASR at %s has no decoder/joint (got %s). Not loading RNNT head.",
            path,
            type(asr).__name__,
        )
        model.rnnt_decoder = None
        model.rnnt_joint = None
        model.rnnt_tokenizer = None
        return

    with open_dict(asr.cfg.decoder):
        if getattr(asr.cfg.decoder, "vocab_size", None) is None and hasattr(asr, "joint"):
            asr.cfg.decoder.vocab_size = len(asr.joint.vocabulary)
    with open_dict(asr.cfg.joint):
        if getattr(asr.cfg.joint, "num_classes", None) is None and hasattr(asr.joint, "vocabulary"):
            asr.cfg.joint.num_classes = len(asr.joint.vocabulary)
        if getattr(asr.cfg.joint, "vocabulary", None) is None and hasattr(asr.joint, "vocabulary"):
            asr.cfg.joint.vocabulary = asr.joint.vocabulary
    model.rnnt_decoder = type(asr.decoder).from_config_dict(asr.cfg.decoder)
    model.rnnt_joint = type(asr.joint).from_config_dict(asr.cfg.joint)
    asr_sd = asr.state_dict()
    rnnt_sd = {}
    for k, v in asr_sd.items():
        if k.startswith("decoder."):
            rnnt_sd["rnnt_decoder." + k[8:]] = v
        elif k.startswith("joint."):
            rnnt_sd["rnnt_joint." + k[6:]] = v
    model.load_state_dict(rnnt_sd, strict=False)
    if hasattr(asr, "tokenizer") and asr.tokenizer is not None:
        model.rnnt_tokenizer = asr.tokenizer
        logging.info(
            "Loaded ASR tokenizer from pretrained checkpoint for RNNT BPE decoding (ids_to_text): %s",
            path,
        )
    else:
        model.rnnt_tokenizer = None
        logging.info(
            "Pretrained ASR checkpoint has no tokenizer; RNNT will use vocabulary-only decoding: %s",
            path,
        )
    logging.info(
        "Loaded RNNT decoder and joint from pretrained ASR checkpoint: %s",
        path,
    )


def setup_ctc_decoder(model: torch.nn.Module, model_path_or_name: str = None):
    """
    Load the CTC head from a hybrid RNNT+CTC pretrained NeMo ASR checkpoint
    (e.g. ``EncDecHybridRNNTCTCBPEModel``). The result is assigned to
    ``model.ctc_head``.

    Mirror of :func:`setup_rnnt_decoder_joint`. Loads the ``ctc_decoder``
    sub-module (typically a ``ConvASRDecoder`` = single ``Conv1d(D, V+1)``
    over the encoder output) plus its config from ``asr.cfg.aux_ctc.decoder``.
    The shared BPE tokenizer is already loaded by ``setup_rnnt_decoder_joint``
    (hybrid invariant: RNNT and CTC share the tokenizer); we re-use
    ``model.rnnt_tokenizer`` and do not duplicate it.

    Args:
        model: The Lightning module (or any module with a ``cfg`` attribute).
        model_path_or_name: Path to a ``.nemo`` ASR checkpoint. If ``None``,
            falls back to ``model.cfg.get("pretrained_ctc_asr")`` and then to
            ``model.cfg.get("pretrained_rnnt_asr")``. If neither is set, sets
            ``model.ctc_head = None`` and returns.

    Intended call site: only when ``model.cfg.use_ctc_loss=true`` (expN+).
    Pre-expN recipes do not call this function, so they pay zero extra ASR
    load and do not gain a ``ctc_head`` attribute.
    """
    path = model_path_or_name
    if path is None:
        path = model.cfg.get("pretrained_ctc_asr", None)
    if path is None:
        path = model.cfg.get("pretrained_rnnt_asr", None)
    if not path:
        model.ctc_head = None
        return

    asr = load_pretrained_nemo(ASRModel, path).eval()
    if not (hasattr(asr, "ctc_decoder") and "aux_ctc" in asr.cfg):
        logging.warning(
            "Pretrained ASR at %s has no ctc_decoder/aux_ctc (got %s). "
            "Not loading CTC head; model.ctc_head left as None.",
            path,
            type(asr).__name__,
        )
        model.ctc_head = None
        return

    # ConvASRDecoder requires either a positive num_classes or a vocabulary on
    # the cfg; fill both from the live module before from_config_dict, mirroring
    # the pattern used in setup_rnnt_decoder_joint above.
    with open_dict(asr.cfg.aux_ctc.decoder):
        if (
            getattr(asr.cfg.aux_ctc.decoder, "vocabulary", None) is None
            and hasattr(asr.ctc_decoder, "vocabulary")
        ):
            asr.cfg.aux_ctc.decoder.vocabulary = asr.ctc_decoder.vocabulary
        if (
            getattr(asr.cfg.aux_ctc.decoder, "num_classes", -1) is None
            or asr.cfg.aux_ctc.decoder.get("num_classes", -1) < 1
        ) and hasattr(asr.ctc_decoder, "vocabulary"):
            asr.cfg.aux_ctc.decoder.num_classes = len(asr.ctc_decoder.vocabulary)

    model.ctc_head = type(asr.ctc_decoder).from_config_dict(asr.cfg.aux_ctc.decoder)
    asr_sd = asr.state_dict()
    ctc_sd = {}
    for k, v in asr_sd.items():
        if k.startswith("ctc_decoder."):
            ctc_sd["ctc_head." + k[len("ctc_decoder."):]] = v
    # Note: mirrors setup_rnnt_decoder_joint above -- this NeMo LightningModule
    # overrides ``load_state_dict`` to return ``None`` (instead of the standard
    # torch.nn.Module ``_IncompatibleKeys`` namedtuple). Do not try to unpack
    # the return value; rely on the strict=False semantics to apply our subset
    # of keys and ignore the rest.
    model.load_state_dict(ctc_sd, strict=False)
    logging.info(
        "Loaded CTC decoder from pretrained ASR checkpoint: %s "
        "(num_classes_with_blank=%d, ctc_state_dict_keys=%d)",
        path,
        getattr(model.ctc_head, "num_classes_with_blank", -1),
        len(ctc_sd),
    )


def set_model_dict_for_partial_init(pretrained_dict, model_dict):
    # 1. filter out different size layers
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(" | > Layer with shape mismatach in the model definition: {}".format(k)) 
    # 2. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 3. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    logging.info(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict
