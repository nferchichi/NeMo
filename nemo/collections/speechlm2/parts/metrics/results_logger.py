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
import json
import os
import shutil

import torch
import torchaudio

from nemo.collections.speechlm2.parts.metrics.text_metric_utils import (
    normalize_for_metric,
    sentence_bleu_on_normalized,
    wer_on_normalized,
)
from nemo.utils import logging


def safe_remove_path(path):
    try:
        shutil.rmtree(path)
    except:
        pass  # File was already deleted by another thread


class ResultsLogger:
    """
    Saves audios and a json file with the model outputs.
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.audio_save_path = os.path.join(save_path, "pred_wavs")
        os.makedirs(self.audio_save_path, exist_ok=True)
        self.matadata_save_path = os.path.join(save_path, "metadatas")
        os.makedirs(self.matadata_save_path, exist_ok=True)

    def reset(self):
        metadata_files = os.listdir(self.matadata_save_path)
        for f in metadata_files:
            open(os.path.join(self.matadata_save_path, f), 'w').close()
        return self

    @staticmethod
    def merge_and_save_audio(
        out_audio_path: str, pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor, user_audio_sr: int
    ) -> None:
        user_audio = torchaudio.functional.resample(user_audio.float(), user_audio_sr, pred_audio_sr)
        T1, T2 = pred_audio.shape[0], user_audio.shape[0]
        max_len = max(T1, T2)
        pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
        user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

        combined_wav = torch.cat(
            [
                user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
            ],
            dim=0,
        )

        torchaudio.save(out_audio_path, combined_wav.squeeze(), pred_audio_sr)
        logging.info(f"Audio saved at: {out_audio_path}")

    def update(
        self,
        name: str,
        refs: list[str],
        hyps: list[str],
        src_refs: list[str],
        samples_id: list[str],
        pred_audio: torch.Tensor,
        pred_audio_sr: int,
        user_audio: torch.Tensor,
        user_audio_sr: int,
        eou_pred: torch.Tensor = None,
        fps: float = None,
        results=None,
        tokenizer=None,
        src_hyps_asr_head: list[str] | None = None,
        src_hyps_rnnt: list[str] | None = None,
        ast_bleu_refs: list[str] | None = None,
        ast_text_normalizer=None,
        asr_wer_normalizer=None,
    ) -> None:

        out_json_path = os.path.join(self.matadata_save_path, f"{name}.json")
        out_dicts = []
        for i in range(len(refs)):
            sample_id = samples_id[i][:150]
            out_dir = os.path.join(self.audio_save_path, name)
            os.makedirs(out_dir, exist_ok=True)
            out_audio_path = os.path.join(out_dir, f"{sample_id}.wav")
            self.merge_and_save_audio(out_audio_path, pred_audio[i], pred_audio_sr, user_audio[i], user_audio_sr)
            if eou_pred is not None:
                out_audio_path_eou = os.path.join(out_dir, f"{sample_id}_eou.wav")
                repeat_factor = int(pred_audio_sr / fps)
                eou_pred_wav = (
                    eou_pred[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, repeat_factor)
                )
                eou_pred_wav = eou_pred_wav.view(1, -1)
                eou_pred_wav = eou_pred_wav.float() * 0.8
                torchaudio.save(out_audio_path_eou, eou_pred_wav.squeeze().unsqueeze(0).detach().cpu(), pred_audio_sr)

            pred_src_asr_head = (
                src_hyps_asr_head[i]
                if src_hyps_asr_head is not None and i < len(src_hyps_asr_head) and src_hyps_asr_head[i] is not None
                else ""
            )
            pred_src_rnnt = (
                src_hyps_rnnt[i]
                if src_hyps_rnnt is not None and i < len(src_hyps_rnnt) and src_hyps_rnnt[i] is not None
                else ""
            )
            ast_ref = (
                ast_bleu_refs[i]
                if ast_bleu_refs is not None and i < len(ast_bleu_refs) and ast_bleu_refs[i] is not None
                else refs[i]
            )
            if ast_text_normalizer is not None:
                ast_ref_norm = normalize_for_metric(ast_ref, ast_text_normalizer)
                pred_tgt_norm = normalize_for_metric(hyps[i], ast_text_normalizer)
                ast_sbleu = sentence_bleu_on_normalized(ast_ref_norm, pred_tgt_norm)
            else:
                ast_ref_norm, pred_tgt_norm, ast_sbleu = "", "", None

            src_ref_norm = ""
            if asr_wer_normalizer is not None:
                src_ref_norm = normalize_for_metric(src_refs[i], asr_wer_normalizer)

            if asr_wer_normalizer is not None and pred_src_rnnt:
                rnnt_norm = normalize_for_metric(pred_src_rnnt, asr_wer_normalizer)
                asr_wer = wer_on_normalized(src_ref_norm, rnnt_norm)
            else:
                rnnt_norm = ""
                asr_wer = None

            out_dict = {
                "target_translation": {
                    "target": refs[i],
                    "pred": hyps[i],
                    "target_normalized": ast_ref_norm,
                    "pred_normalized": pred_tgt_norm,
                },
                "ast_sentence_bleu": ast_sbleu,
                "source_transcript": {
                    "target": src_refs[i],
                    "pred": pred_src_rnnt,
                    "target_normalized": src_ref_norm,
                    "pred_normalized": rnnt_norm,
                },
                "asr_wer_rnnt": asr_wer,
                "audio_path": os.path.relpath(out_audio_path, self.save_path),
            }
            if pred_src_asr_head:
                logging.info(f"[ASR head] Sample {sample_id}: {pred_src_asr_head}")
            if pred_src_rnnt:
                logging.info(f"[ASR RNNT] Sample {sample_id}: {pred_src_rnnt}")
            if results is not None:
                if tokenizer is not None:
                    out_dict["target_translation"]["tokens_text"] = " ".join(
                        tokenizer.ids_to_tokens(results['tokens_text'][i])
                    )
                else:
                    out_dict["target_translation"]["tokens_text"] = results['tokens_text'][i].tolist()
            out_dicts.append(out_dict)
        with open(out_json_path, 'a+', encoding='utf-8') as fout:
            for out_dict in out_dicts:
                json.dump(out_dict, fout, indent=4, ensure_ascii=False)

        logging.info(f"Metadata file for {name} dataset updated at: {out_json_path}")
