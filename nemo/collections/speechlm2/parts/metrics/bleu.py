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
from collections import defaultdict

import sacrebleu
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.speechlm2.parts.metrics.text_metric_utils import normalize_for_metric
from nemo.utils import logging


class BLEU:
    """
    Computes BLEU scores on text predictions.
    References and hypotheses are passed through strip_language_tags_for_scoring then
    EnglishTextNormalizer (or a custom normalizer), matching validation JSON when the same
    normalizer instance is passed to ResultsLogger.
    """

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = True):
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        return self

    def update(self, name: str, refs: list[str], hyps: list[str]) -> None:
        for ref, hyp in zip(refs, hyps):
            normalized_ref = normalize_for_metric(ref, self.normalizer)
            normalized_hyp = normalize_for_metric(hyp, self.normalizer)

            self._refs[name].append(normalized_ref)
            self._hyps[name].append(normalized_hyp)

            if self.verbose:
                asrb = sacrebleu.sentence_bleu(normalized_hyp, [normalized_ref]).score
                logging.info(f"[REF]\t{normalized_ref}\n[HYP]\t{normalized_hyp} [{asrb:.2f}]")

    def compute(self) -> dict[str, torch.Tensor]:
        corpus_metric = {}
        for name in self._refs.keys():
            metric = torch.tensor(sacrebleu.corpus_bleu(self._hyps[name], [self._refs[name]]).score)
            corpus_metric[f"txt_bleu_{name}"] = metric
        self._refs.clear()
        self._hyps.clear()
        if not corpus_metric:
            return {}
        corpus_metric["txt_bleu"] = torch.stack(list(corpus_metric.values())).mean()
        return corpus_metric


def _identity(x):
    return x
