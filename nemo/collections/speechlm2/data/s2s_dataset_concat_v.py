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
import re

import torch
import torch.utils.data

from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.data.force_align import ForceAligner
from nemo.utils import logging


class DuplexS2SDatasetConcatV(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

        force_align_user_text (bool, optional):
            If True, performs force alignment on user audio segments to generate word-level timestamps.
            Only applies to supervision turns where speaker.role is "user". Defaults to False.

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
        - When force_align_user_text is enabled, user audio segments are
          force-aligned using wav2vec2 to generate word-level timestamps, which are then
          converted to frame-level token positions for more precise alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        word_align_position: str = 'left',
        predict_user_text: bool = False,
        cfg: dict = None,
        model_cfg: dict = None,
        force_align_device: str = None,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.word_align_position = word_align_position
        self.predict_user_text = predict_user_text
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.force_align_user_text = self.model_cfg.get("force_align_user_text", False) if self.model_cfg is not None else None
        self.force_align_user_text = True
        self.force_align_asr_model_path = self.model_cfg.get("force_align_asr_model_path", None) if self.model_cfg is not None else None
        self.force_align_device = force_align_device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize force aligner if needed
        self.force_aligner = None
        if self.force_align_user_text:
            self.force_aligner = ForceAligner(device=self.force_align_device, frame_length=self.frame_length)

        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)
        source_audio, decode_source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        vals = [float(c.custom['src_duration'])*self.source_sample_rate for c in cuts]
        source_audio_lens = torch.tensor(vals, dtype=decode_source_audio_lens.dtype, device=decode_source_audio_lens.device)
        if cuts[0].custom.get('target_audio') is not None:
            target_audio, target_audio_lens = collate_audio(
                cuts.resample(self.target_sample_rate), recording_field="target_audio"
            )
            
        else:
            target_audio, target_audio_lens = None, None
            
        target_tokens, target_token_lens = collate_token_channel(
                                            cuts, self.tokenizer, self.frame_length, roles=self.output_roles, bos_id=self.tokenizer.bos, eos_id=self.tokenizer.eos, remove_timestamps=True)

        if self.force_align_user_text:
            self.force_aligner.batch_force_align_user_audio(cuts, source_sample_rate=self.source_sample_rate)

        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, 
            roles=self.input_roles, 
            bos_id=self.tokenizer.text_to_ids('^')[0], 
            eos_id=self.tokenizer.text_to_ids('$')[0], 
            word_align_position=self.word_align_position, 
            remove_timestamps=not self.predict_user_text, 
            user_bos_id=self.tokenizer.text_to_ids('^')[0], 
            agent_bos_id=self.tokenizer.bos, 
            threshold=self.cfg.get("eou_threshold", None) if self.cfg is not None else None, 
            eos_buffer=self.cfg.get("eos_buffer", None) if self.cfg is not None else None
        )
        # extract target speaker first turn audio to uses for speaker conditioning
        # target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
        #     cuts.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
        # )
        first_turn_audio, first_turn_audio_lens = collate_first_turn_audio_source(
            cuts.resample(self.target_sample_rate), roles=self.input_roles
        )

        return {
            "sample_id": [str(cut.id) for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "decode_source_audio_lens": decode_source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "source_texts": [
                " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.input_roles) for cut in cuts
            ],
            "target_texts": [
                " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "all_texts": [
                " ".join(_strip_timestamps(s.text) for s in cut.supervisions) for cut in cuts
            ],
            "first_turn_audio": first_turn_audio,
            "first_turn_audio_lens": first_turn_audio_lens,
            "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in cuts],
        }


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_first_turn_audio_source(
    cuts: CutSet,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_audio()
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    bos_id: int = None,
    eos_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
    user_bos_id: int = None,
    agent_bos_id: int = None,
    threshold: int = None,
    eos_buffer: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, word_align_position=word_align_position, remove_timestamps=remove_timestamps, user_bos_id=user_bos_id,agent_bos_id=agent_bos_id, threshold=threshold, eos_buffer=eos_buffer)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
        cut: Cut,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
        pad_id: int = -1,
        bos_id: int = None,
        eos_id: int = None,
        word_align_position: str = 'left',
        remove_timestamps: bool = False,
        user_bos_id: int = None,
        agent_bos_id: int = None,
        threshold: int = None,
        eos_buffer: int = None,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    count = 0
    for supervision in cut.supervisions:
        if supervision.speaker in roles:

            start_pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if start_pos >= len(tokens):  # Changed from > to >= for robustness
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {start_pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            available_frames_for_text = eospos - start_pos

            if count == 0:
                text = supervision.text
                # text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))
                count += 1
            else:
                text = " " + supervision.text
                # text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(" " + supervision.text))

            # Use different bos_id for user and agent
            text_ids = torch.as_tensor([bos_id] + _text_to_ids(text, tokenizer, available_frames_for_text=available_frames_for_text, word_align_position=word_align_position, remove_timestamps=remove_timestamps, pad_id=pad_id, user_bos_id=user_bos_id, user_eos_id=agent_bos_id, threshold=threshold, eos_buffer=eos_buffer))

            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = start_pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - start_pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = start_pos + len(text_ids)  

            try:
                tokens[start_pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {start_pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # if eospos < len(tokens):
            #     tokens[eospos] = tokenizer.eos
    # TODO: add speech eos at the end of the sequence and use text eos at the end of the tokens
    # if endpos < len(tokens):
    #     tokens[endpos] = tokenizer.eos
    # else:
    if eospos < len(tokens):
        tokens[eospos] = tokenizer.eos
    else:
        tokens[-1] = tokenizer.eos

    return tokens

def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces

def _insert_eos_to_long_pad_segments(text_ids, pad_id, user_eos_id, user_bos_id, threshold=12, eos_buffer=12):
    """
    In text_ids, for any segment of continuous pad_id longer than threshold,
    set the last id of that segment to user_eos_id, ignoring beginning and ending paddings.
    """
    if user_eos_id is None or pad_id is None or not isinstance(text_ids, list) or len(text_ids) == 0:
        return text_ids

    # Find the first and last non-pad_id indices
    first_nonpad = next((i for i, x in enumerate(text_ids) if x != pad_id), None)
    last_nonpad = next((i for i, x in reversed(list(enumerate(text_ids))) if x != pad_id), None)
    if first_nonpad is None or last_nonpad is None or last_nonpad <= first_nonpad:
        return text_ids

    i = first_nonpad
    while i <= last_nonpad:
        if text_ids[i] == pad_id:
            seg_start = i
            while i <= last_nonpad and text_ids[i] == pad_id:
                i += 1
            seg_end = i  # exclusive
            seg_len = seg_end - seg_start
            if seg_len > threshold:
                text_ids[seg_start + eos_buffer] = user_eos_id
                text_ids[seg_end - 1] = user_bos_id
        else:
            i += 1
    return text_ids
    
def _text_to_ids(text: str, tokenizer: TokenizerSpec,
                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
                 available_frames_for_text=None,
                 word_align_position='left',
                 remove_timestamps=False,
                 pad_id=None,
                 user_bos_id=None,
                 user_eos_id=None,
                 threshold=None,
                 eos_buffer=None):
    if not remove_timestamps and re.compile(_TIMESTAMP_PATTERN_STR).search(text):
        text_ids = _text_with_timestamps_to_ids(text, tokenizer, _TIMESTAMP_PATTERN_STR, available_frames_for_text, word_align_position)
        if threshold is not None and threshold > 0:
            text_ids = _insert_eos_to_long_pad_segments(text_ids, pad_id, user_eos_id, user_bos_id, threshold=threshold, eos_buffer=eos_buffer)
    else:
        _TIMESTAMP_PATTERN = re.compile(_TIMESTAMP_PATTERN_STR)
        text = _TIMESTAMP_PATTERN.sub("", text)
        # Remove extra spaces between words
        text = " ".join(text.strip().split())
        text_ids = tokenizer.text_to_ids(text)
    return text_ids


def _text_with_timestamps_to_ids(text: str, tokenizer: TokenizerSpec,
                                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
                                 available_frames_for_text=None,
                                 word_align_position='left') -> list[int]:
    text_ids = []
    text_ids, start_times, end_times, word_lens = _extract_text_and_time_tokens(text, tokenizer, _TIMESTAMP_PATTERN_STR)
    text_ids_with_timestamps = _expand_text_with_timestamps_and_word_lengths(text_ids, word_lens, start_times, end_times, available_frames_for_text, frame_rate=0.08, pad_id=get_pad_id(tokenizer), word_align_position=word_align_position)
    return text_ids_with_timestamps


def _extract_text_and_time_tokens(text, tokenizer: TokenizerSpec,
                                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>"):
    # Find all time tokens
    time_tokens = re.findall(_TIMESTAMP_PATTERN_STR, text)
    start_time = [int(time_tokens[i]) for i in range(0, len(time_tokens), 2)]
    end_time = [int(time_tokens[i]) for i in range(1, len(time_tokens), 2)]
    # Remove all time tokens to isolate words
    words = re.sub(_TIMESTAMP_PATTERN_STR, '', text).split()
    # Process each word, tokenize it, and calculate token lengths
    text_ids = []
    word_lens = []
    for i, word in enumerate(words):
        word_with_space = word if i == 0 else ' ' + word
        word_ids = tokenizer.text_to_ids(word_with_space)
        word_len = len(word_ids)
        text_ids.extend(word_ids)
        word_lens.append(word_len)
    return text_ids, start_time, end_time, word_lens


def _expand_text_with_timestamps_and_word_lengths(
        text_ids, word_lens, start_time, end_time, available_frames_for_text, frame_rate=0.08, pad_id=None, word_align_position='left'
    ):    
    """
    Expand word tokens according to start time tokens and word lengths for a batch of sequences.

    Args:
    - word_tokens: List of text ids w/o timestamps
    - word_lens: List of word lengths
    - start_time: List of start times
    - end_time: List of end times
    - available_frames_for_text: Maximum number of frames for text
    - frame_rate: Frame rate resolution
    - pad_id: Padding ID to use for empty positions in the tensor

    Returns:
    - text ids with word-level timestamps
    """

    def discretize_time(start_token, speech_frame_rate=0.08, timestamp_frame_rate=0.08):
        return int(start_token * timestamp_frame_rate / speech_frame_rate)

    if pad_id is None:
        raise ValueError("pad_id must be provided.")

    max_length = available_frames_for_text

    # Create the empty tensor with pad_id as the default value
    text_ids_with_timestamps = [pad_id] * max_length

    # Populate ids of each word starting at start_idx and ending at end_idx
    cur_word_idx = 0  # Start frame index of current word
    for word_idx, word_len in enumerate(word_lens):
        start_idx = discretize_time(start_time[word_idx], speech_frame_rate=frame_rate)
        end_idx = discretize_time(end_time[word_idx], speech_frame_rate=frame_rate)
        if word_align_position == 'left':
            end_idx = min(start_idx + word_len, end_idx)
        elif word_align_position == 'right':
            start_idx = max(start_idx, end_idx - word_len)
        else:
            raise ValueError(f"Unknown word_align_position: {word_align_position}")

        # Get ids of a single word
        word_ids = text_ids[cur_word_idx : cur_word_idx + word_len]

        # Populate a single word
        for i in range(start_idx, end_idx + 1):  # End inclusive at word level
            if i - start_idx < len(word_ids) and i < max_length:
                token_id = word_ids[i - start_idx]
                text_ids_with_timestamps[i] = token_id

        # Move to the next word in the concatenated word tokens
        cur_word_idx += word_len

    return text_ids_with_timestamps