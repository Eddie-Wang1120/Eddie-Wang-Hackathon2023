from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import tiktoken
import base64
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from collections import OrderedDict
import zlib

from tokenizer import Tokenizer, LANGUAGES, TO_LANGUAGE_CODE

from tensorrt_llm.runtime.session import Session, TensorInfo
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

CHUNK_LENGTH = 30

@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation

@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan

class SequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]



class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError

class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf



class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf

class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError

class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()

class WhisperDecoding:
    def __init__(
        self, 
        decoder_serialize_path, 
        cross_kv_serialize_path,
        dtype, 
        multilingual, 
        language, 
        task,
        model_metadata,
        positional_embedding
        ):
        
        self.deocder_session = self.get_session(decoder_serialize_path)
        self.crossattn_session = self.get_session(cross_kv_serialize_path)
        self.dtype = dtype
        self.tokenizer = self.get_tokenizer(multilingual, language, task)

        self.sot_sequence = self.tokenizer.sot_sequence
        self.initial_tokens = tuple(list(self.sot_sequence))
        self.initial_token_length = len(self.initial_tokens)
        self.tokens = torch.tensor([self.initial_tokens]).repeat(model_metadata['n_audio'], 1)
        self.sot_index = self.initial_tokens.index(self.tokenizer.sot)
        self.options = DecodingOptions()
        
        self.n_audio = model_metadata['n_audio']
        self.n_layer = model_metadata['n_text_layer']
        self.n_ctx = model_metadata['n_text_ctx']
        self.n_group = self.options.beam_size or self.options.best_of or 1
        self.sample_len: int = self.options.sample_len or model_metadata['n_text_ctx'] // 2
        self.sample_begin: int = len(self.initial_tokens)
        
        self.positional_embedding = positional_embedding

        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not self.options.without_timestamps:
            precision = CHUNK_LENGTH / model_metadata['n_audio_ctx']  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if self.options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    self.tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )
        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(self.options.length_penalty)

        # if self.options.beam_size is not None:
        #     self.decoder = BeamSearchDecoder(
        #         self.options.beam_size, self.tokenizer.eot, self.inference, self.options.patience
        #     )
        # else:
        self.decoder = GreedyDecoder(self.options.temperature, self.tokenizer.eot)
        
        self.kv_cache = {}
        self.hooks = []

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    @lru_cache(maxsize=None)
    def get_encoding(self, name: str = "gpt2"):
        vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in open(vocab_path) if line)
        }
        n_vocab = len(ranks)
        special_tokens = {}

        specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
        ]

        for token in specials:
            special_tokens[token] = n_vocab
            n_vocab += 1

        return tiktoken.Encoding(
            name=os.path.basename(vocab_path),
            explicit_n_vocab=n_vocab,
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )

    @lru_cache(maxsize=None)
    def get_tokenizer(
        self,
        multilingual: bool,
        language: Optional[str] = None,
        task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    ) -> Tokenizer:
        if language is not None:
            language = language.lower()
            if language not in LANGUAGES:
                if language in TO_LANGUAGE_CODE:
                    language = TO_LANGUAGE_CODE[language]
                else:
                    raise ValueError(f"Unsupported language: {language}")
        if multilingual:
            encoding_name = "multilingual"
            language = language or "en"
            task = task or "transcribe"
        else:
            encoding_name = "gpt2"
            language = None
            task = None

        encoding = self.get_encoding(name=encoding_name)

        return Tokenizer(encoding=encoding, language=language, task=task)

    def get_session(self, serialize_path):
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        # print(session._print_io_info())
        return session 

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            # print("prefix"+prefix)
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            # print("prompt"+prompt)
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def xa2cross_key_value(self, xa):
        inputs = OrderedDict()
        output_list = []
        inputs.update({'xa': xa.type(torch.float16).contiguous()})
        output_list.append(TensorInfo('xa', str_dtype_to_trt("float16"), xa.shape))

        output_info = self.crossattn_session.infer_shapes(output_list)
        
        logger.debug(f'output info {output_info}')
        outputs = {
                t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
                for t in output_info
        }

        # Execute model inference
        stream = torch.cuda.current_stream()
        ok = self.crossattn_session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
    
        cross_past_key_value = []
        for i in range(self.n_layer):
            cross_past_key_value.append(outputs['cross_present_key_value_'+str(i)])
        
        return cross_past_key_value

    def decode(self, x, cross_past_key_value, past_key_value=None):

        inputs = OrderedDict()
        output_list = []
        inputs.update({'x' : x.type(torch.int32).contiguous()})
        output_list.append(TensorInfo('x', str_dtype_to_trt("int32"), x.shape))
    
        input_lengths = torch.tensor((1,), dtype=torch.int32, device='cuda')
        inputs.update({'input_lengths':input_lengths})
        output_list.append(TensorInfo('input_lengths', str_dtype_to_trt("int32"), input_lengths.shape))
    
        max_input_length = torch.tensor((1,), dtype=torch.int32, device='cuda')
        inputs.update({'max_input_length':max_input_length})
        output_list.append(TensorInfo('max_input_length', str_dtype_to_trt("int32"), max_input_length.shape))
    
        mask = torch.empty(self.n_ctx, self.n_ctx, dtype=torch.float32).fill_(-50000).triu_(1).to("cuda")
        mask = mask[:x.shape[-1], :x.shape[-1]]
        mask = mask.contiguous()
        inputs.update({'mask':mask})
        output_list.append(TensorInfo('mask', str_dtype_to_trt("float32"), mask.shape))
    
        positional_embedding = self.positional_embedding.to("cuda")
        offset = past_key_value[0].shape[3] if past_key_value else 0

        positional_embedding = positional_embedding[offset : offset + x.shape[-1]]
        inputs.update({'positional_embedding':positional_embedding.type(torch.float16).contiguous()})
        output_list.append(TensorInfo('positional_embedding', str_dtype_to_trt("float16"), positional_embedding.shape))
    
        if past_key_value is None:
            past_key_value = []
            for i in range(self.n_layer):
                past_key_value.append(
                    torch.ones((1,), dtype=torch.float16).to('cuda')
                )
                inputs.update({'past_key_value_'+str(i): past_key_value[i].contiguous()})
                output_list.append(TensorInfo('past_key_value_'+str(i), str_dtype_to_trt('float16'), (1, 2, 20, 0, 64)))
        else:
            for i in range(self.n_layer):
                inputs.update({'past_key_value_'+str(i): past_key_value[i].contiguous()})
                output_list.append(TensorInfo('past_key_value_'+str(i), str_dtype_to_trt('float16'), past_key_value[i].shape))

        for i in range(self.n_layer):
            inputs.update({'cross_past_key_value_'+str(i): cross_past_key_value[i].contiguous()})
            output_list.append(TensorInfo('cross_past_key_value_'+str(i), str_dtype_to_trt('float16'), cross_past_key_value[i].shape))

        output_info = self.deocder_session.infer_shapes(output_list)
        
        logger.debug(f'output info {output_info}')
        outputs = {
                t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
                for t in output_info
        }

        # Execute model inference
        stream = torch.cuda.current_stream()
        ok = self.deocder_session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
    
        past_key_value = []
        for i in range(self.n_layer):
            past_key_value.append(outputs['present_key_value_'+str(i)])
        
        logits = outputs['output']
        
        return logits, past_key_value

    def torch_detect_language(self, model, audio_features):
        with torch.no_grad():
            languages = [self.options.language] * audio_features.shape[0]
            language_probs = None

            if self.options.language is None or self.options.task == "lang_id":
                single = audio_features.ndim == 2
                if single:
                    audio_features = audio_features.unsqueeze(0)

                # forward pass using a single token, startoftranscript
                n_audio = audio_features.shape[0]
                x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(audio_features.device)  # [n_audio, 1]
            
                logits = model.logits(x, audio_features)[:, 0]
            
                mask = torch.ones(logits.shape[-1], dtype=torch.bool)
                mask[list(self.tokenizer.all_language_tokens)] = False
                logits[:, mask] = -np.inf
                language_tokens = logits.argmax(dim=-1)
                language_token_probs = logits.softmax(dim=-1).cpu()
                language_probs = [
                    {
                        c: language_token_probs[i, j].item()
                        for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
                    }
                    for i in range(n_audio)
                ]
                if single:
                    language_tokens = language_tokens[0]
                    language_probs = language_probs[0]

                languages = [max(probs, key=probs.get) for probs in language_probs]
                if self.options.language is None:
                    self.tokens[:, self.sot_index + 1] = language_tokens  # write language tokens

        return languages, language_probs        

    def detect_language(self, audio_features):
        languages = [self.options.language] * audio_features.shape[0]
        language_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            single = audio_features.ndim == 2
            if single:
                audio_features = audio_features.unsqueeze(0)

            # forward pass using a single token, startoftranscript
            n_audio = audio_features.shape[0]
            x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(audio_features.device)  # [n_audio, 1]
            
            cross_past_key_value = self.xa2cross_key_value(audio_features)
            logits, _ = self.decode(x, cross_past_key_value)
            logits = logits[:, 0]
            
            mask = torch.ones(logits.shape[-1], dtype=torch.bool)
            mask[list(self.tokenizer.all_language_tokens)] = False
            logits[:, mask] = -np.inf
            language_tokens = logits.argmax(dim=-1)
            language_token_probs = logits.softmax(dim=-1).cpu()
            language_probs = [
                {
                    c: language_token_probs[i, j].item()
                    for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
                }
                for i in range(n_audio)
            ]
            if single:
                language_tokens = language_tokens[0]
                language_probs = language_probs[0]

            languages = [max(probs, key=probs.get) for probs in language_probs]
            if self.options.language is None:
                self.tokens[:, self.sot_index + 1] = language_tokens  # write language tokens

        return languages, language_probs
    
    def torch_main_loop(self, model, audio_features):
        with torch.no_grad():
            tokens = self.tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
            n_batch = tokens.shape[0]
            sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
            no_speech_probs = [np.nan] * n_batch
        
            for i in range(self.sample_len):
                if not self.kv_cache:
                    self.kv_cache, self.hooks = model.install_kv_cache_hooks()
            
                past_tokens = tokens
                if tokens.shape[-1] > self.initial_token_length:
                    # only need to use the last token except in the first forward pass
                    tokens = tokens[:, -1:]
                
                logits = model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
                tokens = past_tokens
            
                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
            
                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)
            
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
            
                if completed or tokens.shape[-1] > self.n_ctx:
                    break
            
        return tokens, sum_logprobs, no_speech_probs
    
    def main_loop(self, audio_features):
        tokens = self.tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        
        past_key_value = None
        
        cross_past_key_value = self.xa2cross_key_value(audio_features)
        
        for i in range(self.sample_len):
            past_tokens = tokens
            if tokens.shape[-1] > self.initial_token_length:
                # only need to use the last token except in the first forward pass
                tokens = tokens[:, -1:]
                
            logits, past_key_value = self.decode(tokens, cross_past_key_value, past_key_value)
            tokens = past_tokens
            
            if (
                i == 0 and self.tokenizer.no_speech is not None
            ):  # save no_speech_probs
                probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
            
            # now we need to consider the logits at the last token only
            logits = logits[:, -1]

            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in self.logit_filters:
                logit_filter.apply(logits, tokens)
            
            tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
            
            if completed or tokens.shape[-1] > self.n_ctx:
                break
            
        return tokens, sum_logprobs, no_speech_probs

    def compression_ratio(self, text) -> float:
        text_bytes = text.encode("utf-8")
        return len(text_bytes) / len(zlib.compress(text_bytes))

    def post_process(self, tokens, sum_logprobs, no_speech_probs, audio_features, languages):
        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == self.n_audio

        tokens = tokens.reshape(self.n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(self.n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == self.tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [self.tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=self.compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]