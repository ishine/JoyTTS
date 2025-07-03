# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw
from transformers import AutoTokenizer
from .dataset_llm import preprocess


AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def parquet_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    # print('~'*20,'parquet_opener')
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                        continue
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':
                        # NOTE do not return sample directly, must initialize a new dict
                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    # print('~'*20,'filter')
    for sample in data:
        con_num = len(sample['conversations'])
        sample['filter'] = [True]*int(con_num/2) #当前对话，音频的个数（QA对数）,记录音频是否满足条件
        for i in range(0,con_num,2):
            wavinfo = sample['wavinfo'][int(i/2)]
            if wavinfo is None:
                sample['filter'][int(i/2)] = False
                continue
            # speech_token = sample['speech_token'][int(i/2)]
            # sample['wav'] is torch.Tensor, we have 100 frames every second
            num_frames = wavinfo['wav_duration'] * 100
            if num_frames < min_length or num_frames > max_length:
                sample['filter'][int(i/2)] = False
                
        if sum(sample['filter']) == 0:
            continue
                
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
        sample['speech'] = waveform
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample['speech_feat'] = mat
        yield sample


def compute_f0(data, sample_rate, hop_size, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    frame_period = hop_size * 1000 / sample_rate
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        _f0, t = pw.harvest(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
        if sum(_f0 != 0) < 5:  # this happens when the algorithm fails
            _f0, t = pw.dio(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)  # if harvest fails, try dio
        f0 = pw.stonemask(waveform.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate)
        f0 = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=sample['speech_feat'].shape[0], mode='linear').view(-1)
        sample['pitch_feat'] = f0
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode='train'):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample

def tokenize_llm(data, tokenizer_path, mode='train'):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    # print('~'*20,'tokenize_llm')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for sample in data:
        #此时的sample是单个样本
        assert 'conversations' in sample
        images_dict = {}
        reference = sample['reference'] if 'reference' in sample else None
        conversations_list = sample['conversations'].tolist()
        reference_dict = {'role':'user','content':reference}
        # if reference is not None:
        #     conversations_list.insert(0, reference_dict)
        slice_config = {
                        "patch_size": 14,
                        "max_slice_nums": 9,
                        "scale_resolution": 448,
                    }
        ret = preprocess(
            images_dict,
            conversations_list,
            tokenizer,
            query_nums=64,
            slice_config=slice_config,
            llm_type="qwen",
            patch_size=14,
            batch_vision=True,
            max_length=2048,
        )
        #ret = dict(
            #     input_ids=ret["input_ids"],
            #     position_ids=ret["position_ids"],
            #     labels=ret["target"],
            #     attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            #     pixel_values=ret["pixel_values"],
            #     tgt_sizes=ret["tgt_sizes"],
            #     image_bound=ret["image_bound"],
            # )
        # print([len(conversations_list), len(ret["input_ids"])])
        sample['text_token'] = ret["input_ids"]
        sample['target'] = ret["target"]
        sample['position_ids'] = ret["position_ids"]
        sample['attention_mask'] = torch.ones_like(ret["input_ids"], dtype=torch.bool)
        sample['pixel_values'] = ret["pixel_values"]
        sample['tgt_sizes'] = ret["tgt_sizes"]
        sample['image_bound'] = ret["image_bound"]
        yield sample
    

def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    # print('~'*20,'shuffle')
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """
    # print('~'*20,'sort')

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            # buf.sort(key=lambda x: x['text_token'].size(0))
            buf.sort(key=lambda x: x['text_token'].size(0), reverse=True)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    # buf.sort(key=lambda x: x['text_token'].size(0))
    buf.sort(key=lambda x: x['text_token'].size(0), reverse=True)
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    # print('~'*20,'batch')
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
    
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train', gan=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    # print('~'*20,'padding')
    for sample in data:
        #此时的sample已经经过了batch函数，是一个batch的样本
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['text_token'].size(0) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)

        utts = [sample[i]['utt'] for i in order]
        filter = [sample[i]['filter'] for i in order]
        # speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
        # speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
        # speech = pad_sequence(speech, batch_first=True, padding_value=0)
        speech_token = [torch.tensor(sample[i]['speech_token'][j]) for i in order for j in range(len(sample[i]['filter'])) if sample[i]['filter'][j]]
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        if speech_token == []:
            speech_token = torch.tensor(speech_token)
        else:
            speech_token = pad_sequence(speech_token,
                                        batch_first=True,
                                        padding_value=0)
        # speech_feat = [sample[i]['speech_feat'] for i in order]
        # speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        # speech_feat = pad_sequence(speech_feat,
        #                            batch_first=True,
        #                            padding_value=0)
        # text = [sample[i]['text'] for i in order]

        # text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        # text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        # text_token = pad_sequence(text_token, batch_first=True, padding_value=0)

        text_token = [sample[i]['text_token'] for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)

        attention_mask = [sample[i]['attention_mask'] for i in order]
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        target = [sample[i]['target'] for i in order]
        target = pad_sequence(target, batch_first=True, padding_value=-100)

        position_ids = [sample[i]['position_ids'] for i in order]
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)

        # utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        # spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)


        pixel_values = [sample[i]['pixel_values'] for i in order]
        image_bound = [sample[i]['image_bound'] for i in order]
        tgt_sizes = [sample[i]['tgt_sizes'] for i in order]

        batch = {
            "utts": utts,
            "filter":filter,
            # "speech": speech,
            # "speech_len": speech_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            # "speech_feat": speech_feat,
            # "speech_feat_len": speech_feat_len,
            # "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "input_ids": text_token,
            "attention_mask": attention_mask,
            "target": target,
            "position_ids": position_ids,
            # "utt_embedding": utt_embedding,
            # "spk_embedding": spk_embedding,
            "pixel_values": pixel_values,
            "image_bound": image_bound,
            "tgt_sizes": tgt_sizes,
        }
        yield batch
