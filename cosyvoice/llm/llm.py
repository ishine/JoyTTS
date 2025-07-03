# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import queue
import random
import time
import threading
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask
import time
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
import re
import copy


# from cosyvoice.llm.extract_hidden import AgentChat

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)
        # config = AutoConfig.from_pretrained(pretrain_path)
        # self.model = AutoModel.from_config(config)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache

class Qwen2Chat(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()

        if pretrain_path != "":
            self.model = AutoModel.from_pretrained(pretrain_path, trust_remote_code=True,
                    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
            self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path, trust_remote_code=True)
            
            
    def forward(self, batch):
        text_token, text_token_len, hidden_states,loss_chat = self.model.get_hidden_forward(batch)
        return text_token, text_token_len, hidden_states,loss_chat




class Qwen2LM(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            chat: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)

        self.llm_projctor = nn.Linear(3584, llm_output_size)
        # self.chatsystem = AutoModel.from_pretrained(chat_path, trust_remote_code=True) #OOM

        # chat_path = "/mnt/afs/zhoufangru/agent/end2end/pretrained_models/MiniCPM-o-2_6"
        # if chat_path != '':

            # from prepare_data_sever.prepare_llmanswer_offline import postprocess
            # self.chatsystem = chatsystem
            # self.postprocess = postprocess

            # self.chatsystem = AutoModel.from_pretrained(chat_path, trust_remote_code=True,
            #         attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
            # self.tokenizer = AutoTokenizer.from_pretrained(chat_path, trust_remote_code=True)


        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        self.chat = chat
        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.loss_fct = nn.CrossEntropyLoss()
        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}
        self.lock = threading.Lock()
        
        
    def prepare_lm_input_target(self, text_token, text_token_emb, text_token_len, speech_token, speech_token_emb, speech_token_len):
        device = text_token.device
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)
        for i in range(len(text_token)):
            # bistream sequence
            if random.random() < 0.5 and speech_token_len[i] / text_token_len[i] > self.mix_ratio[1] / self.mix_ratio[0]:
                this_lm_target, this_lm_input = [], []
                this_lm_target.append(IGNORE_ID)
                # this_lm_input.append(self.llm_embedding.weight[self.sos_eos].reshape(1, -1))
                this_lm_input.append(self.llm_embedding(torch.tensor([self.sos_eos]).to(device)).reshape(1, -1))
                for j in range(((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()):
                    this_text_token = text_token[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]].tolist()
                    this_speech_token = speech_token[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]].tolist()
                    if len(this_text_token) == self.mix_ratio[0]:
                        assert len(this_speech_token) == self.mix_ratio[1]
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.speech_token_size + 2)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]])
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]])
                    else:
                        this_lm_target += [-1] * len(this_text_token)
                        this_lm_target += speech_token[i][j * self.mix_ratio[1]:].tolist()
                        this_lm_target.append(self.speech_token_size)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]:])
                        # this_lm_input.append(self.llm_embedding.weight[self.task_id].reshape(1, -1))
                        this_lm_input.append(self.llm_embedding(torch.tensor([self.task_id]).to(device)).reshape(1, -1))
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]:])
                this_lm_target, this_lm_input = torch.tensor(this_lm_target), torch.concat(this_lm_input, dim=0)
            # unistream sequence
            else:
                try:
                    this_lm_target = torch.tensor([IGNORE_ID] * (1 + text_token_len[i]) + speech_token[i].tolist() + [self.speech_token_size])
                except:
                    import ipdb; ipdb.set_trace()
                # this_lm_input = torch.concat([self.llm_embedding.weight[self.sos_eos].reshape(1, -1), text_token_emb[i],
                #                               self.llm_embedding.weight[self.task_id].reshape(1, -1), speech_token_emb[i]], dim=0)
                this_lm_input = torch.concat([self.llm_embedding(torch.tensor([self.sos_eos]).to(device)).reshape(1, -1), text_token_emb[i],
                                              self.llm_embedding(torch.tensor([self.task_id]).to(device)).reshape(1, -1), speech_token_emb[i]], dim=0)
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        return lm_target, lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        #batch ['utts', 'speech_token', 'speech_token_len', 'speech_feat', 'speech_feat_len', 'text', 'text_token', 'text_token_len', 'utt_embedding', 'spk_embedding', 'embedding']
        loss_chat = None
        if "hidden_states" in batch:
            #冻结llm中的chat
            text_token = batch['text_token'].to(device)
            text_token_len = batch['text_token_len'].to(device)
            hidden_states = batch['hidden_states'].to(device)
            text_token_emb = self.llm.model.model.embed_tokens(text_token)
            text_token_emb += self.llm_projctor(hidden_states)
            # batch['text'] = batch['text']
        elif 'input_ids' in batch:
            #训练完整的llm（chat+tts）
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['position_ids'] = batch['position_ids'].to(device)
            batch['target'] = batch['target'].to(device)
            
            # with torch.no_grad():
            text_token, text_token_len, hidden_states,loss_chat = self.chat(batch)
            
            if text_token is not None:
                # 1. encode text_token
                text_token_emb = self.llm.model.model.embed_tokens(text_token)
                hidden_states = hidden_states.to(torch.float32)
                # text_token_emb = self.llm_projctor(hidden_states)
                text_token_emb += self.llm_projctor(hidden_states)
        else:
            text_token = batch['text_token'].to(device)
            text_token_len = batch['text_token_len'].to(device)
            # 1. encode text_token
            text_token_emb = self.llm.model.model.embed_tokens(text_token)
        # print([batch['text_token'].shape, batch['filter'], batch['speech_token'].shape])
        
        ## tts
        speech_token = batch['speech_token'].to(text_token.dtype).to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        tts_batch_size = 1
        tts_id = random.randint(0,speech_token.size(0)-tts_batch_size)
        text_token = text_token[tts_id:tts_id+tts_batch_size]
        text_token_len = text_token_len[tts_id:tts_id+tts_batch_size]
        hidden_states = hidden_states[tts_id:tts_id+tts_batch_size]
        speech_token = speech_token[tts_id:tts_id+tts_batch_size]
        speech_token_len = speech_token_len[tts_id:tts_id+tts_batch_size]
        
        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # print(f"batch size {speech_token_emb.shape[0]}")

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(text_token, text_token_emb, text_token_len, speech_token, speech_token_emb, speech_token_len)
        
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss_tts = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)
        
        loss = loss_tts + loss_chat if loss_chat is not None else loss_tts
        # loss = loss_tts 
        # print(f'loss:{loss}, loss_chat:{loss_chat}, loss_tts:{loss_tts}')
        
        return {'loss': loss, 'acc': acc}
    
    @torch.inference_mode()
    def inference_end2end_cache(
            self,
            text: str,
            prompt_text: str,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        
        device = prompt_speech_token.device
        # print(prompt_text, text)
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # '''
        user_question = [{'role': 'user', 'content': ["请重复以下文本。",prompt_text]}]
        prompt_text_answer, prompt_text_token, prompt_text_len, prompt_hidden_states = self.chat.model.get_hidden(
                    msgs=user_question,
                    tokenizer=self.chat.tokenizer,
                    max_new_tokens=128,
                )
        # print(user_question, prompt_text_answer)
        prompt_text_token_emb = self.llm.model.model.embed_tokens(prompt_text_token)
        prompt_hidden_states = prompt_hidden_states.to(torch.float32)
        prompt_text_token_emb += self.llm_projctor(prompt_hidden_states)

        # 3. concat llm_input
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, prompt_text_token_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        _, cache = self.llm.forward_one_step(lm_input,
                                                    masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                    cache=None)
        
        # '''
        # cache=None
        time1 = time.time()
        # user_question = [{'role': 'user', 'content': ["请重复以下文本。",text]}]
        user_question = [{'role': 'user', 'content': [text]}]

        text_answer, text_token, text_len, hidden_states = self.chat.model.get_hidden(
                    msgs=user_question,
                    tokenizer=self.chat.tokenizer,
                    max_new_tokens=128,
                )
        suc = 0
        # suc, text_unpad_ori, text_answer, text_token, text_len, hidden_states = self.postprocess(text_answer, text_token, text_len, hidden_states)
        # print(text_unpad_ori, text_answer)
        # text_answer = [text_answer[0][:4]]
        # text_token = text_token[:,:3]
        # text_len = 3
        # hidden_states = hidden_states[:,:3]
        # import ipdb; ipdb.set_trace()
        time2 = time.time()
        if suc==0:
            text_token_emb = self.llm.model.model.embed_tokens(text_token)
            hidden_states = hidden_states.to(torch.float32)
            hidden_states_emb = self.llm_projctor(hidden_states)
            # print(hidden_states_emb)
            text_token_emb += hidden_states_emb
        else:
            text_token_emb = self.llm.model.model.embed_tokens(text_token)
        lm_input = torch.concat([sos_eos_emb, text_token_emb, task_id_emb], dim=1)
        time3 = time.time()
        # print(time3-time2, time2-time1)

        # 4. cal min/max_length
        min_len = int((text_len) * min_token_text_ratio)
        max_len = int((text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        for i in range(max_len):
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            try:
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            except:
                if i ==0:
                    yield 0, text_answer
                break

            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            # print('!'*20, top_ids)
            yield top_ids, text_answer
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        
    def postprocess(self, text_unpad, text_token, text_token_len, hidden_states, max_text_len=50):
        
        sign_dict = {'。\n\n':3407, '。\n':8997, '。”':32945, '。“':53647, '”。':55807, '）。':74276, '。':1773, '！\n\n':17701, '！':6313, '？':11319,'[?]':30,'!':0, 
                    '：\n':28311, '：':5122, ':\n':510, ':':25}
        text_unpad_ori = copy.deepcopy(text_unpad)
        searched_ids = set()
        while len(text_unpad[0]) > max_text_len:
            for sign_id, sign in enumerate(sign_dict):
                search_res = re.search(sign, text_unpad[0])
                if search_res is None:
                    continue
                search_ids = search_res.span()
                if search_ids[1]==len(text_unpad[0]):
                    for i in range(search_ids[0],search_ids[1]):
                        searched_ids.add(i)
                    continue
                if search_ids[0] in searched_ids:
                    continue
                elif search_ids[1] in searched_ids:
                    for i in range(search_ids[0],search_ids[1]):
                        searched_ids.add(i)
                    continue
                else:
                    for i in range(search_ids[0],search_ids[1]):
                        searched_ids.add(i)
                text_unpad = [text_unpad[0][:search_ids[0]]+'?'] if sign=='[?]' else [text_unpad[0][:search_ids[0]]+sign]
                try:
                    token_id = text_token[0].cpu().numpy().tolist().index(sign_dict[sign])
                    text_token = text_token[:, :token_id+1]
                    # if not text_token[0].cpu().numpy().tolist() == chatsystem.tokenizer(text_unpad)['input_ids'][0]:
                    if not self.chat.tokenizer.decode(text_token[0].cpu().numpy().tolist()) == text_unpad[0]:
                        #判断clip后的text和token是否一致，不一致则返回状态码1
                        # import ipdb; ipdb.set_trace()
                        return 1, text_unpad_ori, text_unpad, text_token, text_token_len, None

                except:
                    # import ipdb; ipdb.set_trace()
                    #大概率是在text找到了sign, 但是token中没有对应的sign_dict[sign]，需要查看为什么，返回状态码2
                    return 2, text_unpad_ori, text_unpad, text_token, text_token_len, None
                text_token_len[0] = token_id+1
                hidden_states = hidden_states[:, :token_id+1]
                break
            if sign_id == len(sign_dict)-1:
                break
        return 0, text_unpad_ori, text_unpad, text_token, text_token_len, hidden_states

    @torch.inference_mode()
    def inference_end2end(
            self,
            text: str,
            prompt_text: str,
            prompt_text_token_emb: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid='',
    ) -> Generator[torch.Tensor, None, None]:
        
        device = prompt_speech_token.device
        
        # 1.prepare input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        cache=None
        user_question = [{'role': 'user', 'content': [text]}]
        text_answer, text_token, text_len, hidden_states = self.chat.model.get_hidden(
                    msgs=user_question,
                    tokenizer=self.chat.tokenizer,
                    max_new_tokens=128,
                )
        suc = 0
        # suc, text_unpad_ori, text_answer, text_token, text_len, hidden_states = self.postprocess(text_answer, text_token, text_len, hidden_states)

        if suc==0:
            text_token_emb = self.llm.model.model.embed_tokens(text_token)
            hidden_states = hidden_states.to(torch.float32)
            hidden_states_emb = self.llm_projctor(hidden_states)
            # print(hidden_states_emb)
            text_token_emb += hidden_states_emb
        else:
            text_token_emb = self.llm.model.model.embed_tokens(text_token)
            
        text_token_emb = torch.concat([prompt_text_token_emb, text_token_emb], dim=1)
            
        lm_input = torch.concat([sos_eos_emb, text_token_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        # 2. cal min/max_length
        min_len = int((text_len) * min_token_text_ratio)
        max_len = int((text_len) * max_token_text_ratio)

        # 3. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token, text_answer

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        if hasattr(self, 'vllm'):
            from vllm import SamplingParams, RequestOutput
            if min_len is not None and max_len is not None:
                sampling_params = SamplingParams(top_k=sampling,
                                                stop_token_ids=self.stop_token_ids,
                                                min_tokens=min_len,
                                                max_tokens=max_len)
            else:
                sampling_params = SamplingParams(top_k=sampling,
                                                stop_token_ids=self.stop_token_ids,)
            with self.lock:
                self.vllm.add_request(uuid, {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16).to(lm_input.device)}, sampling_params)
                self.vllm_output_queue[uuid] = queue.Queue()
            out_tokens = []
            while True:
                with self.lock:
                    if self.vllm_output_queue[uuid].empty() is True:
                        request_outputs: List[RequestOutput] = self.vllm.step()
                        for request_output in request_outputs:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(top_ids)
                if self.vllm_output_queue[uuid].empty() is False:
                    top_ids = self.vllm_output_queue[uuid].get()
                    if top_ids in self.stop_token_ids:
                        break
                    # in stream mode, yield token one by one
                    yield top_ids
                    out_tokens.append(top_ids)
                    if len(out_tokens) == max_len:
                        break
                time.sleep(0.001)
            with self.lock:
                self.vllm_output_queue.pop(uuid)
        else:
            out_tokens = []
            cache = None
            for i in range(max_len):
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                          masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                          cache=cache)
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size:
                    break
                if top_ids > self.speech_token_size:
                    continue
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        use_prompt=True

        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(device)
        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(device)

        if use_prompt:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
            
        text = self.llm.model.model.embed_tokens(text)

        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        if use_prompt:
            lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)
        else:
            lm_input = torch.concat([sos_eos_emb, text, task_id_emb], dim=1)

        if use_prompt:
            min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
            max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        else:
            min_len = int((text_len) * min_token_text_ratio)
            max_len = int((text_len) * max_token_text_ratio)

        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
    @torch.inference_mode()
    def inference_prompt_emb(self, prompt_text):
        # '''
        # prompt_text = ','.join([i for i in prompt_text])
        user_question = [{'role': 'user', 'content': ["请重复以下文本。",prompt_text]}]
        prompt_text_answer, prompt_text_token, prompt_text_len, prompt_hidden_states = self.chat.model.get_hidden(
                    msgs=user_question,
                    tokenizer=self.chat.tokenizer,
                    max_new_tokens=128,
                )
        # print(user_question, prompt_text_answer)
        prompt_text_token_emb = self.llm.model.model.embed_tokens(prompt_text_token)
        prompt_hidden_states = prompt_hidden_states.to(torch.float32)
        prompt_text_token_emb += self.llm_projctor(prompt_hidden_states)
        return prompt_text_token_emb

    @torch.inference_mode()
    def inference_end2end_bistream(
            self,
            text: str,
            prompt_text: str,
            prompt_text_token_emb: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid='',
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_speech_token.device
        
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 1.prepare input
        if prompt_text_token_emb.size(-1)==0 and prompt_text:
            prompt_text_token_emb = self.inference_prompt_emb(prompt_text)

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text_token_emb.dtype).to(device)

        # 2. prefill video/audio chunks
        user_question = [{'role': 'user', 'content': [text]}]
        res = self.chat.model.streaming_prefill(
            session_id=uuid,
            msgs=user_question, 
            tokenizer=self.chat.tokenizer
        )
        # import ipdb; ipdb.set_trace()
        # 3. generate
        result = self.chat.model.streaming_generate(
            session_id=uuid,
            tokenizer=self.chat.tokenizer,
            temperature=0.5,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
        ) #res = {"text": text, "text_token":cur_ids, "hidden_states": cur_hidden_states}
        
        # 1. prepare input
        lm_input = torch.concat([sos_eos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = prompt_text_token_emb
        next_fill_index = -1
        text_answer = ''
        for res in result:
            text_answer += res["text"]
            # yield None, text_answer
            text_token_emb = self.llm.model.model.embed_tokens(res["text_token"])
            hidden_states = res["hidden_states"].to(torch.float32)
            text_token_emb += self.llm_projctor(hidden_states)
            
            text_cache = torch.concat([text_cache, text_token_emb], dim=1)
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                    logging.info('append {} text token {} speech token, left {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1), prompt_speech_token_emb.size(1)))
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                    
                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                              masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                              cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.speech_token_size + 2
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True).item()
                    if top_ids == self.speech_token_size + 2:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.speech_token_size + 2:
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    yield top_ids, text_answer
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        # lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1) #没有prompt_speech_token_emb，如果参考音频相对于参考文本长度比例较大，可能导致输出音频有一部分为参考音频
        lm_input = torch.concat([lm_input, text_cache, prompt_speech_token_emb, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False).item()
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.speech_token_size:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids, text_answer
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        # print(text_answer)
        # res = self.chat.model.streaming_prefill(
        #     session_id=uuid,
        #     msgs=[{'role': 'assistant', 'content': text_answer}], 
        #     tokenizer=self.chat.tokenizer
        # )

    @torch.inference_mode()
    def inference_end2end_vllm(
            self,
            text: str,
            prompt_text: str,
            prompt_text_token_emb: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid='',
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_speech_token.device
        
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        user_question = [{'role': 'user', 'content': [text]}]
        res = self.chat.model.streaming_prefill(
            session_id=0,
            msgs=user_question, 
            tokenizer=self.chat.tokenizer
        )

        result = self.chat.model.streaming_generate(
            session_id=0,
            tokenizer=self.chat.tokenizer,
            temperature=0.5,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
        ) #res = {"text": text, "text_token":cur_ids, "hidden_states": cur_hidden_states}

        lm_input = torch.concat([sos_eos_emb], dim=1)

        out_tokens = []
        cache = None

        text_cache = prompt_text_token_emb
        next_fill_index = -1
        text_answer = ''
        for res in result:
            text_answer += res["text"]
            text_token_emb = self.llm.model.model.embed_tokens(res["text_token"])
            hidden_states = res["hidden_states"].to(torch.float32)
            text_token_emb += self.llm_projctor(hidden_states)
            
            text_cache = torch.concat([text_cache, text_token_emb], dim=1)
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                    logging.info('append {} text token {} speech token, left {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1), prompt_speech_token_emb.size(1)))
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                
                for top_ids in self.inference_wrapper(lm_input, sampling, 1, self.mix_ratio[1], uuid):
                    yield top_ids, text_answer
                    lm_input = torch.concat([lm_input, self.speech_embedding.weight[top_ids].reshape(1, 1, -1)], dim=1)

        
        # lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1) #没有prompt_speech_token_emb，如果参考音频相对于参考文本长度比例较大，可能导致输出音频有一部分为参考音频
        lm_input = torch.concat([lm_input, text_cache, prompt_speech_token_emb, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
       
        for top_ids in self.inference_wrapper(lm_input, sampling, None, None, uuid):
            yield top_ids, text_answer
            lm_input = torch.concat([lm_input, self.speech_embedding.weight[top_ids].reshape(1, 1, -1)], dim=1)

    @torch.inference_mode()
    def inference_bistream(
            self,
            text: Generator,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_text.device
        # 1. prepare input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = -1
        for this_text in text:
            text_cache = torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    logging.info('append {} text token {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1)))
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                              masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                              cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.speech_token_size + 2
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True).item()
                    if top_ids == self.speech_token_size + 2:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.speech_token_size + 2:
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False).item()
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.speech_token_size:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference_bistream2(
            self,
            text: Generator,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_text.device
        # 1. prepare input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text.dtype).to(device)

        # 2. iterate text
        out_tokens = []
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        prompt_text_emb = self.llm.model.model.embed_tokens(prompt_text)
        lm_input = torch.concat([sos_eos_emb, prompt_text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        seq_len = lm_input.shape[1]
        _, cache = self.llm.forward_one_step(lm_input,
                                                  masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                  cache=None)

        lm_input = torch.concat([sos_eos_emb], dim=1)
        text_cache = None
        next_fill_index = 16
        for this_text in text:
            text_cache = self.llm.model.model.embed_tokens(this_text) if text_cache is None else torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)
            # no prompt_speech_token_emb remain, can decode some speech token
            if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                logging.info('get fill token, need to append more text token')
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text = text_cache[:, :self.mix_ratio[0]]
                    logging.info('append {} text token'.format(lm_input_text.size(1)))
                    if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                        lm_input = lm_input_text
                    else:
                        lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                    text_cache = text_cache[:, self.mix_ratio[0]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    continue
            while True:
                seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                            masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                            cache=cache)
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                    top_ids = self.speech_token_size + 2
                    next_fill_index += (self.mix_ratio[1] + 1)
                else:
                    top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True).item()
                if top_ids == self.speech_token_size + 2:
                    next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                    logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                out_tokens.append(top_ids)
                if top_ids >= self.speech_token_size:
                    if top_ids == self.speech_token_size + 2:
                        break
                    else:
                        raise ValueError('should not get token {}'.format(top_ids))
                yield top_ids
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False).item()
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.speech_token_size:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


