<h1 align='center'>JoyTTS: LLM-based Spoken Chatbot With Voice Cloning</h1>

<div align='center'>
    <a href='https://github.com/zhoufangru' target='_blank'>Zhou Fangru</a> 
    <a href='https://github.com/zhaojun060708' target='_blank'>Jun Zhao</a> 
    Guoxin Wang
</div>
<div align='center'>
    JD Health International Inc.
</div>

<br>
<div align='center'>
    <a href='https://huggingface.co/jdh-algo/JoyTTS-v1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
</div>
<br>

## 📖 Introduction

JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SM (Similarity Measure) score of 73 and a WER (Word Error Rate) of 5.


### 🧳 Framework

![Network](assets/流式结构图.jpg "Network")

### 🎬 Demo


<table>
<tr>
<td width="25%">
问题
</td>
<td width="25%">
孙悟空
</td>
<td width="25%">
猪八戒
</td>
<td width="25%">
林黛玉
</td>
</tr>
<tr>
<td width="25%">
参考音频
</td>
<td width="25%">

https://github.com/user-attachments/assets/2fa5d07d-a6f0-4daf-ba39-057996fffa05
</td>
<td width="25%">

https://github.com/user-attachments/assets/23235b6c-7f38-47a7-a48a-765e98c938db
</td>
<td width="25%">

https://github.com/user-attachments/assets/127592dd-e4dc-4fdb-9aa5-4f8063d25252
</td>
</tr>
<tr>
<td width="25%">
“今天天气怎么样”
</td>
<td width="25%">

https://github.com/user-attachments/assets/349b67c5-ef93-462e-8e7d-3e8f76c54ef7
</td>
<td width="25%">

https://github.com/user-attachments/assets/afb16ba1-9c90-45ba-9dc6-b4b8173019d4
</td>
<td width="25%">

https://github.com/user-attachments/assets/da696932-8f87-4832-be50-886962fe8608
</td>
</tr>
<tr>
<td width="25%">
“中国的全称是什么”
</td>
<td width="25%">

https://github.com/user-attachments/assets/33f0c58d-59c9-4a57-8016-c0cb88165fbd
</td>
<td width="25%">

https://github.com/user-attachments/assets/5fbb03f3-fa2f-4f3d-a6f3-a341c05980c9
</td>
<td width="25%">

https://github.com/user-attachments/assets/c82e850b-e755-4146-8fb1-07b35b1dcb36
</td>
</tr>
<tr>
<td width="25%">
“手机都有哪些作用？”
</td>
<td width="25%">

https://github.com/user-attachments/assets/cedeb3a0-a307-48df-b1b9-61539357c201
</td>
<td width="25%">

https://github.com/user-attachments/assets/467a337c-373f-45aa-8e5e-b97cb543221f
</td>
<td width="25%">

https://github.com/user-attachments/assets/9981dfff-3a6b-43d9-a5b5-1a92bfb54d55
</td>
</tr>
</table>


### 💻 Results on SEED test-zh

|             Model             | sm $\uparrow$ | wer $\downarrow$ |
| :---------------------------: | :------: | :----: |
|     gpt-sovits        |  0.55  | 5.13 |
|   cosyvoice2  |   **0.74**   |  **2.47**  |
|     JoyTTS        |  0.73  | 5.09 |


## ⚙️ Installation

### 1. Create Conda env

``` sh
conda create -n JoyTTS -y python=3.10
conda activate JoyTTS
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements_JoyTTS.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 2. Model download

``` sh
huggingface-cli download --resume-download --repo-type model jdh-algo/JoyTTS-v1 --local-dir pretrained_models
```

## 🚀 Inference

### 1. Use spk_id

1. Change the prompt info in pretrained_models/prompt_info.py
2. Delete pretrained_models/spk2info.pt
3. Run the following command

```shell
python inference.py --question '今天天气怎么样？' --spk_id '孙悟空' 
```

### 2. Use prompt info
```shell
python inference.py --question '今天天气怎么样？' --prompt_text 'prompt text' --prompt_wav the_path_of_prompt_wav 
```
The result will be saved in "output.wav"


### 3. Gradio demo
```shell
python gradio_demo.py
```


## 🚧 Fine-tuning
### 1. Data preparation

#### 1. Prepare conversation data for LLM-Chat
<details>
  <summary>
    <b>example (normalized_data.json) with 1 samples.</b>
  </summary>

```
  [
    {
      "utt": "0",
      "reference": '数字银行 《数字银行》是2015年12月新世界出版社出版的金融类图书，该书的作者是刘振友。 书 名 数字银行 作 者 刘振友 类 别 金融 出版社 新世界出版社 出版时间 2015年12月 页 数 290 页 定 价 48.8 开 本 16 开 装 帧 平装 ISBN 978-7-5104-5480-6 数字银行内容简介 《数字银行》认为，传统银行若要更好地存续，必须实现自我进化。本书还将带你穿越历史，解读现代金融行业的发展轨道，知古鉴今，把握未来银行的发展思维脉络。 数字银行作者简介 刘振友是全球第一批致力于加密数字资产、数字货币、虚拟货币、商业积分、共赢经济模式、全民免费经济领域研究、应用、推广的资深专家，新锐作家。主要著作有:《颠覆》《3分钟让别人记住你》《互联网+》《数字银行》《让世界免费》。 数字银行作品目录',
      "conversations": [
            {
              'role': 'user', 
              'content': '请问这本《数字银行》是什么时候出版的？'
            }, 
            {
                'role': 'assistant', 
                'content': '这本《数字银行》是2015年12月出版的。它是一本金融类图书，由新世界出版社出版。了解这本书的出版时间可以帮助你了解它的时效性，从而了解该书所描述的内容和观点是否仍然适用于当前的金融环境。'
            },   
            {
                'role': 'user', 
                'content': '作者刘振友都写过哪些其他书籍？'
            },
            {
                'role': 'assistant', 
                'content': '刘振友是一位新锐作家，他的主要著作有《颠覆》、《3分钟让别人记住你》、《互联网+》、《数字银行》和《让世界免费》。他是全球第一批致力于加密数字资产、数字货币、虚拟货币、商业积分、共赢经济模式、全民免费经济领域研究、应用、推广的资深专家。了解他的其他著作可以帮助你更全面地了解他的思想和研究领域。'
            }, 
        ]
    },
  ]
```
</details>

#### 2. Prepare data for  LLM-TTS
To enhance the efficiency of training the LLM-TTS model, we pre-generated hidden states using the already trained LLM-Chat model.

##### launch server
```shell
cd prepare_data_sever and python sever_llmanswer.py
cd prepare_data_sever and python sever_tts.py
```
##### generate parquet file
```shell
cd prepare_data_generate
python 0_generate_datalist.py --data_root your_data_root --data_name your_data_name
python 1_generate_llmanswer.py --data_root your_data_root --data_name your_data_name
python 2_generate_tts.py --data_root your_data_root --data_name your_data_name
python 3_split_train_eval_offline.py --data_root your_data_root --data_name your_data_name
python 4_make_parquet_list.py --data_root your_data_root --data_name your_data_name
```
The generated data will be saved in {your_data_root}/parquet/{your_data_name}

#### 3. Prepare data for end-to-end training
##### launch server
```shell
cd prepare_data_sever and python sever_tts.py
```
##### generate parquet file
```shell
cd prepare_data_generate_end2end
python 0_generate_datalist.py --data_root your_data_root --data_name your_data_name
python 2_generate_tts.py --data_root your_data_root --data_name your_data_name
python 3_split_train_eval_offline.py --data_root your_data_root --data_name your_data_name
python 4_make_parquet_list.py --data_root your_data_root --data_name your_data_name
```
The generated data will be saved in {your_data_root}/parquet_uselabel/{your_data_name}


### 2. Start train
#### 1. LLM-TTS model only

``` shell
cd examples/end2end
ln -s your_data_root data
sh run.sh #set stage=1
```
#### 2. end-to-end training

``` shell
cd examples/end2end
ln -s your_data_root data
sh run.sh #set stage=2
```

## 🎒 TODO
- [x] training and inference codes.
- [x] trained models and technical report.
- [x] freestyle dialogue model with voice cloning.
- [ ] vllm support.
- [ ] a better and faster model.
- [ ] automatic emotion control.


## 📝 Citations

If you find our work helpful, please consider citing us:

```
@misc{zhou2025joytts,
  title={JoyTTS: LLM-based Spoken Chatbot With Voice Cloning}, 
  author={Fangru Zhou and Jun Zhao and Guoxin Wang},
  year={2025},
  howpublished = {\url{https://jdh-algo.github.io/JoyTTS}},
}
```

## 🤝 Acknowledgments

We would like to thank the contributors to the [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o?tab=readme-ov-file), [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice2https://github.com/FunAudioLLM/CosyVoice2)repositories, for their open research and extraordinary work.
