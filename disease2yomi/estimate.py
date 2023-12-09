from __future__ import unicode_literals
import re
import unicodedata
import argparse
import random
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def remove_brackets(text):
    text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
    return text

def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def get_normalized_text(text):
    text = text.strip()
    text = normalize_text(remove_brackets(text))
    text = normalize_text(text)
    return text

def set_seed(seed):
    """
    乱数シードの設定
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def estimate_yomi(disease_name):
    # disease2yomi_v1.ipynb
    # 事前学習済みモデル
    PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

    # 転移学習済みモデル
    MODEL_DIR = f"disease2yomi/trained_model/to_yomi"

    # https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変

    set_seed(42)

    # GPU利用有無
    USE_GPU = torch.cuda.is_available()

    # 各種ハイパーパラメータ
    args_dict = dict(
        # data_dir="../data",  # データセットのディレクトリ
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,

        # max_input_length=512,
        # max_target_length=64,
        # train_batch_size=8,
        # eval_batch_size=8,
        # num_train_epochs=4,

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )

    # 学習に用いるハイパーパラメータを設定する
    args_dict.update({
        # "max_input_length":  512,  # 入力文の最大トークン数
        "max_input_length":  128,  # 入力文の最大トークン数
        "max_target_length": 128,  # 出力文の最大トークン数
        # "train_batch_size":  8,  # 訓練時のバッチサイズ
        "train_batch_size":  8,  # 訓練時のバッチサイズ
        "eval_batch_size":   8,  # テスト時のバッチサイズ
        "num_train_epochs":  8,  # 訓練するエポック数
        })
    args = argparse.Namespace(**args_dict)

    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True, local_files_only=True)

    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    body = disease_name

    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

    # 推論モード設定
    trained_model.eval()

    # 前処理とトークナイズを行う
    inputs = [get_normalized_text(body)]
    batch = tokenizer.batch_encode_plus(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
        padding="longest", return_tensors="pt")

    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    # 生成処理を行う
    outputs = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length,
        temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
        repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

    generated_titles = [tokenizer.decode(ids, skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=False) 
                        for ids in outputs]

    yomi = generated_titles[0]
    return yomi

def estimate_icd10(disease_name):
    # disease2icd10_v1.ipynb
    # 事前学習済みモデル
    PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

    # 転移学習済みモデル
    MODEL_DIR = f"disease2yomi/trained_model/to_icd10"

    set_seed(42)

    # GPU利用有無
    USE_GPU = torch.cuda.is_available()

    # 各種ハイパーパラメータ
    args_dict = dict(
    #     data_dir=DIRPATH_DATA,  # データセットのディレクトリ
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,

        # max_input_length=512,
        # max_target_length=64,
        # train_batch_size=8,
        # eval_batch_size=8,
        # num_train_epochs=4,

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )

    # 学習に用いるハイパーパラメータを設定する
    args_dict.update({
        # "max_input_length":  256,  # 入力文の最大トークン数
        # "max_input_length":  128,  # 入力文の最大トークン数
        "max_input_length":  128,  # 入力文の最大トークン数
        "max_target_length": 8,  # 出力文の最大トークン数
        "train_batch_size":  8,  # 訓練時のバッチサイズ
        "eval_batch_size":   8,  # テスト時のバッチサイズ
        "num_train_epochs":  8,  # 訓練するエポック数
        })
    args = argparse.Namespace(**args_dict)

    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True, local_files_only=True)

    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    body = disease_name

    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

    # 推論モード設定
    trained_model.eval()

    # 前処理とトークナイズを行う
    inputs = [get_normalized_text(body)]
    batch = tokenizer.batch_encode_plus(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
        padding="longest", return_tensors="pt")

    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    # 生成処理を行う
    outputs = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length,
        temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
    )

    # 生成されたトークン列を文字列に変換する
    generated_titles = [tokenizer.decode(ids, skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=False) 
                        for ids in outputs]

    icd10 = generated_titles[0].upper()
    return icd10

def estimate_yomi_from_file(df, column):
    # disease2yomi_v1.ipynb
    # 事前学習済みモデル
    PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

    # 転移学習済みモデル
    MODEL_DIR = f"disease2yomi/trained_model/to_yomi"

    # https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変

    set_seed(42)

    # GPU利用有無
    USE_GPU = torch.cuda.is_available()

    # 各種ハイパーパラメータ
    args_dict = dict(
        # data_dir="../data",  # データセットのディレクトリ
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,

        # max_input_length=512,
        # max_target_length=64,
        # train_batch_size=8,
        # eval_batch_size=8,
        # num_train_epochs=4,

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )

    # 学習に用いるハイパーパラメータを設定する
    args_dict.update({
        # "max_input_length":  512,  # 入力文の最大トークン数
        "max_input_length":  128,  # 入力文の最大トークン数
        "max_target_length": 128,  # 出力文の最大トークン数
        # "train_batch_size":  8,  # 訓練時のバッチサイズ
        "train_batch_size":  8,  # 訓練時のバッチサイズ
        "eval_batch_size":   8,  # テスト時のバッチサイズ
        "num_train_epochs":  8,  # 訓練するエポック数
        })
    args = argparse.Namespace(**args_dict)

    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True, local_files_only=True)

    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

    # 推論モード設定
    trained_model.eval()

    list_yomi = [""] * len(df[column])
    for i in range(len(df[column])):
        body = df[column][i]

        # 前処理とトークナイズを行う
        inputs = [get_normalized_text(body)]
        batch = tokenizer.batch_encode_plus(
            inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
            padding="longest", return_tensors="pt")

        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        # 生成処理を行う
        outputs = trained_model.generate(input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=args.max_target_length,
            temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
            repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
            )

        generated_titles = [tokenizer.decode(ids, skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False) 
                            for ids in outputs]

        list_yomi[i] = generated_titles[0]
    return list_yomi

def estimate_icd10_from_file(df, column):
    # disease2icd10_v1.ipynb
    # 事前学習済みモデル
    PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

    # 転移学習済みモデル
    MODEL_DIR = f"disease2yomi/trained_model/to_icd10"

    set_seed(42)

    # GPU利用有無
    USE_GPU = torch.cuda.is_available()

    # 各種ハイパーパラメータ
    args_dict = dict(
    #     data_dir=DIRPATH_DATA,  # データセットのディレクトリ
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,

        # max_input_length=512,
        # max_target_length=64,
        # train_batch_size=8,
        # eval_batch_size=8,
        # num_train_epochs=4,

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )

    # 学習に用いるハイパーパラメータを設定する
    args_dict.update({
        # "max_input_length":  256,  # 入力文の最大トークン数
        # "max_input_length":  128,  # 入力文の最大トークン数
        "max_input_length":  128,  # 入力文の最大トークン数
        "max_target_length": 8,  # 出力文の最大トークン数
        "train_batch_size":  8,  # 訓練時のバッチサイズ
        "eval_batch_size":   8,  # テスト時のバッチサイズ
        "num_train_epochs":  8,  # 訓練するエポック数
        })
    args = argparse.Namespace(**args_dict)

    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True, local_files_only=True)

    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

    # 推論モード設定
    trained_model.eval()

    list_icd10 = [""] * len(df[column])
    for i in range(len(df[column])):
        body = df[column][i]

        # 前処理とトークナイズを行う
        inputs = [get_normalized_text(body)]
        batch = tokenizer.batch_encode_plus(
            inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
            padding="longest", return_tensors="pt")

        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        # 生成処理を行う
        outputs = trained_model.generate(input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=args.max_target_length,
            temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
        )

        # 生成されたトークン列を文字列に変換する
        generated_titles = [tokenizer.decode(ids, skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False) 
                            for ids in outputs]

        list_icd10[i] = generated_titles[0].upper()
    return list_icd10


# print(estimate_yomi("帯状疱疹"))
