# DNABERT-CRISPR

The code in this repository developed for 

## Usage

- Dataset preparation

    The dataset is located in "/data/dataset/". You should unzip this zip file "tsvdata.zip" before executing the program. 
    Next, download the DNABERT model from [this URL](https://huggingface.co/zhihan1996/DNA_bert_3/tree/main) and save it as "pretrained_dnabert_3" in "/data/saved_weights/".

- Create Envirionment

    The code was tested with:\
    Python interpreter == 3.8.12\
    Python packages requared as env.yml (other versions may work as well)\
    GPU : RTX3090

- Execute programs

    To train the DNABERT-CRISPR model, you should execute (if fold 1 and regression)
    ``` python
    python3 script/finetuning/dnabert_pair_ft.py
    python3 script/dnabert/dnabert_ot_ft.py --fold 1 --task regr
    ```

    To test the DNABERT-CRIPSR model, you should execute (if fold 1 and regression)
    ``` python
    python3 script/dnabert/ot_ft_test.py --fold 1 --task regr
    ```

-------------------------------------------------
Kai Kimata

kkaibioinformatics(at-mark)gmail_domain

August 28 2024


## 使用方法

- データセット準備

    データセットは"/data/dataset/"ディレクトリに配置します。プログラムの実行前に"tsvdata.zip"を解凍して下さい。
    次に、[このURL](https://huggingface.co/zhihan1996/DNA_bert_3/tree/main)からDNABERTモデルをダウンロードし、"/data/saved_weights/"ディレクトリに"pretrained_dnabert_3"として保存してください。

- 環境構築

    プログラムコードは以下の条件で実行しました。\
    Pythonインタープリター == 3.8.12\
    Pythonのパッケージはenv.ymlファイルの通りです。（他のバージョンでも実行可能と思われます）\
    GPU : RTX3090

- プログラム実行方法

    DNABERT-CRISPRモデルを訓練するには、以下を実行してください。（第1交差目で回帰の場合）
    ``` python
    python3 script/finetuning/dnabert_pair_ft.py
    python3 script/dnabert/dnabert_ot_ft.py --fold 1 --task regr
    ```

    DNABERT-CRISPRモデルをテストする場合には、以下を実行してください。（第1交差目で回帰の場合）
    ``` python
    python3 script/dnabert/ot_ft_test.py --fold 1 --task regr
    ```

-------------------------------------------------
木俣海

kkaibioinformatics(at-mark)gmail_domain

2024年8月28日
