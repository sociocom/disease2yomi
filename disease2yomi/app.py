import pandas as pd
from datetime import datetime
import jaconv
from estimate import estimate_yomi, estimate_icd10, estimate_yomi_of_body, estimate_yomi_from_file, estimate_icd10_from_file, estimate_yomi_of_body_from_file
import streamlit as st
import streamlit_ext as ste
from stqdm import stqdm
from io import StringIO


def set_streamlit():
    # カスタムテーマの定義
    st.set_page_config(
        page_title="Yomi and ICD-10 code estimator from disease name",
        # page_icon=":chipmunk:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            # "Get Help": "https://www.extremelycoolapp.com/help",
            # "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": """
            ## Yomi and ICD-10 code estimator from disease name
            病名や症状からそのふりがなとICD-10コードを推定するWebアプリです。

            奈良先端科学技術大学院大学 ソーシャル・コンピューティング研究室""",
        },
    )
    st.title("Yomi and ICD-10 code estimator from disease name")
    # st.markdown("###### 脳卒中のリスク因子を構造化し、csv形式で出力するシステムです。")
    st.markdown("病名や症状からそのふりがなとICD-10コードを推定するWebアプリです。")

    st.sidebar.write("### サンプルファイルで実行する場合は以下のファイルをダウンロードしてください")
    sample_disease_name_csv = pd.read_csv("disease2yomi/data/sample_disease_name_100.csv")
    sample_disease_name_csv = sample_disease_name_csv.to_csv(index=False)
    ste.sidebar.download_button("病名のsample data", sample_disease_name_csv, f"sample_disease_name_100.csv")
    sample_body_name_csv = pd.read_csv("disease2yomi/data/sample_body_name_100.csv")
    sample_body_name_csv = sample_body_name_csv.to_csv(index=False)
    ste.sidebar.download_button("部位名のsample data", sample_body_name_csv, f"sample_body_name_100.csv")

    st.sidebar.markdown("### ふりがなとICD-10コードを推定する病名データの入ったcsvファイルを選択してください")
    # ファイルアップロード
    file_uploaded_disease_name = st.sidebar.file_uploader(
        "Choose a CSV file", accept_multiple_files=False, key="disease_name"
    )
    st.sidebar.markdown("### ふりがなを推定する部位名データの入ったcsvファイルを選択してください")
    # ファイルアップロード
    file_uploaded_body_name = st.sidebar.file_uploader(
        "Choose a CSV file", accept_multiple_files=False, key="body_name"
    )

    st.sidebar.markdown("奈良先端科学技術大学院大学 ソーシャル・コンピューティング研究室")

    return file_uploaded_disease_name, file_uploaded_body_name

def convert_to_utf8(content, encoding):
    try:
        return content.decode(encoding).encode("utf-8")
    except UnicodeDecodeError:
        return None

def read_uploaded_file_as_utf8(uploaded_file):
    # ファイルをバイナリモードで読み込み
    content = uploaded_file.read()

    # エンコーディングを自動検出し、UTF-8に変換
    encodings_to_try = [
        "utf-8",
        "shift-jis",
        "cp932",
        "latin-1",
        "ISO-8859-1",
        "euc-jp",
        "euc-kr",
        "big5",
        "utf-16",
    ]
    utf8_content = None

    for encoding in encodings_to_try:
        utf8_content = convert_to_utf8(content, encoding)
        if utf8_content is not None:
            break

    try:
        df = pd.read_csv(StringIO(utf8_content.decode("utf-8")))
    except pd.errors.EmptyDataError:
        st.error("データが読み込めませんでした。utf-8のエンコードのcsvファイルを選んでください。")

    return df

# def replace_spaces(text):
#     # 2つ以上連続したスペースを1つのスペースに置換
#     text = re.sub(r" {2,}", " ", text)
#     # タブをスペースに置換
#     text = re.sub(r"\t", " ", text)
#     return text

# class GenerateText:
#     """GenerateText"""

#     def __init__(
#         self,
#         data_batch_size=16,
#         token_max_length_src=512,
#         token_max_length_tgt=8,
#     ):
#         self.data_batch_size = data_batch_size
#         self.token_max_length_src = token_max_length_src
#         self.token_max_length_tgt = token_max_length_tgt
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#     def download_model(self):
#         """Download model"""
#         tokenizer = T5Tokenizer.from_pretrained(f"models/tokenizer")
#         model = T5ForConditionalGeneration.from_pretrained(f"models/model").to(
#             self.device
#         )
#         return tokenizer, model

#     def generate_text(self, model, tokenizer, df0, text, target_columns):
#         """Generate text"""
#         df = df0.copy()
#         original_texts = df[text].to_list()
#         prediction = []
#         df[text] = df[text].apply(replace_spaces)
#         df[text] = df[text].apply(mojimoji.han_to_zen)
#         df[text] = str(target_columns) + "：" + df[text]

#         for i in tqdm(range(0, len(df), self.data_batch_size)):
#             batch = df.iloc[i : i + self.data_batch_size, :]
#             soap = batch[text].to_list()
#             generated_text = generate_text_from_model(
#                 tags=soap,
#                 trained_model=model,
#                 tokenizer=tokenizer,
#                 num_return_sequences=1,
#                 max_length_src=self.token_max_length_src,
#                 max_length_target=self.token_max_length_tgt,
#                 num_beams=10,
#                 device=self.device,
#             )
#             # original_texts.extend(soap)
#             prediction.extend(generated_text)

#         column_df = pd.DataFrame(
#             {
#                 text: original_texts,
#                 target_columns: prediction,
#             }
#         )
#         return column_df

# def main():
file_uploaded_disease_name, file_uploaded_body_name = set_streamlit()

disease_name = st.text_input('病名や症状を入力してください')
if disease_name:
    st.write(f"入力された病名・症状：{disease_name}")
    with st.spinner("ふりがなを推定中..."):
        yomi = estimate_yomi(disease_name)
        yomi = jaconv.kata2hira(yomi)
    st.write(f"推定されたふりがな　：**{yomi}**")
    with st.spinner("ICD-10コードを推定中..."):
        icd10 = estimate_icd10(disease_name)
    st.write(f"推定されたICD-10コード：**{icd10}**")

# generator = GenerateText(
#     data_batch_size=4,
#     token_max_length_src=512,
#     token_max_length_tgt=8,
# )
# target_columns = [
#     "ふりがな",
#     "ICD-10コード",
# ]

if file_uploaded_disease_name:
    df = read_uploaded_file_as_utf8(file_uploaded_disease_name)
    st.write("入力ファイル (先頭5件までを表示)")
    st.dataframe(df.head(5))
    text = st.selectbox(
        "ふりがなとICD-10コードを推定する列を選んでください",
        (df.columns),
        index=None,
        placeholder="Select...",
    )
    st.write("選択した列:", text)

    if text:
        with st.spinner("ふりがなを推定中..."):
            output_df = pd.DataFrame()
            output_df[text] = df[text]
            output_df['ふりがな'] = estimate_yomi_from_file(df=output_df, column=text)
            for i in stqdm(range(len(output_df))):
                output_df.iat[i, 1] = jaconv.kata2hira(output_df.iat[i, 1])
        st.write("✅ ふりがなの推定が完了しました")
        with st.spinner("ICD-10コードを推定中..."):
            output_df['ICD-10コード'] = estimate_icd10_from_file(df=output_df, column=text)
        st.write("✅ ICD-10コードの推定が完了しました")

            # # tokenizer, model = generator.download_model()
            # for i, column in enumerate(target_columns):
            #     column_df = generator.generate_text(model, tokenizer, df, text, column)
            #     # display_df = column_df.copy()
            #     # display_df[text] = display_df[text].iloc[:5].str[:5] + "..."

            #     if i == 0:
            #         output_df = column_df
            #         # mytable = st.table(display_df.iloc[:5].T)
            #     else:
            #         output_df = pd.merge(output_df, column_df, on=text, how="inner")
            #         # mytable.add_rows(display_df[[column]].iloc[:5].T)

        st.write("推定結果 (先頭5件までを表示)")
        st.dataframe(output_df.head(5))
        if "completed" not in st.session_state:
            st.session_state["completed"] = True

        file_name_output = file_uploaded_disease_name.name.replace(".csv", "")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        csv = output_df.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode("utf-8-sig")).decode()
        # href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}-riskun-{timestamp}.csv">Download Link</a>'
        # st.markdown(f"CSVファイルのダウンロード: {href}", unsafe_allow_html=True)

        ste.download_button(
            "Click to download data", csv, f"{file_name_output}_yomi_and_icd-10_code_{timestamp}.csv"
        )

body_name = st.text_input('部位名を入力してください')
if body_name:
    st.write(f"入力された部位名　：{body_name}")
    with st.spinner("ふりがなを推定中..."):
        yomi_of_body = estimate_yomi_of_body(body_name)
        yomi_of_body = jaconv.kata2hira(yomi_of_body)
    st.write(f"推定されたふりがな：**{yomi_of_body}**")

if file_uploaded_body_name:
    df_body = read_uploaded_file_as_utf8(file_uploaded_body_name)
    st.write("入力ファイル (先頭5件までを表示)")
    st.dataframe(df_body.head(5))
    text_body = st.selectbox(
        "ふりがなを推定する列を選んでください",
        (df_body.columns),
        index=None,
        placeholder="Select...",
    )
    st.write("選択した列:", text_body)

    if text_body:
        with st.spinner("ふりがなを推定中..."):
            df_output_body = pd.DataFrame()
            df_output_body[text_body] = df_body[text_body]
            df_output_body['ふりがな'] = estimate_yomi_of_body_from_file(df=df_output_body, column=text_body)
            for i in stqdm(range(len(df_output_body))):
                df_output_body.iat[i, 1] = jaconv.kata2hira(df_output_body.iat[i, 1])
        st.write("✅ ふりがなの推定が完了しました")

            # # tokenizer, model = generator.download_model()
            # for i, column in enumerate(target_columns):
            #     column_df = generator.generate_text(model, tokenizer, df, text, column)
            #     # display_df = column_df.copy()
            #     # display_df[text] = display_df[text].iloc[:5].str[:5] + "..."

            #     if i == 0:
            #         output_df = column_df
            #         # mytable = st.table(display_df.iloc[:5].T)
            #     else:
            #         output_df = pd.merge(output_df, column_df, on=text, how="inner")
            #         # mytable.add_rows(display_df[[column]].iloc[:5].T)

        st.write("推定結果 (先頭5件までを表示)")
        st.dataframe(df_output_body.head(5))
        if "completed" not in st.session_state:
            st.session_state["completed"] = True

        file_name_output_body = file_uploaded_body_name.name.replace(".csv", "")
        timestamp_body = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        csv_output_body = df_output_body.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode("utf-8-sig")).decode()
        # href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}-riskun-{timestamp}.csv">Download Link</a>'
        # st.markdown(f"CSVファイルのダウンロード: {href}", unsafe_allow_html=True)

        ste.download_button(
            "Click to download data", csv_output_body, f"{file_name_output_body}_yomi_{timestamp_body}.csv"
        )


# if __name__ == "__main__":
# main()
