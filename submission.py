import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

#ページ タイトル
st.title("Optuna チューニング結果の提出")

# データ保存ディレクトリ
UPLOAD_DIR = 'uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ground truth データの読み込み
GROUND_TRUTH_PATH = os.path.join("test_data.csv")
ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

st.markdown("""
    ## データセットの評価
    テストデータセット (predict_group(?).csv) に対する予測精度をAccuracyで評価します。
    
    ### 注意
    - ファイル名は predict_group(?).csv (? は班名)としてください。
        - A班の場合: predict_groupA.csv
        - オンラインの方々は、predict_group_氏名.csv としてください。
    - 予測結果は match 列に格納してください。
    """)

# ファイルのアップロード
uploaded_file = st.file_uploader("予測ファイルをアップロードしてください (CSV形式)", type=["csv"])
if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"ファイル {uploaded_file.name} が正常にアップロードされました")

    # アップロードされたファイルを読み込み、評価する
    predictions = pd.read_csv(file_path)
    accuracy = (predictions["match"] == ground_truth["match"]).mean()
    st.write(f"Accuracy: {accuracy}")

# 既存のファイルリスト
uploaded_files = os.listdir(UPLOAD_DIR)

# リーダーボードの作成
leaderboard = []
for file_name in uploaded_files:
    file_path = os.path.join(UPLOAD_DIR, file_name)
    predictions = pd.read_csv(file_path)
    accuracy = (predictions["match"] == ground_truth["match"]).mean()
    leaderboard.append({"ファイル名": file_name, "Accuracy": accuracy})

# DataFrameに変換してソート
leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="Accuracy", ascending=False)
leaderboard_df.reset_index(drop=True, inplace=True)
leaderboard_df.index += 1  # インデックスを1から始める

# 順位列を追加
leaderboard_df['順位'] = leaderboard_df.index

# 列の順序を変更
leaderboard_df = leaderboard_df[['順位', 'ファイル名', 'Accuracy']]

# 1位の行に金メダルを追加
leaderboard_df.loc[leaderboard_df['順位'] == 1, '順位'] = '🥇 1'

# 2位の行に銀メダルを追加
leaderboard_df.loc[leaderboard_df['順位'] == 2, '順位'] = '🥈 2'

# 3位の行に銅メダルを追加
leaderboard_df.loc[leaderboard_df['順位'] == 3, '順位'] = '🥉 3'
# 3位以降の行には何も追加しない

st.markdown("""
    ## リーダーボード""")
st.write(leaderboard_df)