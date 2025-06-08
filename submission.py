import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 今回は使っていませんが将来使うなら
import os

# ─── ページタイトル ───
st.title("Optuna チューニング結果の提出")

# ─── アップロード用ディレクトリ ───
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Ground-truth の読み込み ───
GROUND_TRUTH_PATH = "test_data.csv"
try:
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
except FileNotFoundError:
    st.error(f"Ground-truth ファイルが見つかりません：{GROUND_TRUTH_PATH}")
    st.stop()

# ─── 説明文 ───
st.markdown("""
## データセットの評価  
テストデータセット (predict_group(?).csv) に対する予測精度を Accuracy で評価します。

### 注意  
- ファイル名は predict_group(?).csv (? は班名) としてください。  
  - A班の場合: predict_groupA.csv  
  - オンラインの方々は、predict_group_氏名.csv  
- 予測結果は `match` 列に格納してください。
""")

# ─── ファイルアップローダー ───
uploaded_file = st.file_uploader(
    "予測ファイルをアップロードしてください (CSV形式)", type="csv"
)
if uploaded_file is not None:
    dst = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"ファイル `{uploaded_file.name}` がアップロードされました")

# ─── 提出履歴と削除ボタン ───
st.markdown("## 提出履歴（Accuracy と削除）")
files = sorted(os.listdir(UPLOAD_DIR))
if not files:
    st.warning("まだ提出がありません。提出ファイルをお待ちしています。")
    st.stop()

for fn in files:
    path = os.path.join(UPLOAD_DIR, fn)
    # 精度計算
    try:
        df_pred = pd.read_csv(path)
        acc = (df_pred["match"] == ground_truth["match"]).mean()
        acc_str = f"{acc:.4f}"
    except Exception as e:
        acc_str = "読み込み失敗"
        st.warning(f"{fn} の読み込みに失敗: {e}")

    # 1行分の UI
    col1, col2, col3 = st.columns([4, 2, 1])
    col1.write(fn)
    col2.write(acc_str)
    if col3.button("削除", key=f"del_{fn}"):
        os.remove(path)
        st.success(f"`{fn}` を削除しました。")
        st.experimental_rerun()

# ─── リーダーボード表示 ───
# 提出済みファイルの上位3件だけを見やすく DataFrame にまとめても良いです
leaderboard = []
for fn in files:
    path = os.path.join(UPLOAD_DIR, fn)
    try:
        df_pred = pd.read_csv(path)
        acc = (df_pred["match"] == ground_truth["match"]).mean()
        leaderboard.append({"ファイル名": fn, "Accuracy": acc})
    except:
        continue

if leaderboard:
    lb = pd.DataFrame(leaderboard)
    lb = lb.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    lb.index += 1
    lb.insert(0, "順位", lb.index)
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lb["順位"] = lb["順位"].map(lambda i: f"{medals.get(i,'')} {i}" if i in medals else i)
    st.markdown("## リーダーボード（Top）")
    st.dataframe(lb)
