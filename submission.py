import streamlit as st
import pandas as pd
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

# ─── 削除用コールバック定義 ───
def delete_and_rerun(fn: str):
    """この関数は削除→再実行だけをやる"""
    os.remove(os.path.join(UPLOAD_DIR, fn))
    # 削除したら即ページ全体を再実行
    st.experimental_rerun()

# ─── 提出履歴と削除ボタン ───
st.markdown("## 提出履歴（Accuracy と削除）")

# ← ここで一度だけ一覧を取得
files = sorted(os.listdir(UPLOAD_DIR))
if not files:
    st.warning("まだ提出がありません。提出ファイルをお待ちしています。")
    st.stop()

for fn in files:
    path = os.path.join(UPLOAD_DIR, fn)

    # 精度の計算
    try:
        df_pred = pd.read_csv(path)
        acc_str = f"{(df_pred['match'] == ground_truth['match']).mean():.4f}"
    except Exception as e:
        acc_str = "読み込み失敗"
        st.warning(f"{fn} の読み込みに失敗: {e}")

    # １行ごとに「ファイル名」「Accuracy」「削除ボタン」
    c1, c2, c3 = st.columns([4, 2, 1])
    c1.write(fn)
    c2.write(acc_str)
    # on_click の中では delete_and_rerun が呼ばれるだけ
    c3.button(
        "削除",
        key=f"del_{fn}",
        on_click=delete_and_rerun,
        args=(fn,)
    )

# ─── リーダーボード表示（任意） ───
st.markdown("## リーダーボード（Top）")
leaderboard = []
for fn in files:
    path = os.path.join(UPLOAD_DIR, fn)
    try:
        df_pred = pd.read_csv(path)
        leaderboard.append({
            "ファイル名": fn,
            "Accuracy": (df_pred["match"] == ground_truth["match"]).mean()
        })
    except:
        pass

if leaderboard:
    lb = (
        pd.DataFrame(leaderboard)
          .sort_values("Accuracy", ascending=False)
          .reset_index(drop=True)
    )
    lb.index += 1
    lb.insert(0, "順位", lb.index)
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lb["順位"] = lb["順位"].map(lambda i: f"{medals.get(i,'')} {i}" if i in medals else i)
    st.dataframe(lb)
