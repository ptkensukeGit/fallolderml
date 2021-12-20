import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


#pycaretで簡単に機械学習
# from pycaret.classification import *
# from pycaret.datasets import get_data

st.title('地域高齢者転倒予測アプリ')
#Excel形式のファイルをDataFrameとして読み込む
#DataFrameは表としてデータを読み込んでいるものになります。
df = pd.read_csv('Inami.csv')

#欠損値のある行を除外
dfex = df.dropna()

#説明変数に必要のない列を削除する
X =  dfex.drop(['Fall'],axis=1)
#従属変数を転倒の有無にする
Y =  dfex.Fall

# LogisticRegressionクラスのインスタンスを作る
log_model = LogisticRegression() # fit_intercept=False, C=1e9) statsmodelsの結果に似せるためのパラメータ。

# データを使って、モデルを作る
log_model.fit(X,Y)

# coef = log_model.coef_
# st.write('標準回帰係数')
# st.write(coef)
# intercept = log_model.intercept_
# st.write('切片')
# st.write(intercept)


# モデルの精度を確認してみる（横断的な分析）
pred = log_model.score(X,Y)
predper = pred * 100
st.write('現在のこのアプリでの転倒歴の予測能力は', round(predper, 2), '%です')

st.write('皆さんのデータでも予測してみましょう')
age = st.sidebar.selectbox(
    '年齢を選択してください',
    list(range(65, 120))
)

if st.sidebar.checkbox('女性ですか?'):
    sex = 1
else:
    sex = 0

if st.sidebar.checkbox('普段の生活で転倒に対する恐怖感がある'):
    FoF = 1
else:
    FoF = 0

mednum = st.sidebar.selectbox(
    '現在服用している薬の数（種類）を記入してください',
    list(range(0, 20))
)

pain = st.sidebar.selectbox(
    '痛みのある関節の数を選んでください',
    list(range(0, 20))
)


tug = st.sidebar.number_input('TUGの秒数を記載して下さい', 3.0, 30.0, 3.0, step=0.1)

dtug = st.sidebar.number_input('dTUGの秒数を記載して下さい', 3.0, 30.0, 3.0, step=0.1)

DTC = (dtug-tug)/((dtug+tug)/2)*100

lrtest = [[age, sex, FoF, mednum, pain, tug, DTC]]
testdf = pd.DataFrame(lrtest)
prob = log_model.predict_proba(testdf)[:, 1] # 目的変数が1である確率を予測
probper = prob*100
fallrisk = round(probper[0], 2)
st.write('あなたの過去一年間の転倒確率は....',fallrisk,'%です！！')