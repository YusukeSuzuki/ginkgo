用語
================================================================================

名前はそのうち変える

- prophet : 盤面判定
  - 相手の手番の盤面について自分の勝ちか相手の勝ちかを学習する
- enforcer : 手決定
  - 自分の手番の盤面について prophet から合格を得る手を学習する

入出力
================================================================================

## 入力形式

コードを参照すること。同じ shape について prophet は相手手番とみなし enforcer は自分手番とみなす。
自身が後手であっても自分が下から攻める形に整形して入力する。

- shape : [148,9,9]
- レイヤー
  - [配置レイヤー, 合法動きレイヤー] * 駒数
  - 駒数は先手後手合算
  - 駒と駒順 : k*2, r*2, +r*2, b*2, +b*2, g*4, s*4, +s*4, n*4, +n*4, l*4, +l*4, p*18, +p*18
  - 自分駒とその動きは対応位置に1が当てられる。相手駒とその動きは対応位置に-1が当てられる。
  - 駒がない位置、動けない位置には0が当てられる。

## 出力形式

- prophet
  - 入力盤面に対する自身の勝敗予測
  - out = [2]
  - out[0] : 自身の勝敗, 1:win, 0:lose
  - out[1] : 相手の勝敗, 1:win, 0:lose
- enforcer
  - 入力盤面に対する次の自身の手
  - 入力形式から配置レイヤーを除いた形式
    - [74,9,9]
    - 相手玉は手駒にできないが計算の簡便のためレイヤーに含める
  - 教師データとしては手候補の位置に正の実数、そうでない位置に0以下の実数を当てる
  - 手の決定はargmaxで行う
    - to位置 : 出力の位置
    - from位置 : 出力レイヤーに対応する入力レイヤーの配置レイヤーに存在する非ゼロの位置

create 3 pairs of prophet/enforcer
================================================================================

- pair_a = prophet_a and enforcer_a
- pair_b = prophet_b and enforcer_b
- pair_c = prophet_c and enforcer_c

training plan(prophet pre train)
================================================================================

- 1st stage
  - 既存棋譜の各手を自分視点と見て盤面と勝敗の対応を学習する
    - ロスは（手数/総手数）で適当に重み付けする。終盤ほど重視する。

training plan(enforcer pre train)
================================================================================

- 1st stage
  - 自分の手番の盤面について合法な手の集合を出力することを学習する
- 2nd stage
  - 既存棋譜の勝者視点を学習する
  - 自分の手番の盤面について棋譜が指した位置との差をロスとする

training plan(prophet/enforcer)
================================================================================

3つのペアを順繰りで対戦させていく
(先手,後手) in [(a,b), (b,c), (c,a), (b,a), (c,b), (a,c)]

- prophet が合格を出すまで
  - 自分の手番の盤面について enforcer が次手候補を算出する
    - 出力層で正の位置を値順に取り出す
  − 次手を指したあとの盤面について prophet が勝敗判定をする
  - prophet が勝利判定を出した手を実際に次に指す手とする

データ
================================================================================

- いわゆる棋譜からの変換データを扱う
- ただし後手については盤面を180度回転させる
  - つまり先手・後手どちらにあっても指し手は自分の駒が下から攻める棋譜を考える
- 勝敗の表現
  - 以下のデータ形式で考える
    − [float32: 自身が勝つ割合, float32: 相手が勝つ割合]
    - 論理的に等価なものを2次元ベクトルにしているのはラベリングタスクの流儀である

必要なプログラム
================================================================================

- prophet model
  - train from records
      - model performance test
- enforcer model
  - train from random data(candedates)
      - model performance test
  - train from records(candedates)
      - model performance test
  - train from records(determin)
      - model performance test
- autonomous training
  - with prophet model
  - without prophet model
- autonomous playing
- floodgate client
- usi backend

ワークサイクル
================================================================================

- model root(eg, ginkgo-2016-10-25)
  - ginkgo (source dir)
  - models
  - logs
  - autonomous-records
    - 20161025-101005-999999
      0000000.sfenx
      0000001.sfenx
      0000002.sfenx
  - 2chkifu-data

storage/directory
================================================================================

- ginkgo
  - ginkgo
  - data
  - series
    - series-001
      - 00000000-000000-000-initial
      - 20161101-101005-008-train-prophet-with-records
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-records
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-records
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-records
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-records
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-records
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-random
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-random
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-random
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-random
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-random
      - YYYYmmdd-HHMMSS-nnn-auto-match
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-match-result
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-match-result
      - YYYYmmdd-HHMMSS-nnn-auto-match
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-match-result
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-match-result
      - YYYYmmdd-HHMMSS-nnn-auto-match
      - YYYYmmdd-HHMMSS-nnn-train-prophet-with-match-result
      - YYYYmmdd-HHMMSS-nnn-train-enforcer-with-match-result
    - series-002
    - series-003

workflow
================================================================================

```bash
# assume 130 turn per match

# pre train prophet
python train_prophet_with_records.py --out=prophet_pret_e000.ckpt --optimizer=...
python train_prophet_with_records.py --in=prophet_pret_e000.ckpt --out=prophet_pret_e001.ckpt --optimizer=...
python train_prophet_with_records.py --in=prophet_pret_e001.ckpt --out=prophet_pret_e002.ckpt --optimizer=...
python train_prophet_with_records.py --in=prophet_pret_e002.ckpt --out=prophet_pret_e003.ckpt --optimizer=...
python train_prophet_with_records.py --in=prophet_pret_e003.ckpt --out=prophet_pret_e004.ckpt --optimizer=...

cp prophet_pret_e004.ckpt latest_prophet_model.ckpt

# pre train enforcer
python train_enforcer_with_random.py --num=100000 --out=enforcer_random_e000.ckpt --optimizer=...

cp enforcer_random_e000.ckpt latest_prophet_model.ckpt

# pre train enforcer
python train_enforcer_with_records.py --in=enforcer_random_e000.ckpt --out=enforcer_pret_e000.ckpt --optimizer=...
python train_enforcer_with_records.py --in=enforcer_pret_e000.ckpt   --out=enforcer_pret_e001.ckpt --optimizer=...
python train_enforcer_with_records.py --in=enforcer_pret_e001.ckpt   --out=enforcer_pret_e002.ckpt --optimizer=...
python train_enforcer_with_records.py --in=enforcer_pret_e002.ckpt   --out=enforcer_pret_e003.ckpt --optimizer=...
python train_enforcer_with_records.py --in=enforcer_pret_e003.ckpt   --out=enforcer_pret_e004.ckpt --optimizer=...

cp enforcer_pret_e004.ckpt latest_enforcer_model.ckpt

# generate autonomous matching log

for i in {0..10}; do
    DATENAME=`date +%Y%m%d-%H%M%S-%3N`

    python self_match.py --match-num=5000 \
        --prefetch-depth=3 \
        --prefetch-width=3 \
        --matching-logdir=latest_matches \
        --enforcer=latest_enforcer_model.ckpt \
        --prophet=latest_prophet_model.ckpt

    python train_prophet_with_records.py  --in=latest_prophet_model.ckpt --out=latest_prophet_model.ckpt --optimizer=...
    python train_enforcer_with_records.py --in=latest_enforcer_model.ckpt --out=latest_enforcer_model.ckpt --optimizer=...
done
```

multi GPU and distribution
================================================================================


