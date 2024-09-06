# 概要
Axell AI Contest 2024(https://signate.jp/competitions/1374 )の3位解法。このコンペは4倍超解像モデルの精度(PSNRで評価)を競うものだったが、推論時間の制限(Tesla T4で1枚あたり0.035sec以内)があり、モデルの高速化も重要だった。

# モデル
EDSR(https://arxiv.org/abs/1707.02921 )をベースとするCNNを使用した。EDSRはSRResNetを改造したモデルであり、次のような特長がある。
1. 残差ブロックを利用するのでネットワークを深くできる。
2. アップサンプリングを最後に行うので計算量が少ない。
3. SRResNetからbatch normalization層を削除し、超解像タスクに最適化している。

残差ブロックは6つ、特徴マップ数は各層64、カーネルサイズは全て3とした。元論文のモデルとの違いは主に以下の6つ。
1. ReLUの代わりにPReLUを使用する。
2. 各残差ブロックの最後にPReLUを配置する。
3. アップサンプリング層の直前にPReLUを配置する。
4. 2倍のアップサンプリングを2回行うのではなく、1回で4倍にする。
5. アップサンプリング後に畳み込みを行わない。
6. L1誤差ではなくL2誤差を損失関数とする。

1, 2, 3, 6により精度が改善され、4, 5により精度と速度が改善された。

![model](./model.png "モデルの概要")

更に、高速化のための工夫として以下の2つを行った。
1. モデルを半精度化する。
2. 入力の(バッチサイズ, チャンネル数)を(3, 1)ではなく(1, 3)とする。

学習方法などの詳細は後日追記予定
