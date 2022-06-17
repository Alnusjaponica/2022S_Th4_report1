# 2022S_Th4_report1
 2022年度Sセメスター木４数理最適化特論のレポート課題の実装

### 仮想環境の導入・依存ライブラリのインストール
[Poetry公式](https://python-poetry.org)に従ってPoetryを導入する。
`2022S_Th4_report1/`で以下を実行し、仮想環境の作成と必要なライブラリの導入をする。
```sh
poetry install
```

同様に、`2022S_Th4_report1/`で以下を実行すると課題のグラフを得る。

```sh {.copy}
poetry run python -m Th4_report --fig [図の番号]
```

仮想環境の削除は以下を実行。
```sh
poetry env remove [仮想環境名]
```

