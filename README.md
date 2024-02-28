# Yomi and ICD-10 code estimator from disease name

病名や症状からそのふりがなとICD-10コードを推定するWebアプリ

Link: [https://aoi.naist.jp/diseasetoyomi/](https://aoi.naist.jp/diseasetoyomi/)

## Caution

ファイルアップロード機能による推定時に，プログレスバーが表示されず，一向に処理が終わらない場合が確認されています．

その場合は，以下の手順に従って，コンテナを一度削除して，再びコンテナを起動させてください．

```
$ cd disease2yomi/
$ docker ps
$ docker rm -f <CONTAINER ID>
$ make start
```

## Reference

- [https://github.com/sy-zvjkqv/text2location](https://github.com/sy-zvjkqv/text2location)
- [https://github.com/sociocom/risk-extractor](https://github.com/sociocom/risk-extractor)
