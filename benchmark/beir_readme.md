先用beir_ingestion.py导入数据
再用beir_eval.py评估
示例：
```
python beir_ingestion.py --collection-name beir --dataset scifact
python beir_eval.py --collection-name beir --dataset scifact

```