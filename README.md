# LLM Tabular AutoML Agent (custom, no AutoML frameworks)

## Install
pip install -r requirements.txt

## Run training
python run_train.py --train path/to/train.csv --target target_col --time_budget_s 1200 --output_dir runs/my_exp

## Outputs
- runs/.../runs.jsonl  : all tried candidates with scores
- runs/.../best_model.pkl : best fitted pipeline
- runs/.../best_meta.json : best config + dataset profile

## LLM
By default runs with heuristic candidate generator.
To enable LLM, implement LLMClient in automl/llm_client.py (HTTP/local/API) and set use_llm=True.