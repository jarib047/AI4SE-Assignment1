# 0) Install deps
pip install "transformers>=4.44" tokenizers datasets accelerate sentencepiece astor

# 1) Extract dataset
python function_extractor.py

# 2) Pre-train
python main.py \
  --functions_json datasets/functions.json \
  --workdir runs/ai4se \
  --do_pretrain --pretrain_rows 150000 --pretrain_epochs 1

# 3) Fine-tune (mask one if-condition; predict it)
python main.py \
  --functions_json datasets/functions.json \
  --workdir runs/ai4se \
  --do_finetune --finetune_rows 50000 --finetune_epochs 2

# 4) Evaluate (write generated-testset.csv)
python train_if_recommender.py \
  --functions_json datasets/functions.json \
  --workdir runs/ai4se \
  --do_eval --eval_rows 5000 --generated_csv generated-testset.csv
