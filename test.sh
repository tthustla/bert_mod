export BERT_BASE_DIR='chinese_L-12_H-768_A-12'
python3 test.py \
	--max_seq_length=128 \
	--predict_file=$GLUE_DIR/CoLA/dev.tsv \
	--export_dir=gs://${PROJECT_ID}/tmp/cola_model/1546433070
