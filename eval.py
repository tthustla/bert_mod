import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import os

flags = tf.flags

FLAGS = flags.FLAGS

# flags.DEFINE_integer(
#     "max_seq_length", 128,
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter "
#     "than this will be padded.")

flags.DEFINE_string(
    "saved_dir", None,
    "The dir where the exported model has been written.")

flags.DEFINE_string(
    "model_dir", None,
    "The dir where the base model is.")

flags.DEFINE_string(
    "task_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

BERT_PRETRAINED_DIR = FLAGS.saved_dir
BERT_BASE_DIR = FLAGS.model_dir
TASK_DATA_DIR = FLAGS.task_data_dir
# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_BASE_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_BASE_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_BASE_DIR, 'bert_model.ckpt')
# DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

processor = run_classifier.ColaProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

train_examples = processor.get_train_examples(TASK_DATA_DIR)
num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

tpu_cluster_resolver = None
if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

# estimator = tf.contrib.estimator.SavedModelEstimator(BERT_PRETRAINED_DIR)

# Eval the model.
eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
eval_features = run_classifier.convert_examples_to_features(
    eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
eval_input_fn = run_classifier.input_fn_builder(
    features=eval_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=True)
result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

print(result)