# import tensorflow as tf 
# import os

# flags = tf.flags

# FLAGS = flags.FLAGS

# flags.DEFINE_integer(
#     "max_seq_length", 128,
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter "
#     "than this will be padded.")

# flags.DEFINE_string(
#     "export_dir", None,
#     "The dir where the exported model has been written.")

# flags.DEFINE_string(
#     "predict_file", None,
#     "The file of predict records")

# # dir_path = os.path.dirname('.') #current directory
# # exported_path= os.path.join(dir_path,  "1536315752")

# eval_input_fn = file_based_input_fn_builder(
#             input_file=eval_file,
#             seq_length=FLAGS.max_seq_length,
#             is_training=False,
#             drop_remainder=eval_drop_remainder)


# dataset = tf.data.TFRecordDataset(filenames).map(lambda r: parse(r))

# iterator = tf.data.Iterator.from_structure(dataset.output_types,
#                                            dataset.output_shapes)
# next_element = iterator.get_next()

# training_init_op = iterator.make_initializer(dataset)

#         result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

#         predictor   = tf.contrib.predictor.from_saved_model(full_model_dir)

# def load_tfrecord():
#     graph = tf.Graph()
#     with graph.as_default():
#         tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
#         record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.predict_file)
#         for string_record in record_iterator:
#             example = tf.train.Example()
#             example.ParseFromString(string_record)

#     features = {'x': tf.FixedLenFeature([2], tf.int64)}
#     data = []
#     for s_example in tf.python_io.tf_record_iterator(file_name):
#         example = tf.parse_single_example(s_example, features=features)
#         data.append(tf.expand_dims(example['x'], 0))
#     return tf.concat(0, data)

# data = load_tfrecord('test_tfrecord')

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         Y = sess.run([data])
#         print(Y)

# graph = tf.Graph()
# with graph.as_default():
#     with tf.Session() as sess:
#         tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
#         tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
#         tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
#         tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
#         tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
#         tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')
#         record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.predict_file)
#         for string_record in record_iterator:
#             example = tf.train.Example()
#             example.ParseFromString(string_record)


# def main():
#     with tf.Session() as sess:
#         tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)

#         # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)

#         model_input= tf.train.Example(features=tf.train.Features(feature={
#                 'x': tf.train.Feature(float_list=tf.train.FloatList(value=[6.4, 3.2, 4.5, 1.5]))        
#                 })) 

#         predictor= tf.contrib.predictor.from_saved_model(exported_path)

#         input_tensor=tf.get_default_graph().get_tensor_by_name("input_tensors:0")

#         model_input=model_input.SerializeToString()

#         output_dict= predictor({"inputs":[model_input]})

#         print(" prediction is " , output_dict['scores'])


# if __name__ == "__main__":
#     main()


import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "export_dir", None,
    "The dir where the exported model has been written.")

flags.DEFINE_string(
    "predict_file", None,
    "The file of predict records")

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
        tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
        tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
        tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
        tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
        tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')
        record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.predict_file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            input_ids = example.features.feature['input_ids'].int64_list.value
            input_mask = example.features.feature['input_mask'].int64_list.value
            label_ids = example.features.feature['label_ids'].int64_list.value
            segment_ids = example.features.feature['segment_ids'].int64_list.value
            result = sess.run(tensor_outputs, feed_dict={
                tensor_input_ids: np.array(input_ids).reshape(-1, FLAGS.max_seq_length),
                tensor_input_mask: np.array(input_mask).reshape(-1, FLAGS.max_seq_length),
                tensor_label_ids: np.array(label_ids),
                tensor_segment_ids: np.array(segment_ids).reshape(-1, FLAGS.max_seq_length),
            })
            print(result)
