import tensorflow as tf

filename_queue = tf.train.string_input_producer(['inflow.tfrecord'], num_epochs=None)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
feature = tf.parse_single_example(serialized_example, features={'date':tf.FixedLenFeature([], tf.string), 'mat': tf.FixedLenFeature([], tf.string)})
date = feature['date']
mat_raw = feature['mat']
mat = tf.decode_raw(mat_raw, tf.int32)
mat = tf.reshape(mat, [101, 71])
date_batch, mat_batch = tf.train.batch([date, mat], batch_size=93, num_threads=2)

#queue = tf.FIFOQueue([94, 94], ["string", "int32"])
#enqueue_op = queue.enqueue([date, mat])
#out = queue.dequeue_many(1)

#qr = tf.train.QueueRunner([queue, [enqueue_op]*4])
#coord = tf.train.Coordinator()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(2):
        date1, mat1 = sess.run([date_batch, mat_batch])
        print(date1)
    coord.request_stop()
    coord.join(threads)

