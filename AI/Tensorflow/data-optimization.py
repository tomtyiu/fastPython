dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.cache()                     # Cache data in memory
dataset = dataset.shuffle(buffer_size=1000)   # Shuffle with an appropriate buffer
dataset = dataset.batch(32)                   # Batch data
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
