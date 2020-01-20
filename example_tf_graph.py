import tensorflow as tf

# disable eager execution, so we can export the tf graph
# and graph_def
tf.compat.v1.disable_eager_execution()

# create feed forward neural net
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# load data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize rgb colors of the images
X_train, X_test = X_train / 255.0, X_test / 255.0


# run the model
model.fit(X_train, y_train)

s = model.evaluate(X_test, y_test)
print("loss: ", s[0], "accuracy: ", s[1])


# generate a tf session so we can use tensorboard for
# viewing the tf graph
with tf.compat.v1.Session() as sess:
    tf.io.write_graph(sess.graph_def, "tf_graphs", "model.pb")
    tf.compat.v1.summary.FileWriter("tf_logs/", sess.graph).close()
