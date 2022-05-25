# %% [markdown]
# Quadratic function
# 

# %%
import tensorflow as tf
import numpy as np

# %%
class Rmodel (tf.keras.Model):
    def __init__(self):
        super(Rmodel, self).__init__()
        self.d1 = tf.keras.layers.Dense(10)
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)
model = Rmodel()

# %%
loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy= tf.keras.metrics.MeanAbsoluteError(
    name='mean_absolute_error', dtype=None
)


# %%
Scalar = tf.constant(4, shape=(1,1))

# %%
train_x = np.array([  tf.constant(x, shape=(1,1)) for x in range(100) ])
train_y = np.array([  tf.constant(x**(1/2), shape=(1,1)) for x in train_x ])

# %%
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(x, training=True)
    loss = loss_object(y, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(y, predictions)

# %%
EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  

  for x, y in zip(train_x, train_y):
    train_step(x, y)


  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}'
    )


