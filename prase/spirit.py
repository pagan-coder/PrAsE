import random
import string

import numpy

import tensorflow.keras
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.utils
import tensorflow.keras.callbacks


class Spirit(object):
	"""
	An interface every spirit must implement.
	"""

	def __init__(self, name="spirit"):
		self.Name = name
		self.Model = None
		self.AddLayerMethods = None
		self.TrainInput = None
		self.TrainOutput = None
		self.TestInput = None
		self.TestOutput = None


	def divide(self):
		pass


	def reset(self):
		pass


	def incarnate(self, epochs: int = 1, batch_size: int = 1, verbose: int = 0) -> float:
		pass


	def predict(self, data):
		pass


	def save(self):
		self.Model.save("{}.h5".format(self.Name))


	def load(self):
		self.Model = tensorflow.keras.models.load_model("{}.h5".format(self.Name))


	def talk(self):
		self.Model.summary()


	def draw(self):
		tensorflow.keras.utils.plot_model(self.Model, "{}.png".format(self.Name), show_shapes=True)



class LSTMSpirit(Spirit):
	"""
	Long Short-Term Memory Neural Network
	"""


	def __init__(self, name="lstmspirit", span=5, lstm_count=100, activation="relu", kernel_initializer="he_normal", optimizer="adam"):
		super().__init__(name=name)

		self.Span = span
		self.LSTMCount = lstm_count
		self.Activation = activation
		self.KernelInitializer = kernel_initializer
		self.Optimizer = optimizer

		self.AddLayerMethods = [
			self.dense,
		]

		self.reset()


	def divide(self):
		spirit = LSTMSpirit(
			name=self.Name,
			span=self.Span,
			lstm_count=self.LSTMCount,
			activation=self.Activation,
			kernel_initializer=self.KernelInitializer,
			optimizer=self.Optimizer
		)

		spirit.TrainInput = self.TrainInput
		spirit.TrainOutput = self.TrainOutput
		spirit.TestInput = self.TestInput
		spirit.TestOutput = self.TestOutput
		spirit.Model = self.Model

		return spirit


	def reset(self):
		self.Model = tensorflow.keras.Sequential()
		self.Model.add(
			tensorflow.keras.layers.LSTM(
				self.LSTMCount,
				name="LSTM_" + self._get_random_name(),
				activation=self.Activation,
				kernel_initializer=self.KernelInitializer,
				input_shape=(self.Span, 1)
			)
		)


	def incarnate(self, epochs: int = 1, batch_size: int = 1, verbose: int = 0) -> float:
		# Add batch normalization (whitening)
		self.Model.add(
			tensorflow.keras.layers.BatchNormalization(name="batch_" + self._get_random_name())
		)

		# There is only one output
		self.Model.add(
			tensorflow.keras.layers.Dense(1, name="dense_last_" + self._get_random_name())
		)

		# Make sure the model is not overtrained
		early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

		# Compile and fit the model
		self.Model.compile(optimizer=self.Optimizer, loss="mse", metrics=["mae"])
		self.Model.fit(
			self.TrainInput,
			self.TrainOutput,
			epochs=epochs,
			batch_size=batch_size,
			verbose=verbose,
			validation_data=(self.TestInput, self.TestOutput),
			callbacks=[early_stopping]
		)

		# Evaluate the model
		mse, mae = self.Model.evaluate(self.TestInput, self.TestOutput, verbose=verbose)
		return numpy.sqrt(mse)


	def predict(self, data):
		return self.Model.predict(data)


	def dense(self, count: int, previous: int):
		self.Model.add(
			tensorflow.keras.layers.Dense(
				count,
				name="dense_" + self._get_random_name(),
				activation=self.Activation,
				kernel_initializer=self.KernelInitializer
			)
		)

		return count


	def _get_random_name(self):
		return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
