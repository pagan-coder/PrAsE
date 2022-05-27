import time
import random

import pygraphviz
import numpy

from .spirit import LSTMSpirit


class Prophet(object):
	"""
	An interface every prophet must implement.
	"""

	def __init__(self, name):
		self.Name = name
		self.Spirit = None
		self.LastResult = None
		self.LastGraph = None
		self.Data = None


	def observe(self, event):
		pass


	def meditate(self):
		pass


	def predict(self, events=None):
		pass


	def is_inspired(self):
		return self.LastResult is not None


	def draw(self):

		if self.Spirit is None or self.LastGraph is None:
			return

		# Plot the graph from god
		nodes, edges, labels = self.LastGraph

		gr = pygraphviz.AGraph()
		gr.add_nodes_from(nodes)
		gr.add_edges_from(edges)
		gr.layout(prog="dot")

		for i in nodes:
			n = gr.get_node(i)
			n.attr["label"] = labels[i]

		gr.draw("{}.png".format(self.Name))

		# Plot the spirit
		self.Spirit.draw()


class TimeProphet(Prophet):

	def __init__(self, name, timestamp_field, span, resolution):
		super().__init__(name)

		self.Spirit = LSTMSpirit(name="{}_s".format(name), span=span)

		try:
			self.Spirit.load()
			self.LastResult = 1

		except OSError:
			pass

		self.TimestampField = timestamp_field
		self.Span = span
		self.Resolution = resolution
		self.StartTime = int(time.time())

		self.Data = [0]
		self.DataLength = 1


	def observe(self, event):
		event_timestamp = event.get(self.TimestampField)

		if event_timestamp is None:
			return True

		event_position = (event_timestamp - self.StartTime) // self.Resolution

		# Store the event in the dataset
		if event_position >= self.DataLength:

			for i in range(0, event_position - self.DataLength + 1):
				self.Data.append(0)

			self.DataLength = len(self.Data)

		self.Data[event_position] += 1


	def meditate(self):
		input_data = []
		output_data = []

		for i in range(0, self.DataLength):
			end = i + self.Span

			if end >= self.DataLength:
				break

			part_input, part_output = self.Data[i:end], self.Data[end]
			input_data.append(part_input)
			output_data.append(part_output)

		input_data_array = numpy.asarray(input_data, dtype=numpy.int64)
		input_data_array = input_data_array.reshape(
			input_data_array.shape[0],
			self.Span,
			1
		)
		output_data_array = numpy.asarray(output_data, dtype=numpy.int64)

		for_test = random.randint(
			max(1, input_data_array.shape[1] // 16), max(input_data_array.shape[1] // 8, 2)
		)

		self.Spirit.TrainInput = input_data_array[:-for_test]
		self.Spirit.TrainOutput = output_data_array[:-for_test]
		self.Spirit.TestInput = input_data_array[-for_test:]
		self.Spirit.TestOutput = output_data_array[-for_test:]


	def predict(self, events=None):
		"""
		Predict for given events or for the last span.
		"""

		if events is None:
			data = [self.Data[-self.Span:]]
			predict_for = numpy.asarray(data, dtype=numpy.int64)
			predict_for = predict_for.reshape(
				predict_for.shape[0],
				self.Span,
				1
			)

		else:
			start_time = events[0][self.TimestampField]
			predict_for = numpy.zeros((1, self.Span, 1), dtype=numpy.int64)

			for event in events:
				event_position = (event[self.TimestampField] - start_time) // self.Resolution
				predict_for[event_position // self.Span][event_position % self.Span][0] += 1

		return self.Spirit.predict(predict_for)
