import time
import asyncio

import prase


if __name__ == "__main__":
	"""
	Mythological note:
	Pythia used to be the name of the high priestess of god Apollo at Delphi.
	Her name refers to the serpent named Python once living under Delphi.
	"""

	loop = asyncio.get_event_loop()
	current_time = int(time.time())

	# Evolutionary algorithm
	apollo = prase.God(generations_count=5)

	# Neural networks and data storage
	pythia = prase.TimeProphet(
		name="pythia",
		timestamp_field="timestamp",
		span=3,
		resolution=10
	)

	print("\n\nApollo and Pythia are prepared.")

	# Try to make predictions here, if the model was already build (i. e. loaded from a file)
	if pythia.is_inspired():
		print("Pythia is already inspired and can make a prediction here.")
		predicted_value = pythia.predict([
			{"timestamp": current_time + 100 + 10},
			{"timestamp": current_time + 100 + 15},
			{"timestamp": current_time + 100 + 20},
			{"timestamp": current_time + 100 + 25},
			{"timestamp": current_time + 100 + 30},
		])
		print("The predicted value from Pythia is: '{}'.".format(predicted_value))

	# Building the model

	# Observe the real-time data
	print("Pythia observes the data stream.")

	# The following code is for illustration, it should be connected to a real data source
	for i in range(0, 10000):

		if i % 6 == 0:
			continue

		event = {
			"timestamp": current_time + i * 5,
		}

		pythia.observe(event)

	# The following code could happen once per some time period

	# Transform the observed data into data suitable for the model
	print("Pythia is meditating.")
	pythia.meditate()

	# Create, train, test and save the neural network model using genetic programming
	print("Pythia is getting inspired by Apollo.")

	loop.run_until_complete(
		apollo.inspire(pythia)
	)

	# Describe the model
	print("Pythia's spirit looks as follows.")
	pythia.Spirit.talk()

	# Draw both the model and genetic primitive tree
	print("Drawing Pythia to files.")
	pythia.draw()

	# Predict the following values
	print("Pythia is making a prediction.")
	predicted_value = pythia.predict([
		{"timestamp": current_time + 100 + 10},
		{"timestamp": current_time + 100 + 15},
		{"timestamp": current_time + 100 + 20},
		{"timestamp": current_time + 100 + 25},
		{"timestamp": current_time + 100 + 30},
	])
	print("The predicted value from Pythia is: '{}'.".format(predicted_value))
