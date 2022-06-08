import random
import numpy
import concurrent

import asyncio

import deap.base
import deap.creator
import deap.tools
import deap.algorithms
import deap.gp


class God(object):

	def __init__(
		self,
		population_size=10,
		generations_count=10,
		tournament_size=7,
		max_init_individuals=5,
		crossover_probability=0.6,
		mutation_probability=0.2,
		division=1,
	):
		self.PopulationSize = population_size
		self.GenerationsCount = generations_count
		self.TournamentSize = tournament_size
		self.MaximumInitIndividuals = max_init_individuals
		self.CrossoverProbability = crossover_probability
		self.MutationProbability = mutation_probability
		self.Division = division

		self.BestSpiritData = list()

		for i in range(0, division):
			self.BestSpiritData.append(None)

		self.Executor = concurrent.futures.ThreadPoolExecutor(
			max_workers=division,
			thread_name_prefix="_god"
		)

		deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
		deap.creator.create("Individual", deap.gp.PrimitiveTree, fitness=deap.creator.FitnessMin)


	async def inspire(self, prophet, max_count=100, loop=None):

		if loop is None:
			loop = asyncio.get_event_loop()

		tasks = []
		spirit = prophet.Spirit

		# Run the evolutionary algorithm on a predefined number of workers
		for i in range(0, self.Division):
			spirit_divided = spirit.divide()
			spirit_divided.reset()

			tasks.append(
				loop.run_in_executor(
					self.Executor,
					self._inspire_worker,
					spirit_divided,
					max_count,
					i
				)
			)

		done, pending = await asyncio.wait(tasks)

		best_of_best = None

		# Select the best result
		for spirit_data in self.BestSpiritData:

			if spirit_data is None:
				continue

			if best_of_best is None:
				best_of_best = spirit_data
				continue

			if spirit_data[0] < best_of_best[0]:
				best_of_best = spirit_data
				continue

		if best_of_best is None:
			return True

		# Assign the inspired spirit to the prophet
		result, spirit, graph = best_of_best

		prophet.Spirit = spirit
		prophet.LastResult = result
		prophet.LastGraph = graph
		prophet.Spirit.save()

		return False


	def _inspire_worker(self, spirit, max_count, index):

		def add(value1, value2):
			return min(max_count, value1 + value2)

		def sub(value1, value2):
			return max(1, value1 - value2)

		primitive_set = deap.gp.PrimitiveSetTyped("{}_{}".format(spirit.Name, index), [], int)

		for layer_method in spirit.AddLayerMethods:
			primitive_set.addPrimitive(layer_method, [int, int], int)

		primitive_set.addPrimitive(add, [int, int], int)
		primitive_set.addPrimitive(sub, [int, int], int)
		primitive_set.addEphemeralConstant("start_{}".format(index), lambda: random.randint(1, max_count), int)

		def evaluate(individual):
			deap.gp.compile(individual, primitive_set)
			result = spirit.incarnate(350, 32)

			if self.BestSpiritData[index] is None:
				self.BestSpiritData[index] = (result, spirit.divide(), deap.gp.graph(individual))

			else:
				previous_result, previous_spirit, graph_tuple = self.BestSpiritData[index]

				if result < previous_result:
					self.BestSpiritData[index] = (result, spirit.divide(), deap.gp.graph(individual))

			spirit.reset()
			return result,

		toolbox = deap.base.Toolbox()
		toolbox.register("expr_init", deap.gp.genFull, pset=primitive_set, min_=1, max_=self.MaximumInitIndividuals)
		toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, toolbox.expr_init)
		toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", evaluate)
		toolbox.register("select", deap.tools.selTournament, tournsize=self.TournamentSize)
		toolbox.register("mate", deap.gp.cxOnePoint)
		toolbox.register("expr_mut", deap.gp.genFull, min_=0, max_=self.MaximumInitIndividuals)
		toolbox.register("mutate", deap.gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)

		population = toolbox.population(n=self.PopulationSize)
		stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", numpy.mean)
		stats.register("std", numpy.std)
		stats.register("min", numpy.min)
		stats.register("max", numpy.max)

		deap.algorithms.eaSimple(
			population,
			toolbox,
			cxpb=self.CrossoverProbability,
			mutpb=self.MutationProbability,
			ngen=self.GenerationsCount,
			stats=stats
		)
