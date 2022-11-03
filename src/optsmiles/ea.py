from jmetal.algorithm.multiobjective import NSGAII, SPEA2
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.algorithm.singleobjective import GeneticAlgorithm, SimulatedAnnealing
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

import signal
import sys
from abc import ABC, abstractmethod

from .observers import PrintObjectivesStatObserver
from .utils.process import get_evaluator, cpu_count
from .operators import build_operators

# SOEA alternatives
soea_map = {
    'GA': GeneticAlgorithm,
    'SA': SimulatedAnnealing
}
# MOEA alternatives
moea_map = {
    'NSGAII': NSGAII,
    'SPEA2': SPEA2,
    'NSGAIII': NSGAIII
}

MAX_GENERATIONS = 100
KILL_DUMP = True

# Auxiliary functions ----------------------------------------------------

def filter_duplicates(population):
    """ Filters equal solutions from a population
    """

    def remove_equal(individual, population):
        filtered = []
        for other in population:
            if individual != other:
                filtered.append(other)
        return filtered

    fitered_list = []
    p = population
    while len(p) > 1:
        individual = p[0]
        fitered_list.append(individual)
        p = remove_equal(individual, p)
    if p:
        fitered_list.extend(p)
    return fitered_list

# Abstract EA -------------------------------------------------------------

class AbstractEA(ABC):

    def __init__(self, problem, max_generations=MAX_GENERATIONS,
                 mp=True, **kwargs):

        self.problem = problem
        self.max_generations = max_generations
        self.mp = mp
        self.final_population = None

    def run(self):
        """ Runs the optimization for the defined problem.
        The number of objectives is defined to be the number
        of evaluation functions in fevalution.
        """
        # Register signal handler for linux
        signal.signal(signal.SIGINT, self.__signalHandler)

        if self.problem.fevaluation is None or len(self.problem.fevaluation) == 0:
            raise ValueError("At leat one objective should be provided.")

        if self.problem.number_of_objectives == 1:
            final_pop = self._run_so()
        else:
            final_pop = self._run_mo()
        self.final_population = filter_duplicates(final_pop)
        return self.final_population

    def dataframe(self):
        """Returns a dataframe of the final population.

        :raises Exception: if the final population is empty or None.
        :return: Returns a dataframe of the final population
        :rtype: pandas.Dataframe
        """
        if not self.final_population:
            raise Exception("No solutions")
        table = [[x.smiles, x.values] +
                 x.fitness for x in self.final_population]
        import pandas as pd
        columns = ["SMILES", "Representation"]
        columns.extend([obj.short_str() for obj in self.problem.fevaluation])
        df = pd.DataFrame(table, columns=columns)
        return df

    def __signalHandler(self, signum, frame):
        if KILL_DUMP:
            print("Dumping current population.")
            try:
                pop = self._get_current_population()
                data = [s.toDict() for s in pop]
                import json
                from datetime import datetime
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y-%H%M%S")
                with open(f'dump-{dt_string}.json', 'w') as outfile:
                    json.dump(data, outfile)
            except Exception:
                print("Unable to dump population.")
            print("Exiting")
        sys.exit(0)

    @ abstractmethod
    def _run_so(self):
        raise NotImplementedError

    @ abstractmethod
    def _run_mo(self):
        raise NotImplementedError

    @ abstractmethod
    def _get_current_population(self):
        raise NotImplementedError

# EA ---------------------------------------------------------------------

class EA(AbstractEA):
    """
    EA running helper for JMetal.

    :param problem: The optimization problem.
    :param initial_population: (list) The EA initial population.
    :param max_generations: (int) The number of iterations of the EA (stopping criteria).
    """

    def __init__(self, problem, max_generations=MAX_GENERATIONS, mp=False, algorithm=None, **kwargs):

        super(EA, self).__init__(problem, max_generations=max_generations, mp=mp, **kwargs)

        self.algorithm_name = algorithm
        self.crossover, self.mutation = build_operators()
        self.max_evaluations = self.max_generations * self.problem.population_size

    def get_population_size(self):
        return self.population_size

    def _run_so(self):
        """ Runs a single objective EA optimization ()
        """
        self.problem.reset_cursor()
        if self.algorithm_name == 'SA':
            print("Running SA")
            self.mutation.probability = 1.0
            algorithm = SimulatedAnnealing(
                problem=self.problem,
                mutation=self.mutation.probability,
                termination_criterion=StoppingByEvaluations(
                    max_evaluations=self.max_evaluations)
            )

        else:
            print("Running GA")
            algorithm = GeneticAlgorithm(
                problem=self.problem,
                population_size=self.problem.population_size,
                offspring_population_size=self.problem.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
                selection=BinaryTournamentSelection(),
                termination_criterion=StoppingByEvaluations(
                    max_evaluations=self.max_evaluations)
            )

        algorithm.observable.register(observer=PrintObjectivesStatObserver())
        self.algorithm = algorithm
        algorithm.run()

        result = algorithm.solutions
        return result

    def _run_mo(self):
        """ Runs a multi objective EA optimization
        """
        self.problem.reset_cursor()
        if self.algorithm_name in moea_map.keys():
            f = moea_map[self.algorithm_name]
        else:
            if self.problem.number_of_objectives > 2:
                self.algorithm_name == 'NSGAIII'
            else:
                f = moea_map['SPEA2']

        args = {
            'problem': self.problem,
            'population_size': self.problem.population_size,
            'mutation': self.mutation,
            'crossover': self.crossover,
            'termination_criterion': StoppingByEvaluations(max_evaluations=self.max_evaluations)
        }

        if self.mp:
            args['population_evaluator'] = get_evaluator(
                self.problem, n_mp=cpu_count())

        print(f"Running {self.algorithm_name}")
        if self.algorithm_name == 'NSGAIII':
            args['reference_directions'] = UniformReferenceDirectionFactory(self.problem.number_of_objectives,
                                                                            n_points=self.problem.population_size-1)
            algorithm = NSGAIII(**args)
        else:
            args['offspring_population_size'] = self.problem.population_size
            algorithm = f(**args)

        
        algorithm.observable.register(
                observer=PrintObjectivesStatObserver())
        self.algorithm = algorithm
        algorithm.run()
        result = algorithm.solutions
        return result

    def _get_current_population(self):
        """Dumps the population for gracefull exit."""
        return self.algorithm.solutions
        

