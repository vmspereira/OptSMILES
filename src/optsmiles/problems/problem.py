
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from abc import ABC
import warnings
from ..utils.dominance import dominance_test
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluation.evaluators import EvaluationFunction

# define SMILES characters
SMILES_CHARS = [' ', '~', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6','7', '8', '9','=', '@','A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P','R', 'S', 'T', 'V', 'X', 'Z','[', '\\', ']','a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's','t', 'u']

# define encoder and decoder
smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))


def smiles_encoder(smiles:str) -> List[int]:
    return [smi2index[c] for c in smiles]


def smiles_decoder(x:List[int]) -> str:
    return "".join([index2smi[i] for i in x])

# Define a jMetal Solution

class SmilesSolution(Solution[int]):
    """ Class representing a SMILES solution encoded as a list of integers """

    def __init__(self, 
                 lower_bound: int,
                 upper_bound: int,
                 number_of_variables: int,
                 number_of_objectives: int):
        super(SmilesSolution, self).__init__(number_of_variables, number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.smiles = None

    def __eq__(self, solution:"SmilesSolution") -> bool:
        if isinstance(solution, self.__class__):
            return self.variables.sort() == solution.variables.sort()
        return False

    # JMetal consideres all problems as minimization
    # Based on pareto dominance

    def __gt__(self, solution:"SmilesSolution") -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) == 1
        return False

    def __lt__(self, solution:"SmilesSolution") -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) == -1
        return False

    def __ge__(self, solution:"SmilesSolution") -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) != -1
        return False

    def __le__(self, solution:"SmilesSolution") -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) != 1
        return False

    def __copy__(self) -> "SmilesSolution":
        new_solution = SmilesSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()
        new_solution.smiles = self.smiles.copy()

        return new_solution

    def get_representation(self):
        """
        Returns a representation of the candidate
        """
        return self.variables

    def get_fitness(self):
        """
        Returns the candidate fitness list
        """
        return self.objectives

    def __str__(self):
        return " ".join((self.variables))


class SmilesProblem(Problem[SmilesSolution], ABC):
    """Class representing SMILES problems."""

    def __init__(self, fevaluation:List["EvaluationFunction"]=[], **kwargs):
        super(SmilesProblem, self).__init__()
        self.lower_bound = 0
        self.upper_bound = len(SMILES_CHARS)-1

        self.fevaluation = fevaluation

        self.number_of_objectives = len(self.fevaluation)
        self.obj_directions = []
        self.obj_labels = []
        for f in self.fevaluation:
            self.obj_labels.append(str(f))
            if f.maximize:
                self.obj_directions.append(self.MAXIMIZE)
            else:
                self.obj_directions.append(self.MINIMIZE)

        # default population size is set to 100
        self.population_size = kwargs.get('population_size', 100)

        pop = kwargs.get('initial_polulation',None) 
        if not pop:
            raise ValueError('You need to provide a smi file or a list of SMILES.')
        elif isinstance(pop, str):
            self.initial_polulation = self.population_from_file(pop)
        elif isinstance(pop, list):
            self.initial_polulation = pop
            self.population_size = len(pop)
        else:
            raise ValueError('Invalid initial population') 

        self._cursor = 0

    def population_from_file(self, filename):
        """ Reads mols from a smi file"""
        from rdkit.Chem import SmilesMolSupplier, MolToSmiles
        supplier = SmilesMolSupplier(filename)
        pop =[]
        cursor = 0
        while cursor < self.population_size:
            mol = next(supplier)
            if mol is None:
                warnings.warn('Molecule {} could not be parsed'.format(cursor))
            else:
                pop.append(MolToSmiles(mol, isomericSmiles=False))
            cursor+=1
        return pop

    def create_solution(self) -> SmilesSolution:
        new_solution = SmilesSolution(
            self.lower_bound, self.upper_bound, 1, self.number_of_objectives)
        new_solution.smiles = self.initial_polulation[self._cursor]
        new_solution.variables = smiles_encoder(new_solution.smiles)
        self._cursor += 1
        return new_solution

    def reset_cursor(self):
        self._cursor = 0

    def evaluate_solution(self, solution, decode=True):
        """
        Evaluates a single solution

        :param solution: The solution to be evaluated.
        :param decode: If the solution needs to be decoded.
        :returns: A list of fitness.
        """
        smiles = ''
        if decode:
            smiles = smiles_decoder(solution)
        else:
            smiles = solution

        p = []
        for f in self.fevaluation:
            v = f(smiles)
            p.append(v)
        return p

    def evaluate(self, solution: SmilesSolution) -> SmilesSolution:
        candidate = smiles_decoder(solution.variables)
        solution.smiles = candidate
        p = self.evaluate_solution(candidate, decode=False)
        for i in range(len(p)):
            # JMetalPy only deals with minimization problems
            if self.obj_directions[i] == self.MAXIMIZE:
                solution.objectives[i] = -1 * p[i]
            else:
                solution.objectives[i] = p[i]
        return solution

    def evaluator(self, candidates, *args):
        res = []
        for candidate in candidates:
            res.append(self.evaluate(candidate))
        return res

    def get_name(self) -> str:
        return self.problem.get_name()
