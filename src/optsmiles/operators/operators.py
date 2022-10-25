import random
import copy
from jmetal.core.operator import Mutation, Crossover
from optsmiles.operators.smiles_click_chem import SmilesClickChem
from optsmiles.operators.execute_mutations import make_mutants, run_smiles_click_for_multithread
from optsmiles.problems.problem import SmilesSolution, smiles_encoder, smiles_decoder
from typing import List
import os

DATA_FILES = os.path.join(os.path.dirname(__file__),
                          'reaction_libraries')

class MutationContainer(Mutation[SmilesSolution]):
    """A container for the mutation operators.

    :param probability: (float) The probability of applying a mutation.
    :param mutators: (list) The list of mutators.

    """

    def __init__(self, probability: float = 0.5, mutators=[]):
        super(MutationContainer, self).__init__(probability=probability)
        self.mutators = mutators

    def execute(self, solution: SmilesSolution) -> SmilesSolution:
        # randomly select a mutator and apply it
        if random.random() <= self.probability:
            idx = random.randint(0, len(self.mutators) - 1)
            mutator = self.mutators[idx]
            return mutator.execute(solution)
        else:
            return solution

    def get_name(self):
        return 'Mutation container'


class NullCrossover(Crossover[SmilesSolution, SmilesSolution]):
    """
        Null Crossover
    """

    def __init__(self, probability: float = 0.1):
        super(NullCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[SmilesSolution]) -> List[SmilesSolution]:
        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Null Crossover'


class SingleIntMutation(Mutation[SmilesSolution]):
    """
    Mutates a single element
    """

    def __init__(self, probability: float = 0.1):
        super(SingleIntMutation, self).__init__(probability=probability)

    def execute(self, solution: SmilesSolution) -> SmilesSolution:
        if random.random() <= self.probability:
            mutant = copy.copy(solution.variables)
            index = random.randint(0, len(mutant) - 1)
            newElem = random.randint(
                solution.lower_bound, solution.upper_bound)
            mutant[index] = newElem
            solution.variables = mutant
        return solution

    def get_name(self):
        return 'Single Mutation'


class ClickMutation(Mutation[SmilesSolution]):
    """
    Mutates a Smile using in silico reactions
    """

    def __init__(self, probability: float = 0.1):
        super(ClickMutation, self).__init__(probability=probability)
        
        self.rxn_library_variables = ["all_rxns",
                                     DATA_FILES+"/All_Rxns_rxn_library.json",
                                     DATA_FILES+"/All_Rxns_functional_groups.json",
                                     DATA_FILES+"/complementary_mol_dir/"]


    def execute(self, solution: SmilesSolution) -> SmilesSolution:
        if random.random() <= self.probability:
            mutant = copy.copy(solution.variables)
            smile = smiles_decoder(mutant)

            new_mutation_smiles_list = []
            filter_object_dict = {}

            a_smiles_click_chem_object = SmilesClickChem(self.rxn_library_variables,
                                                         new_mutation_smiles_list,
                                                         filter_object_dict)

            ## take a single smilestring and perform SmileClick on it
            result_of_run = run_smiles_click_for_multithread(smile, a_smiles_click_chem_object)

            #result_of_run returns
            # list containing:
            # [0] the reaction product
            # [1] the id_number of the reaction
            # [2] the id for the complementary mol (None if it was a single reactant reaction)
            # or None if all reactions failed or input failed to convert to a sanitizable rdkit mol

            if result_of_run is not None:
                mutant = smiles_encoder(result_of_run[0])
            solution.variables = mutant
        return solution

    def get_name(self):
        return 'Click Mutation'

def build_operators():
    # TODO: make json configurable
    crossover = NullCrossover(0.0)
    mutators = []
    mutators.append(ClickMutation(1.0))
    mutations = MutationContainer(0.3, mutators=mutators)
    return crossover, mutations


