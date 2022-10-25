from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions
from rdkit.Chem import EnumerateStereoisomers, GetSymmSSSR, AllChem 
from rdkit.Chem.Descriptors import qed
from rdkit import Chem
import pickle
from abc import ABCMeta, abstractmethod
from functools import reduce
import numpy as np

MIN_THRESHOLD = -np.inf
MAX_THRESHOLD = np.inf


class EvaluationFunction:
    """
    This abstract class should be extended by all evaluation functions.

    """
    __metaclass__ = ABCMeta

    def __init__(self, maximize=True, worst_fitness=MIN_THRESHOLD):
        self.worst_fitness = worst_fitness
        self.maximize = maximize

    @abstractmethod
    def _get_fitness_single(self, candidate):
        """
        Candidate :  Candidate beeing evaluated
        """
        return

    @abstractmethod
    def _get_fitness_batch(self, listMols):
        """
        listMols :  List of rdKit Mols beeing evaluated
        """
        return

    def get_fitness(self, candidate, batched=False):
        if batched:
            return self._get_fitness_batch(candidate)
        else:
            return self._get_fitness_single(candidate)

    @abstractmethod
    def method_str(self):
        return

    @abstractmethod
    def short_str(self):
        return ""

    def __str__(self):
        return self.method_str()

    def __call__(self, candidate, batched=False):
        return self.get_fitness(candidate, batched)


class DummiEvalFunction(EvaluationFunction):

    def get_fitness(self, smile):
        """
        candidate :  Candidate beeing evaluated
        args: additional arguments
        """
        return sum(smile) / len(smile)

    def method_str(self):
        return "Dummi"


class AggregatedSum(EvaluationFunction):
    """
    Aggredated Sum Evaluation Function 

    Arguments:
        fevaluation (list): list of evaluation functions
        tradeoffs (list) : tradeoff values for each evaluation function. If None, all functions have the same weight

    """

    def __init__(self, fevaluation, tradeoffs=None, maximize=True):
        super(AggregatedSum, self).__init__(
            maximize=maximize, worst_fitness=0.0)
        self.fevaluation = fevaluation
        if tradeoffs and len(tradeoffs) == len(fevaluation):
            self.tradeoffs = np.array(tradeoffs)
        else:
            self.tradeoffs = np.array(
                [1/len(self.fevaluation)] * (len(self.fevaluation)))

    def _get_fitness_single(self, candidate):
        res = []
        for f in self.fevaluation:
            res.append(f._get_fitness_single(candidate))
        return np.dot(res, self.tradeoffs)

    def _get_fitness_batch(self, listMols):

        evals = []
        for f in self.fevaluation:
            evals.append(f._get_fitness_batch(listMols))
        evals = np.transpose(np.array(evals))
        res = np.dot(evals, self.tradeoffs)
        return res

    def method_str(self):
        return "Aggregated Sum = " + \
               reduce(lambda a, b: a+" "+b, 
                     [f.method_str() for f in self.fevaluation],
                      "")


class logP(EvaluationFunction):
    """ Octanol-water partition coefficient (logP) """

    def __init__(self, maximize=True, worst_fitness=-100.0):
        super(logP, self).__init__(
            maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate):
        """ candidate :  Candidate beeing evaluated """
        print(candidate)
        mol = Chem.MolFromSmiles(candidate)

        if mol:
            score = Chem.Descriptors.MolLogP(mol)
        else:
            score = self.worst_fitness

        return score

    def _get_fitness_batch(self, listMols):
        """ candidate :  Candidate beeing evaluated """

        listScore = []
        for mol in listMols:
            if mol:
                listScore.append(Chem.Descriptors.MolLogP(mol))
            else:
                listScore.append(self.worst_fitness)

        return listScore

    def method_str(self):
        return "logP"


class numberLargeRings(EvaluationFunction):
    """number of large rings"""

    def __init__(self):
        super(numberLargeRings, self).__init__(
            maximize=False, worst_fitness=10.0)

    def _get_fitness_single(self, candidate):
        """ candidate :  Candidate beeing evaluated """

        mol = Chem.MolFromSmiles(candidate)

        if mol:
            ringsSize = [len(ring) for ring in GetSymmSSSR(mol)]

            if len(ringsSize) > 0:  # has rings
                largestRing = max(ringsSize)  # penalize largest ring
                if largestRing > 6:
                    return largestRing - 6.0

            return 0.0

        else:
            score = self.worst_fitness

        return score

    def _get_fitness_batch(self, listMols):
        """ candidate :  Candidate beeing evaluated """

        listScore = []
        for mol in listMols:
            if mol:
                ringsSize = [len(ring) for ring in GetSymmSSSR(mol)]
                if len(ringsSize) > 0:  # has rings
                    largestRing = max(ringsSize)  # penalize largest ring
                    if largestRing > 6:
                        score = largestRing - 6.0
                    else:
                        score = 0.0
                else:
                    score = 0.0
                listScore.append(score)

            else:
                listScore.append(self.worst_fitness)

        return listScore

    def method_str(self):
        return "largeRings"


class drd2Activity(EvaluationFunction):
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, configs):
        super(drd2Activity, self).__init__(maximize=True, worst_fitness=0.0)
        with open(configs.svmDRD2, "rb") as f:
            self.clf = pickle.load(f)

    def fingerprints_from_mol_single(cls, mol):
        fp = AllChem.GetMorganFingerprint(
            mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            nfp[0, nidx] += int(v)
        return nfp

    def _fingerprints_from_mol_batched(self, listMols):
        size = 2048
        nfp = np.zeros((len(listMols), size), np.int32)
        invalids = np.zeros(len(listMols), np.bool)

        for i, mol in enumerate(listMols):
            if mol:
                fp = AllChem.GetMorganFingerprint(
                    mol, 3, useCounts=True, useFeatures=True)
                for idx, v in fp.GetNonzeroElements().items():
                    nidx = idx % size
                    nfp[i, nidx] += int(v)
            else:
                invalids[i] = True
        return nfp, invalids

    def _get_fitness_single(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = self._fingerprints_from_mol_single(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return self.worst_fitness

    def _get_fitness_batch(self, listMols):

        fingerprints, invalids = self._fingerprints_from_mol_batched(listMols)

        listScores = self.clf.predict_proba(fingerprints)[:, 1]

        listScores[invalids] = self.worst_fitness

        return listScores

    def method_str(self):
        return "drd2"


class QED(EvaluationFunction):
    """
    Quantitative Estimation of Drug-likeness (QED)

    """

    def __init__(self):
        super(QED, self).__init__(maximize=True, worst_fitness=-10.0)

    def _get_fitness_single(self, candidate):
        """
        candidate :  Candidate beeing evaluated
        """
        mol = Chem.MolFromSmiles(candidate)

        if mol:
            score = qed(mol)
        else:
            score = self.worst_fitness

        return score

    def _get_fitness_batch(self, listMols):
        """
        candidate :  Candidate beeing evaluated
        """
        listScore = []
        for mol in listMols:
            if mol:
                try:
                    listScore.append(qed(mol))
                except Chem.rdchem.MolSanitizeException:
                    listScore.append(self.worst_fitness)
            else:
                listScore.append(self.worst_fitness)

        return listScore

    def method_str(self):
        return "QED"



class simpleChain(EvaluationFunction):
    """ Filter simple chains """

    def __init__(self):
        super(simpleChain, self).__init__(maximize=True, worst_fitness=100.0)

    def _get_fitness_single(self, candidate):

        mol = Chem.MolFromSmiles(candidate)
        if mol:
            if len(set(Chem.MolToSmiles(mol))-set(['C', 'O', '=', '(', ')'])) == 0:
                return self.worst_fitness
            else:
                return -1.0
        else:
            return self.worst_fitness

    def _get_fitness_batch(self, listMols):
        """
        candidate :  Candidate beeing evaluated
        """
        listScore = []
        for mol in listMols:
            if mol:
                if len(set(Chem.MolToSmiles(mol))-set(['C', 'O', '=', '(', ')'])) == 0:
                    listScore.append(self.worst_fitness)
                else:
                    listScore.append(-1.0)

            else:
                listScore.append(self.worst_fitness)

        return listScore

    def method_str(self):
        return "chains"



class Stereoisomers(EvaluationFunction):
    """ Stereoisomer Count Evaluation Function """

    def __init__(self):
        super(Stereoisomers, self).__init__(maximize=True, worst_fitness=0.0)
        self.opt = StereoEnumerationOptions(unique=True)

    def _get_fitness_single(self, candidate):
        """ candidate :  Candidate beeing evaluated """

        mol = Chem.MolFromSmiles(candidate)

        if mol:
            chiralCount = EnumerateStereoisomers.GetStereoisomerCount(
                mol, options=self.opt)
            if chiralCount < 5:
                score = 1                   # ]0;5[     -> 1
            else:
                # [5;+inf]  -> 1/ln(Count*100)
                score = 1.0 / np.log(chiralCount*100.0)

        else:
            score = self.worst_fitness

        return score

    def _get_fitness_batch(self, listMols):
        """ candidate :  Candidate beeing evaluated """

        listScore = np.zeros(len(listMols))
        listScore += self.worst_fitness

        for i, mol in enumerate(listMols):
            if mol:
                listScore[i] = EnumerateStereoisomers.GetStereoisomerCount(
                    mol, options=self.opt)

        # ]0;5[     -> 1
        listScore[listScore < 5] = 1.0
        # [5;+inf]  -> 1/ln(Count*100)
        listScore[listScore >= 5] = 1.0 / \
            np.log(listScore[listScore >= 5] * 100.0)

        return listScore

    def method_str(self):
        return "Stereoisomer count"

