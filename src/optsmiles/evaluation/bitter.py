from .evaluators import EvaluationFunction
from rdkit.Chem import MolFromSmiles, Lipinski
import rdkit.Chem.Descriptors as Descriptors
import functools
import joblib
import numpy as np
import os
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

class Bitter(EvaluationFunction):
    def __init__(self):
        super(Bitter, self).__init__(maximize=True, worst_fitness=0.0)
        DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
        MODEL_FILE = os.path.join(DATA_PATH, 'bitter.plk')
        
        self.pipe = joblib.load(MODEL_FILE)

        self.descriptors = (
               (Lipinski.NumAromaticRings, "NumAromaticRings"),
               (Lipinski.NumHAcceptors, "HBondAcceptors"),
               (Lipinski.NumHDonors, "HBondDonors"),
               (Descriptors.MolWt, "MolecularWeight"),
               (Descriptors.MinPartialCharge, "MinPartialCharge"),
               (Descriptors.MaxPartialCharge, "MaxPartialCharge"),
               (Descriptors.FpDensityMorgan1, "FPDensityMorgan1"),
               (Descriptors.FpDensityMorgan2, "FPDensityMorgan2"),
               (Descriptors.FpDensityMorgan3, "FPDensityMorgan3"),
              )

    def fd(self,smile):
        def _get_descriptor(smiles, fun):
            try:
                result = fun(MolFromSmiles(smiles))
            except:
                result = None
            return result
        d = []
        for descriptor, name in self.descriptors:
            f = functools.partial(_get_descriptor, fun=descriptor)
            d.append(f(smile))
        return np.array(d)
 

    def _get_fitness_single(self, candidate):
        """ candidate :  Candidate beeing evaluated """
        descrp = np.array([self.fd(candidate)])
        return self.pipe.predict_proba(descrp)[:,1][0]
    
    def _get_fitness_batch(self, X):
        descrp = np.array([self.fd(x) for x in X])
        return self.pipe.predict_proba(descrp)[:,1]

    def method_str(self):
        return "Bitter"

    def short_str(self):
        return "BIT"
