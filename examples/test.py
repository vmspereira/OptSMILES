from optsmiles.problems.problem import SmilesProblem
from optsmiles.ea import EA
from optsmiles.evaluation.evaluators import logP
from optsmiles.evaluation.bitter import Bitter

f1 = logP()
f2 = Bitter()

problem = SmilesProblem([f1,f2], initial_polulation='Fragment_MW_100_to_150.smi')
ea = EA(problem)
ea.run()
df=ea.dataframe()
df.to_csv('results.csv')




