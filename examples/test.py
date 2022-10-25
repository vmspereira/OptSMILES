from src.optsmiles.problems.problem import SmilesProblem
from src.optsmiles.ea import EA
from src.optsmiles.evaluation.evaluators import logP

f1 = logP()

problem = SmilesProblem([f1], initial_polulation='Fragment_MW_100_to_150.smi')
ea = EA(problem)
ea.run()




