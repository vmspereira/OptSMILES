from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..problems.problem import SmilesSolution

def dominance_test(solution1:Union[List[float],SmilesSolution],
                   solution2:Union[List[float],SmilesSolution], 
                   maximize:bool=True) -> int:
    """
    Testes Pareto dominance

    :param solution1: The first solution.
    :param solution2: The second solution.
    :param maximize: (bool) maximization (True) or minimization (False)
    :returns: 1 : if the first solution dominates the second; -1 : if the second solution dominates the first; \
         0 : if non of the solutions dominates the other.

    """

    best_is_one = 0
    best_is_two = 0

    if isinstance(solution1, list):
        s1 = solution1
    else:
        s1 = solution1.fitness

    if isinstance(solution2, list):
        s2 = solution2
    else:
        s2 = solution2.fitness

    for i in range(len(s1)):
        value1 = s1[i]
        value2 = s2[i]
        if value1 != value2:
            if value1 < value2:
                best_is_one = 1
            if value1 > value2:
                best_is_two = 1

    if best_is_one > best_is_two:
        result = 1
    elif best_is_two > best_is_one:
        result = -1
    else:
        result = 0

    if not maximize:
        result = -1 * result

    return result
