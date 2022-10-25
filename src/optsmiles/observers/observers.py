import numpy

class PrintObjectivesStatObserver():

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency
        self.first = True

    def fitness_statistics(self, solutions, problem):
        """Return the basic statistics of the population's fitness values.
        :param list solutions: List of solutions.
        :param problem: The jMetalPy problem.
        :returns: A statistics dictionary.
        """
        def minuszero(value):
            return round(value, 6)

        stats = {}
        first = solutions[0].objectives
        # number of objectives
        n = len(first)
        for i in range(n):
            direction = problem.obj_directions[i]
            factor = 1 if direction == problem.MINIMIZE else -1
            f = [(factor * p.objectives[i]) for p in solutions]
            if direction == problem.MAXIMIZE:
                worst_fit = min(f)
                best_fit = max(f)
            else:
                worst_fit = max(f)
                best_fit = min(f)
            med_fit = numpy.median(f)
            avg_fit = numpy.mean(f)
            std_fit = numpy.std(f)
            stats['obj_{}'.format(i)] = {'best': minuszero(best_fit), 'worst': minuszero(worst_fit),
                                         'mean': minuszero(avg_fit), 'median': minuszero(med_fit), 'std': minuszero(std_fit)}
        return stats

    def stats_to_str(self, stats, evaluations, title=False):
        if title:
            title = "Eval(s)|"
        values = " {0:>6}|".format(evaluations)

        for key in stats:
            s = stats[key]
            if title:
                title = title + "     Worst      Best    Median   Average   Std Dev|"
            values = values + "  {0:.6f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}|".format(s['worst'],
                                                                                      s['best'],
                                                                                      s['median'],
                                                                                      s['mean'],
                                                                                      s['std'])
        if title:
            return title + "\n" + values
        else:
            return values

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']
        problem = kwargs['PROBLEM']
        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                stats = self.fitness_statistics(solutions, problem)
                message = self.stats_to_str(stats, evaluations, self.first)
                self.first = False
            else:
                fitness = solutions.objectives
                res = abs(fitness[0])
                message = 'Evaluations: {}\tFitness: {}'.format(
                    evaluations, res)
            print(message)
