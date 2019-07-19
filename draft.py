from typing import Optional, List

from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.standard_benchmarks import smina_dock
from guacamol.utils.proteins import get_proteins

class RandomGenerator(GoalDirectedGenerator):

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        return ['NCCc1c[nH]c2ccc(OCCO)cc12', 'N#CCN1CCC(c2c[nH]cn2)CC1', 'IBc1cn[nH]c1']


if __name__ == '__main__':
    # print(get_proteins())
    generator = RandomGenerator()
    benchmark = smina_dock('5ht1b')
    results = benchmark.assess_model(generator)
    print(results)
