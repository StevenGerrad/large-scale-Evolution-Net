
# Mutations act on DNA instances. The set of mutations restricts the space explored
# somewhat (Section 3.2). The following are some example mutations. The
# AlterLearningRateMutation simply randomly modiﬁes the attribute in the DNA:

import copy
import random


class AlterLearningRateMutation(Mutation):
    """Mutation that modifies the learning rate."""

    def mutate(self, dna):
        mutated_dna = copy.deepcopy(dna)

        # Mutate the learning rate by a random factor between 0.5 and 2.0,
        # uniformly distributed in log scale.
        factor = 2 ** random.uniform(-1.0, 1.0)
        mutated_dna.learning_rate = dna.learning_rate * factor
        return mutated_dna
