########################################################################################################
#
# Large-Scale Evolution of Image Classiﬁers
#
# Supplementary Material
#
# S1.MethodsDetails
#
# This section contains additional implementation details, roughly following the order in Section 3.
# Short code snippets illustrate the ideas. The code is not intended to run on its own and it has
# been highly edited for clarity.
#
# In our implementation, each worker runs an outer loop that is responsible for selecting a pair
# of random individuals from the population. The individual with the highest ﬁtness usually becomes
# a parent and the one with the lowest ﬁtness is usually killed (Section 3.1). Occasionally, either
# of these two actions is not carried out in order to keep the population size close to a set-point:
#
########################################################################################################


def evolve_population(self):
    # Iterate indefinitely.
    while True:
        # Select two random individuals from the population.
        # 每次挑选两个个体
        valid_individuals = []
        for individual in self.load_individuals():  # Only loads the IDs and states.
            if individual.state in [TRAINING, ALIVE]:
                valid_individuals.append(individual)
        individual_pair = random.sample(valid_individuals, 2)

        for individual in individual_pair:
            # Sync changes from other workers from file-system. Loads everything else.
            # 同步系统中的其他更新
            individual.update_if_necessary()

            # Ensure the individual is fully trained.
            # 确保个体已经被训练过了
            if individual.state == TRAINING:
                self._train(individual)

        # Select by fitness (accuracy).
        # 挑选个体的fitness，(population过大->kill不好的)，反之(population过小->reproduce好的)
        individual_pair.sort(key=lambda i: i.fitness, reverse=True)
        better_individual = individual_pair[0]
        worse_individual = individual_pair[1]

        # If the population is not too small, kill the worst of the pair.
        # ?????? 宁写反了把
        if self._population_size() >= self._population_size_setpoint:
            self._kill_individual(worse_individual)

        # If the population is not too large, reproduce the best of the pair.
        if self._population_size() < self._population_size_setpoint:
            self._reproduce_and_train_individual(better_individual)

# Much of the code is wrapped in try-except blocks to handle various kinds of errors. These have
# been removed from the code snippets for clarity. For example, the method above would be wrapped
# like this:


def evolve_population(self):
    while True:
    try:
        # Select two random individuals from the population.
        ...
    except:
        except exceptions.PopulationTooSmallException:
            self._create_new_individual()
            continue
        except exceptions.ConcurrencyException:
            # Another worker did something that interfered with the action of this worker.
            # Abandon the current task and keep going.
            continue


graph = tf.Graph()
with graph.as_default():
    # Build the neural network using the ‘Model‘ class and the ‘DNA‘ instance.
    ...
    tf.Session.reset(self._master)
    with tf.Session(self._master, graph=graph) as sess:
        # Initialize all variables
        ...
        # Make sure we can inherit batch-norm variables properly.
        # The TF-slim batch-norm variables must be handled separately here because some
        # of them are not trainable (the moving averages).
        batch_norm_extras = [x for x in tf.all_variables() if (
            x.name.find('moving_var') != -1 or
            x.name.find('moving_mean') != -1)]

        # These are the variables that we will attempt to inherit from the parent.
        vars_to_restore = tf.trainable_variables() + batch_norm_extras
        # Copy as many of the weights as possible.
        if mutated_weights:
            assignments = []
            for var in vars_to_restore:
                stripped_name = var.name.split(': ')[0]
                if stripped_name in mutated_weights:
                    shape_mutated = mutated_weights[stripped_name].shape
                    shape_needed = var.get_shape()
                    if shape_mutated == shape_needed:
                        assignments.append(var.assign(
                            mutated_weights[stripped_name]))
            sess.run(assignments)
