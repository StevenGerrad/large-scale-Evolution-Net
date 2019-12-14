# Many mutations modify the structure. Mutations to insert and excise vertex-edge pairs
# build up a main convolutional column, while mutations to add and remove edges can handle
# the skip connections. For example, the AddEdgeMutation can add a skip connection between
# random vertices.

import copy
import random


class AddEdgeMutation(Mutation):
    """Adds a single edge to the graph."""
    def mutate(self, dna):
        # Try the candidates in random order until one has the right connectivity.
        for from_vertex_id, to_vertex_id in self._vertex_pair_candidates(dna):
            mutated_dna = copy.deepcopy(dna)
            if (self._mutate_structure(mutated_dna, from_vertex_id,
                                       to_vertex_id)):
                return mutated_dna
        raise exceptions.MutationException()  # Try another mutation.

    def _vertex_pair_candidates(self, dna):
        """Yields connectable vertex pairs."""
        from_vertex_ids = _find_allowed_vertices(dna, self._to_regex, ...)
        if not from_vertex_ids:
            raise exceptions.MutationException()  # Try another mutation.
        random.shuffle(from_vertex_ids)

        to_vertex_ids = _find_allowed_vertices(dna, self._from_regex, ...)
        if not to_vertex_ids:
            raise exceptions.MutationException()  # Try another mutation.
        random.shuffle(to_vertex_ids)

        for to_vertex_id in to_vertex_ids:
            # Avoid back-connections.
            disallowed_from_vertex_ids, _ = topology.propagated_set(
                to_vertex_id)
            for from_vertex_id in from_vertex_ids:
                if from_vertex_id in disallowed_from_vertex_ids:
                    continue
                # This pair does not generate a cycle, so we yield it.
                yield from_vertex_id, to_vertex_id

    def _mutate_structure(self, dna, from_vertex_id, to_vertex_id):
        """Adds the edge to the DNA instance."""
        edge_id = _random_id()
        edge_type = random.choice(self._edge_types)
        if dna.has_edge(from_vertex_id, to_vertex_id):
            return False
        else:
            new_edge = dna.add_edge(from_vertex_id, to_vertex_id, edge_type,
                                    edge_id)
            ...
            return True
