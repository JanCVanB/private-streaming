import numpy
import random


class Curator:
    """Database curator that creates and queries databases for analysts

    :param float epsilon: privacy parameter
    :ivar float epsilon: privacy parameter
    :ivar network: private content network
    """

    def __init__(self, epsilon=100):
        self.epsilon = epsilon
        self.network = None

    def exponential_mechanism(self, utilities):
        """Return a list of probabilities generated from the utilities

        :return: list of probabilities generated from the utilities
        :rtype: list
        """
        weights = numpy.exp(0.5 * self.epsilon * numpy.array(utilities))
        return list(weights / sum(weights))

    def query(self, sequence_length, preferences):
        """Return a list of nodes picked with the exponential mechanism

        Each step in this sequence is picked by running the exponential
        mechanism on the utilities of links leaving the previous node.

        The utility is 0 for any undefined links, but the probability is not.

        The first node is always the first node in self.network.nodes.

        :param int sequence_length: length of sequence to query
        :param tuple preferences: user input at each node in self.network.nodes
        :return: list of nodes picked with the exponential mechanism
        :rtype: list
        """
        sequence = [random.choice(self.network.nodes)]
        for sequence_step in range(sequence_length - 1):
            this_node = sequence[-1]
            this_index = self.network.nodes.index(this_node)
            utilities = [0 if (node not in this_node.links
                               or preferences[this_index] not in this_node.links[node])
                         else this_node.links[node][preferences[this_index]].utility
                         for node in self.network.nodes]
            probabilities = self.exponential_mechanism(utilities)
            shuffled_probabilities = probabilities[:]
            random.shuffle(shuffled_probabilities)
            next_probability = numpy.random.choice(shuffled_probabilities,
                                                   p=shuffled_probabilities)
            # In case of equal probabilities, eliminate earlier-in-list bias
            next_probability_indices = [i for i, x in enumerate(probabilities)
                                        if x == next_probability]
            next_node_index = random.choice(next_probability_indices)
            next_node = self.network.nodes[next_node_index]
            sequence.append(next_node)
        return [node.name for node in sequence]
