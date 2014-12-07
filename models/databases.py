from itertools import product
import random


DEBUG = True


class Network:
    """Network of Nodes connected by weighted Links

    :param int size: number of Nodes
    :param float interactivity: number of user inputs possible at every Node
    :ivar int size: number of nodes
    :ivar float interactivity: number of user inputs possible at every Node
    :ivar list nodes: all Nodes
    """

    def __init__(self, size=10, interactivity=2):
        self.size = size
        self.interactivity = interactivity
        self.nodes = []
        self.make_nodes()

    def make_nodes(self):
        """Make list of self.size Nodes
        """
        self.nodes = [Node(name='Node' + str(n + 1)) for n in range(self.size)]

    def make_all_links(self):
        """Make 1-utility Links for all pairs of different nodes in self.nodes

        Links are defined for a specific combination of source, destination, and
        response, where the response is the user input at the source Node.
        """
        for source, destination in product(self.nodes, self.nodes):
            for response in range(self.interactivity):
                link = Link(source, destination, response, utility=1)
                try:
                    source.links[destination][response] = link
                except KeyError:
                    source.links[destination] = {response: link}

    def make_random_links(self, density=0.1, skew_power=1):
        """Make random-utility Links for ``density`` fraction of self.nodes

        Links are defined for a specific combination of source, destination, and
        response, where the response is the user input at the source Node.

        :param float density: fraction of defined Links over all Links
        :param float skew_power: Link utility distribution power (u~x^SP on (0, 1))
        """
        for source, destination in product(self.nodes, self.nodes):
            for response in range(self.interactivity):
                # Skip some pairs, according to ``density``
                if random.random() > density:
                    continue
                # Draw utility from x^skew_power on (0, 1)
                utility = random.random() ** skew_power
                link = Link(source, destination, response, utility)
                try:
                    source.links[destination][response] = link
                except KeyError:
                    source.links[destination] = {response: link}

    def get_all_preferences(self):
        """Return a tuple of all possible tuples of input preferences.

        :return: tuple of all input preferences
        :rtype: tuple
        """
        return tuple(sequence for sequence in product(range(self.interactivity), repeat=self.size))

    def get_random_preferences(self):
        """Return a random tuple of user input preferences, one for each node

        :return: random tuple of user input preferences
        :rtype: tuple
        """
        return tuple(random.choice(range(self.interactivity)) for _ in self.nodes)

    def sequence_probabilities(self, preferences, sequence_length, probability_conversion):
        """Return a list of probabilities for every possible sequence

        :return: list of probabilities for every possible sequence
        :rtype: list
        """
        sequences = list(product(self.nodes, repeat=sequence_length))
        sequence_probabilities = [1.0 / len(self.nodes)] * len(sequences)
        progress_bar_prefix = 'Calculating every sequence probability'
        progress_bar_progress = 0
        progress_bar_size = 76 - len(progress_bar_prefix)
        if DEBUG:
            print(progress_bar_prefix + ' |' + ' ' * progress_bar_size + '|', end='\r')
        for sequence_index in range(len(sequences)):
            if sequence_index / len(sequences) > progress_bar_progress / progress_bar_size:
                progress_bar_progress += 1
                if DEBUG:
                    print(progress_bar_prefix + ' |' +
                          '-' * progress_bar_progress +
                          ' ' * (progress_bar_size - progress_bar_progress) + '|', end='\r')
            for sequence_step in range(sequence_length - 1):
                this_node = sequences[sequence_index][sequence_step]
                next_node = sequences[sequence_index][sequence_step + 1]
                preference = preferences[self.nodes.index(this_node)]
                utilities = []
                for other_node in self.nodes:
                    try:
                        utilities.append(this_node.links[other_node][preference].utility)
                    except KeyError:
                        utilities.append(0)
                probabilities = probability_conversion(utilities)
                step_probability = probabilities[self.nodes.index(next_node)]
                sequence_probabilities[sequence_index] *= step_probability
        if DEBUG:
            print('\r' + progress_bar_prefix + ' |' + '-' * progress_bar_size + '|')
        assert sum(sequence_probabilities) < 1.0001
        assert sum(sequence_probabilities) > 0.9999
        return sequence_probabilities


class Link:
    """Link in a Network, connecting two source and destination Nodes

    :param source: source Node
    :param destination: destination Node
    :param int preference: user input at source Node
    :param float utility: utility of use
    :ivar source: source Node
    :ivar destination: destination Node
    :ivar int preference: user input at source Node
    :ivar float utility: utility of use
    """

    def __init__(self, source, destination, preference, utility):
        self.source = source
        self.destination = destination
        self.preference = preference
        self.utility = utility

    def __repr__(self):
        return ('Link ' + str(self.source) + ' to ' + str(self.destination) +
                ' \t if %d \t utility %.3g' % (self.preference, self.utility))

    def __str__(self):
        return ('Link ' + str(self.source) + ' to ' + str(self.destination) +
                ' \t if %d \t utility %.3g' % (self.preference, self.utility))


class Node:
    """Node in a Network, connected to other nodes by weighted Links

    :param str name: node ID or descriptor
    :ivar str name: node ID or descriptor
    :ivar dict links: neighbor Nodes keying to Links
    """

    def __init__(self, name):
        self.name = name
        self.links = {}

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name    
