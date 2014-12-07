from adversaries import Adversary
from curators import Curator
from databases import Network
from matplotlib import pyplot
import numpy as np


DEBUG = False


def kl_divergence(distribution, approx_distribution):
    """Calculate the KL divergence for two distributions

    :param distribution: original proabability distribution
    :param approx_distribution: approximate of the original distribution
    """
    return tuple(np.sum(distribution) * np.log(np.array(distribution) / np.array(approx_distribution)))


def test_adversary():
    # Network parameters
    size = 10
    interactivity = 2
    sequence_length = 4

    # Testing parameters
    cutoff_fractions = (0.1, 0.5)
    preference_count = 5
    query_counts = tuple(np.logspace(0, 3.5, num=20))

    curator = Curator()
    curator.network = Network(size, interactivity)
    curator.network.make_random_links()
    adversary = Adversary(curator)
    preferences = tuple(curator.network.get_random_preferences() for _ in range(preference_count))

    errors = {cutoff_fraction: {preference: [] for preference in preferences}
              for cutoff_fraction in cutoff_fractions}
    kl_divergences = {preference: [] for preference in preferences}
    for query_count in query_counts:
        if DEBUG:
            print()
        adversary.pirate(sequence_length, number_of_queries=int(query_count))
        for preference in preferences:
            if DEBUG:
                print('Preference %d' % (preferences.index(preference) + 1))
            adversary_probabilities = adversary.network.sequence_probabilities(preference,
                                                                               sequence_length,
                                                                               Adversary.normalize)
            curator_probabilities = curator.network.sequence_probabilities(preference,
                                                                           sequence_length,
                                                                           curator.exponential_mechanism)
            for cutoff_fraction in cutoff_fractions:
                cutoff_number = int(cutoff_fraction * len(curator_probabilities))
                error_count = (sum(adversary_probabilities[i] not in sorted(adversary_probabilities)[:cutoff_number]
                               for i in range(len(curator_probabilities[:cutoff_number]))))
                errors[cutoff_fraction][preference].append(1.0 * error_count / cutoff_number)

            # KL Divergence after a preference query
            kl_divergences[preference].append(kl_divergence(curator_probabilities, adversary_probabilities))

    figure, (axes_error, axes_kl) = pyplot.subplots(1, 2)
    figure.canvas.set_window_title('Top Sequences Error and KL Divergence vs. Number of Queries')
    axes_error.set_title('Top Sequences Error vs. Number of Queries')
    axes_error.set_xlabel('Number of Queries')
    axes_error.set_xscale('log')
    axes_error.set_ylabel('Top Sequences Error')
    axes_kl.set_title('KL Divergence vs. Number of Queries')
    axes_kl.set_xlabel('Number of Queries')
    axes_kl.set_xscale('log')
    axes_kl.set_ylabel('KL Divergence')
    for cutoff_fraction in cutoff_fractions:
        axes_error.plot([sum(query_counts[:i]) for i in range(1, len(query_counts) + 1)],
                        [np.mean([errors[cutoff_fraction][preference][i] for preference in preferences])
                         for i in range(len(query_counts))],
                        label='Top %.2g%%' % (100 * cutoff_fraction))
    axes_error.legend(loc='best')
    axes_kl.plot([sum(query_counts[:i]) for i in range(1, len(query_counts) + 1)],
                 [np.mean([kl_divergences[preference][i] for preference in preferences])
                  for i in range(len(query_counts))])
    pyplot.show()


def test_adversary_no_kl():
    # Network parameters
    size = 10
    interactivity = 2
    sequence_length = 4

    # Testing parameters
    cutoff_fractions = (0.1, 0.5)
    preference_count = 1
    query_counts = tuple(np.logspace(0, 6, num=30))

    curator = Curator()
    curator.network = Network(size, interactivity)
    curator.network.make_random_links()
    adversary = Adversary(curator)
    preferences = tuple(curator.network.get_random_preferences() for _ in range(preference_count))

    errors = {cutoff_fraction: {preference: [] for preference in preferences}
              for cutoff_fraction in cutoff_fractions}
    for query_count in query_counts:
        if DEBUG:
            print()
        adversary.pirate(sequence_length, number_of_queries=int(query_count))
        for preference in preferences:
            if DEBUG:
                print('Preference %d' % (preferences.index(preference) + 1))
            adversary_probabilities = adversary.network.sequence_probabilities(preference,
                                                                               sequence_length,
                                                                               Adversary.normalize)
            curator_probabilities = curator.network.sequence_probabilities(preference,
                                                                           sequence_length,
                                                                           curator.exponential_mechanism)
            for cutoff_fraction in cutoff_fractions:
                cutoff_number = int(cutoff_fraction * len(curator_probabilities))
                sorted_adversary_probabilities = sorted(adversary_probabilities)
                error_count = (sum(adversary_probabilities[i] not in sorted_adversary_probabilities[:cutoff_number]
                               for i in range(len(curator_probabilities[:cutoff_number]))))
                errors[cutoff_fraction][preference].append(1.0 * error_count / cutoff_number)

    figure, axes_error = pyplot.subplots()
    figure.canvas.set_window_title('Top Sequences Error and KL Divergence vs. Number of Queries')
    axes_error.set_title('Top Sequences Error vs. Number of Queries')
    axes_error.set_xlabel('Number of Queries')
    axes_error.set_xscale('log')
    axes_error.set_ylabel('Top Sequences Error')
    for cutoff_fraction in cutoff_fractions:
        axes_error.plot([sum(query_counts[:i]) for i in range(1, len(query_counts) + 1)],
                        [np.mean([errors[cutoff_fraction][preference][i] for preference in preferences])
                         for i in range(len(query_counts))],
                        label='Top %.2g%%' % (100 * cutoff_fraction))
    axes_error.legend(loc='best')
    pyplot.show()


def test_etas():
    # Network parameters
    size = 10
    interactivity = 2
    sequence_length = 4

    # Testing parameters
    cutoff_fractions = (0.1, 0.5)
    etas = (1e-2, 1e-4, 1e-6)
    preference_count = 1
    query_counts = tuple(np.logspace(0, 2.5, num=20))

    curator = Curator()
    curator.network = Network(size, interactivity)
    curator.network.make_random_links()
    adversaries = [Adversary(curator, eta=eta) for eta in etas]
    preferences = tuple(curator.network.get_random_preferences() for _ in range(preference_count))

    errors = {adversary: {cutoff_fraction: {preference: [] for preference in preferences}
                          for cutoff_fraction in cutoff_fractions}
              for adversary in adversaries}
    kl_divergences = {adversary: {preference: [] for preference in preferences}
                      for adversary in adversaries}
    for query_count in query_counts:
        for adversary in adversaries:
            if DEBUG:
                print()
            adversary.pirate(sequence_length, number_of_queries=int(query_count))
            for preference in preferences:
                if DEBUG:
                    print('Preference %d' % (preferences.index(preference) + 1))
                adversary_probabilities = adversary.network.sequence_probabilities(preference,
                                                                                   sequence_length,
                                                                                   Adversary.normalize)
                curator_probabilities = curator.network.sequence_probabilities(preference,
                                                                               sequence_length,
                                                                               curator.exponential_mechanism)
                for cutoff_fraction in cutoff_fractions:
                    cutoff_number = int(cutoff_fraction * len(curator_probabilities))
                    error_count = (sum(adversary_probabilities[i] not in sorted(adversary_probabilities)[:cutoff_number]
                                   for i in range(len(curator_probabilities[:cutoff_number]))))
                    errors[adversary][cutoff_fraction][preference].append(1.0 * error_count / cutoff_number)

                # KL Divergence after a preference query
                kl_divergences[adversary][preference].append(kl_divergence(curator_probabilities, adversary_probabilities))

    figure, (axes_error, axes_kl) = pyplot.subplots(1, 2)
    figure.canvas.set_window_title('Top Sequences Error and KL Divergence vs. Number of Queries')
    axes_error.set_title('Top Sequences Error vs. Number of Queries')
    axes_error.set_xlabel('Number of Queries')
    axes_error.set_xscale('log')
    axes_error.set_ylabel('Top Sequences Error')
    axes_kl.set_title('KL Divergence vs. Number of Queries')
    axes_kl.set_xlabel('Number of Queries')
    axes_kl.set_xscale('log')
    axes_kl.set_ylabel('KL Divergence')
    for adversary in adversaries:
        for cutoff_fraction in cutoff_fractions:
            axes_error.plot([sum(query_counts[:i]) for i in range(1, len(query_counts) + 1)],
                            [np.mean([errors[adversary][cutoff_fraction][preference][i] for preference in preferences])
                             for i in range(len(query_counts))],
                            lw=3, label='Top %.2g%% (Eta = %.3g)' % (100 * cutoff_fraction, adversary.eta))
    axes_error.legend(loc='best')
    for adversary in adversaries:
        axes_kl.plot([sum(query_counts[:i]) for i in range(1, len(query_counts) + 1)],
                     [np.mean([kl_divergences[adversary][preference][i] for preference in preferences])
                      for i in range(len(query_counts))],
                     lw=3, label='Eta = %.3g' % adversary.eta)
    pyplot.show()


def test_network():
    epsilons = [100]
    fractions_of_links_defined = [0.2, 0.4, 0.6, 0.8]
    numbers_of_nodes = [10]
    numbers_of_responses = [2]
    sequence_lengths = [4]
    utility_skew_powers = [1]
    variables = fractions_of_links_defined
    variable_name = 'Fraction of Links Defined'
    number_of_repeat_runs = 5
    if DEBUG:
        print('Testing model by varying ' + variable_name)

    figure, (link_axes, sequence_axes) = pyplot.subplots(2, 1)
    figure.canvas.set_window_title('Content Network with Varying ' + variable_name)
    figure.suptitle(('%.2g Epsilon, %.2g Fraction of Links Defined, %.2g Nodes, ' +
                     '%.2g Responses, %.2g Sequence Length, %.2g Utility Skew Power') %
                    (epsilons[0], fractions_of_links_defined[0], numbers_of_nodes[0],
                     numbers_of_responses[0], sequence_lengths[0], utility_skew_powers[0]),
                    size='large')
    link_axes.set_title('Link Utilities')
    link_axes.set_xlabel('Link Index (Sorted)')
    link_axes.set_ylabel('Link Utility')
    link_colors = [next(link_axes._get_lines.color_cycle) for _ in range(len(variables))]
    sequence_axes.set_title('Sequence Probabilities')
    sequence_axes.set_xlabel('Sequence Index (Sorted)')
    sequence_axes.set_xscale('log')
    sequence_axes.set_ylabel('Sequence Probability')
    sequence_axes.set_yscale('log')
    sequence_colors = [next(sequence_axes._get_lines.color_cycle) for _ in range(len(variables))]

    for run_number in range(len(variables)):
        variable = variables[run_number]
        epsilon = variable if epsilons == variables else epsilons[0]
        fraction_of_links_defined = variable if fractions_of_links_defined == variables else fractions_of_links_defined[0]
        number_of_nodes = variable if numbers_of_nodes == variables else numbers_of_nodes[0]
        number_of_responses = variable if numbers_of_responses == variables else numbers_of_responses[0]
        sequence_length = variable if sequence_lengths == variables else sequence_lengths[0]
        utility_skew_power = variable if utility_skew_powers == variables else utility_skew_powers[0]

        for repeat_run_number in range(number_of_repeat_runs):
            curator = Curator()
            curator.network = Network(size=number_of_nodes, interactivity=number_of_responses)
            curator.network.make_random_links(density=fraction_of_links_defined, skew_power=utility_skew_power)
            preferences = curator.network.get_random_preferences()
            if DEBUG:
                print('%d Nodes, %d Responses, Sequences of %d, Density=%.3g, Epsilon=%.3g, Skew=%.3g, Run %d' %
                      (number_of_nodes, number_of_responses, sequence_length,
                       fraction_of_links_defined, epsilon, utility_skew_power, repeat_run_number))
            sequence_probabilities = sorted([probability for probability in
                                             curator.network.sequence_probabilities(preferences,
                                                                                    sequence_length,
                                                                                    curator.exponential_mechanism)
                                             if probability > 10 ** (- sequence_length)], reverse=True)
            link_utilities = sorted([link.utility
                                     for node in curator.network.nodes
                                     for links in node.links.values()
                                     for link in links.values()], reverse=True)
            assert all(probability for probability in sequence_probabilities)

            label = str(variable) + ' ' + variable_name
            link_color = link_colors[run_number]
            sequence_color = sequence_colors[run_number]
            link_graph, = link_axes.plot(range(len(link_utilities)), link_utilities, c=link_color, lw=3)
            sequence_graph, = sequence_axes.plot(range(len(sequence_probabilities)), sequence_probabilities, c=sequence_color, lw=3)
            if not repeat_run_number:
                link_graph.set_label(label)
                sequence_graph.set_label(label)

    link_axes.legend()
    sequence_axes.legend()
    pyplot.show()


if __name__ == '__main__':
    test_network()
