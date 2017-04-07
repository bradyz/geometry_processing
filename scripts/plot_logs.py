"""
Sample usage:

python3 scripts/plot_logs.py --log_file=logs/saliency.log --labels=acc,val_acc,loss,val_loss
"""
import argparse
import csv

from matplotlib import pyplot as plt


# Command line arguments.
parser = argparse.ArgumentParser(description='Train a top K SVM.')

parser.add_argument('--log_file', required=True, type=str,
        help='Path to log file.')
parser.add_argument('--labels', required=False, type=str, default='',
        help='Comma delimited column labels. Default is all')

args = parser.parse_args()
log_file = args.log_file
labels = args.labels


def show_graph(to_plot):
    plt.ylim([0.0, 1.0])

    for label, graph in to_plot.items():
        plt.plot(list(range(len(graph))), graph, "o-", label=label)

    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    to_plot = dict()

    if labels:
        to_plot = {label: list() for label in labels.split(',')}

    with open(log_file) as fd:
        for i, row in enumerate(csv.DictReader(fd)):
            if i == 0:
                if not labels:
                    to_plot = {label: list() for label in row}
                continue

            if labels:
                for label in labels.split(','):
                    to_plot[label].append(row[label])
            else:
                for label, value in row.items():
                    to_plot[label].append(value)

    show_graph(to_plot)
