import matplotlib.pyplot as plt

def plot_distribution(distributions, labels):
    fig, axs = plt.subplots(1, len(distributions))
    fig.set_size_inches(15, 5)
    for distribution, label, axis in zip(distributions, labels, axs):
        axis.bar(distribution.keys(), distribution.values())
        axis.set_title(label)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()