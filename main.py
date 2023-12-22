import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use('ggplot')


def readfile() -> tuple:
    selection1 = list()
    selection2 = list()
    selection3 = list()
    with open("var14.txt") as file:
        while True:
            line = file.readline().split(",")
            if line[0] == "":
                break
            counter = int(line[0])
            x = float(line[1].rstrip())
            if 1 <= counter <= 10:
                selection1.append(x)
                selection2.append(x)
                selection3.append(x)
            elif 11 <= counter <= 50:
                selection2.append(x)
                selection3.append(x)
            else:
                selection3.append(x)
    return sorted(selection1), sorted(selection2), sorted(selection3)


def average_selective(selections: tuple) -> tuple:
    sum_selection1 = 0
    sum_selection2 = 0
    sum_selection3 = 0
    for counter, x in enumerate(selections[2], start=1):
        if 1 <= counter <= 10:
            sum_selection1 += x
            sum_selection2 += x
            sum_selection3 += x
        elif 11 <= counter <= 50:
            sum_selection2 += x
            sum_selection3 += x
        else:
            sum_selection3 += x
    return sum_selection1 / len(selections[0]), sum_selection2 / len(selections[1]), sum_selection3 / len(selections[2])


def sample_variance(selections: tuple, mediums: tuple, unbiased_variance=False) -> tuple:
    summa1 = 0
    summa2 = 0
    summa3 = 0
    for counter, x in enumerate(selections[2], start=1):
        if 1 <= counter <= 10:
            summa1 += pow(x-mediums[0], 2)
            summa2 += pow(x-mediums[1], 2)
            summa3 += pow(x-mediums[2], 2)
        elif 11 <= counter <= 50:
            summa2 += pow(x - mediums[1], 2)
            summa3 += pow(x - mediums[2], 2)
        else:
            summa3 += pow(x - mediums[2], 2)
    if unbiased_variance:
        return summa1 / (len(selections[0]) - 1), summa2 / (len(selections[1]) - 1), summa3 / (len(selections[2]) - 1)
    else:
        return summa1 / len(selections[0]), summa2 / len(selections[1]), summa3 / len(selections[2])


def evaluation_parameters_a(mediums: tuple):
    return mediums[0], mediums[1], mediums[2]


def evaluation_parameters_v2(variance_unbiased: tuple):
    return variance_unbiased[0], variance_unbiased[1], variance_unbiased[2]


# def normal_dist(x, mean, sd):
#     prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
#     return prob_density


def expansion_selection(selection: list):
    var = np.var(selection)
    min_ = min(selection)
    max_ = max(selection)
    values_x = np.linspace(min_, max_, 10000)
    return values_x


def empirical_and_normal(selections: tuple):
    fig, axs = plt.subplots(nrows=3, ncols=1,)
    # plt.xlim([0,2])
    # axs[0].set_xlim([-0.126473389253296, 1.89113824371571])
    # axs[1].set_xlim([-0.126473389253296, 2.40090611973363])
    # axs[2].set_xlim([-0.33990738543508, 2.40090611973363])

    value_x0 = expansion_selection(selections[0])
    value_x1 = expansion_selection(selections[1])
    value_x2 = expansion_selection(selections[2])


    axs[0].plot(value_x0, parameters_normal_function(value_x0), color='green')
    axs[0].step(parameters_empirical_function(selections[0])[0], parameters_empirical_function(selections[0])[1])
    axs[0].plot([min(selections[0]) - 2 * np.var(selections[0]), min(selections[0])], [0, 0], color='red')
    axs[0].plot([max(selections[0]), max(selections[0]) + 2 * np.var(selections[0])], [1, 1], color='red')

    axs[1].plot(value_x1, parameters_normal_function(value_x1), color='black')
    axs[1].step(parameters_empirical_function(selections[1])[0], parameters_empirical_function(selections[1])[1])
    axs[1].plot([min(selections[1]) - 2 * np.var(selections[1]), min(selections[1])], [0, 0], color='red')
    axs[1].plot([max(selections[1]), max(selections[1]) + 2 * np.var(selections[1])], [1, 1], color='red')

    axs[2].plot(value_x2, parameters_normal_function(value_x2), color='blue')
    axs[2].step(parameters_empirical_function(selections[2])[0], parameters_empirical_function(selections[2])[1])
    axs[2].plot([min(selections[2]) - 2 * np.var(selections[2]), min(selections[2])], [0, 0], color='red')
    axs[2].plot([max(selections[2]), max(selections[2]) + 2 * np.var(selections[2])], [1, 1], color='red')

    plt.show()


def parameters_densities_normal_function(selection) -> tuple:
    mean = np.mean(selection)
    sd = np.std(selection)
    pdf = sc.norm.pdf(selection, mean, sd)
    return pdf


def parameters_empirical_function(selection):
    ecdf = sm.distributions.ECDF(selection)
    x = np.linspace(min(selection), max(selection))
    y = ecdf(x)
    return x, y


def parameters_normal_function(selection):
    mean = np.mean(selection)
    sd = np.std(selection)
    cdf = sc.norm.cdf(selection, mean, sd)
    return cdf


def hist_relative_frequencies_and_normal(selections: tuple) -> None:
    fig, axs = plt.subplots(nrows=3, ncols=1)

    value_x0 = expansion_selection(selections[0])
    # value_x1 = expansion_selection(selections[1])
    # value_x2 = expansion_selection(selections[2])

    axs[0].hist(selections[0], density=True)
    axs[0].plot(value_x0, parameters_densities_normal_function(value_x0))

    axs[1].hist(selections[1], density=True)
    axs[1].plot(selections[1], parameters_densities_normal_function(selections[1]))

    axs[2].hist(selections[2], density=True)
    axs[2].plot(selections[2], parameters_densities_normal_function(selections[2]))

    plt.show()


def confidence_interval_a(selections: tuple, level_trust=0.95):
    return sc.t.interval(level_trust, df=len(selections[0]) - 1, loc=np.mean(selections[0]), scale=sc.sem(selections[0])), \
        sc.t.interval(level_trust, df=len(selections[1]) - 1, loc=np.mean(selections[1]), scale=sc.sem(selections[1])),\
        sc.t.interval(level_trust, df=len(selections[2]) - 1, loc=np.mean(selections[2]), scale=sc.sem(selections[2]))


def confidence_interval_v2(selections: tuple, level_trust=0.95):
    return sc.norm.interval(level_trust, loc=np.std(selections[0]), scale=sc.sem(selections[0])), \
        sc.norm.interval(level_trust, loc=np.std(selections[1]), scale=sc.sem(selections[1])), \
        sc.norm.interval(level_trust, loc=np.std(selections[2]), scale=sc.sem(selections[2]))


def df(selections):
    levels_trust = []
    lens_confidence_interval_v2 = [[], [], []]
    lens_confidence_interval_a = [[], [], []]
    level = 0.90000
    for i in range(1111):
        levels_trust.append(level)
        test = confidence_interval_v2(selections, level)
        test1 = confidence_interval_a(selections, level)
        lens_confidence_interval_v2[0].append(abs(test[0][1] - test[0][0]))
        lens_confidence_interval_v2[1].append(abs(test[1][1] - test[1][0]))
        lens_confidence_interval_v2[2].append(abs(test[2][1] - test[2][0]))

        lens_confidence_interval_a[0].append(abs(test1[0][1] - test1[0][0]))
        lens_confidence_interval_a[1].append(abs(test1[1][1] - test1[1][0]))
        lens_confidence_interval_a[2].append(abs(test1[2][1] - test1[2][0]))
        level += 0.00009

    fig, axs = plt.subplots(nrows=1, ncols=2,)
    axs[0].plot(levels_trust, lens_confidence_interval_v2[0])
    axs[0].plot(levels_trust, lens_confidence_interval_v2[1])
    axs[0].plot(levels_trust, lens_confidence_interval_v2[2])

    axs[1].plot(levels_trust, lens_confidence_interval_a[0])
    axs[1].plot(levels_trust, lens_confidence_interval_a[1])
    axs[1].plot(levels_trust, lens_confidence_interval_a[2])

    plt.show()


if __name__ == "__main__":
    buffer = readfile()

    # empirical_and_normal(buffer)
    hist_relative_frequencies_and_normal(buffer)
    # confidence_interval_a(buffer)
    # confidence_interval_v2(buffer)
    # df(buffer)
    #
    # print(confidence_interval_a(buffer))
    # print(confidence_interval_v2(buffer))
    # print("-------------")
    # print(sample_variance(buffer, average_selective(buffer)))
    # print(sample_variance(buffer, average_selective(buffer), unbiased_variance=True))
    # print("-------------")
    # print(average_selective(buffer))#
    # hist_relative_frequencies_and_normal(buffer, parameters_normal_function(buffer))
    # axs[0].hist(buffer[2], edgecolor='black')
    # empirical_and_normal(buffer, parameters_densities_normal_function(buffer), parameters_empirical_function(buffer))
