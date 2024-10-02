import numpy as np
from scipy.stats import ttest_rel, ttest_ind


def read_data_from_file(file_path, single_file=True, delimiter='\t'):
    '''Read file'''
    good = []
    bad = []

    if single_file:
        with open(file_path, 'r') as file:
            for line in file:
                values = line.split(delimiter)
                good.append(float(values[0]))
                bad.append(float(values[1]))
    else:
        with open(file_path[0], 'r') as file:
            for line in file:
                values = line.split(delimiter)
                good.append(float(values[0]))

        with open(file_path[1], 'r') as file:
            for line in file:
                values = line.split(delimiter)
                bad.append(float(values[0]))

    return np.array(good), np.array(bad)


def calculate_p_value(good, bad, paired=True):
    if paired:
        t_stat, p_value = ttest_rel(good, bad)
    else:
        t_stat, p_value = ttest_ind(good, bad)

    return t_stat, p_value


def main(single_file=True, file_path=None, delimiter='\t'):
    if single_file:
        good, bad = read_data_from_file(file_path, single_file=True, delimiter=delimiter)
        t_stat, p_value = calculate_p_value(good, bad, paired=True)
    else:
        good, bad = read_data_from_file(file_path, single_file=False)
        t_stat, p_value = calculate_p_value(good, bad, paired=False)

    print(f'T-statistic: {t_stat}, P-value: {p_value}')

# Compare the good and bad scores in one file
main(single_file=True, file_path='Yi_complement_verb.txt')

# Compare the good scores in two files
# main(single_file=False, file_path=['CPM_complement_verb_1.txt', 'Yi_complement_verb.txt'])
