import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


def balanced_results(results: list[int], interval=10) -> list[int]:
    balanced = []
    for i in range(len(results) - interval):
        balanced.append(sum(results[i : i + interval]) / interval)
    return balanced


def plot_result(results: list[int], interval=10):
    style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    pl_result = balanced_results(results, interval=interval)
    ax.plot(pl_result)
    plt.legend()
    plt.show()


def plot_evals(evals: list[tuple[int, int, int]]):
    style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    evals.sort()
    left, singles_left, reward = list(zip(*evals))
    ax.plot(left, label="cells left")
    ax.plot(singles_left, label="isolated cells left")
    ax.plot(reward, label="total reward gained")
    plt.legend()
    plt.show()
