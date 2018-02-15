import numpy as np
import matplotlib.pyplot as plt

def plot_precision_recall():
    precision_pb = np.loadtxt('./plots/data/PB_precision.txt', delimiter=',')
    recall_pb = np.loadtxt('./plots/data/PB_recall.txt', delimiter=',')

    precision_cf = np.loadtxt('./plots/data/CF_precision.txt', delimiter=',')
    recall_cf = np.loadtxt('./plots/data/CF_recall.txt', delimiter=',')

    precision_cf_pb = np.loadtxt('./plots/data/CF_PB_precision.txt', delimiter=',')
    recall_cf_pb = np.loadtxt('./plots/data/CF_PB_recall.txt', delimiter=',')


    plt1 = plt.figure()
    precision_recall_plot = plt1.add_subplot(211)
    precision_recall_plot.plot(recall_pb, precision_pb)
    precision_recall_plot.plot(recall_cf, precision_cf)
    precision_recall_plot.plot(recall_cf_pb, precision_cf_pb)

    plt.legend(['PB', 'CF', 'CF_PB'], loc='upper right')
    plt1.savefig('./plots/precision-recall.png')

def plot_f1():
  number_recommended_artists = range(10, 500, 10)
  f1_pb = np.loadtxt('./plots/data/PB_f1.txt', delimiter=',')
  f1_cf = np.loadtxt('./plots/data/CF_f1.txt', delimiter=',')
  f1_cf_pb = np.loadtxt('./plots/data/CF_PB_f1.txt', delimiter=',')

  plt1 = plt.figure()
  f1_plot = plt1.add_subplot(211)
  f1_plot.plot(number_recommended_artists, f1_pb)
  f1_plot.plot(number_recommended_artists, f1_cf)
  f1_plot.plot(number_recommended_artists, f1_cf_pb)

  plt.legend(['PB', 'CF', 'CF_PB'], loc='upper right')
  plt1.savefig('./plots/f1.png')


plot_precision_recall()
plot_f1()
