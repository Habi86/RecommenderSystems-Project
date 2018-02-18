import numpy as np
import matplotlib.pyplot as plt

def plot_precision_recall():
    precision_pb = np.loadtxt('./plots/data/PB_precision.txt', delimiter=',')
    recall_pb = np.loadtxt('./plots/data/PB_recall.txt', delimiter=',')

    precision_cf = np.loadtxt('./plots/data/CF_precision.txt', delimiter=',')
    recall_cf = np.loadtxt('./plots/data/CF_recall.txt', delimiter=',')

    precision_cf_pb = np.loadtxt('./plots/data/CF_PB_precision.txt', delimiter=',')
    recall_cf_pb = np.loadtxt('./plots/data/CF_PB_recall.txt', delimiter=',')

    precision_rb_u = np.loadtxt('./plots/data/RB_U_precision.txt', delimiter=',')
    recall_rb_u = np.loadtxt('./plots/data/RB_U_recall.txt', delimiter=',')

    precision_rb_a = np.loadtxt('./plots/data/RB_A_precision.txt', delimiter=',')
    recall_rb_a = np.loadtxt('./plots/data/RB_A_recall.txt', delimiter=',')

    precision_cb = np.loadtxt('./plots/data/CB_precision.txt', delimiter=',')
    recall_cb = np.loadtxt('./plots/data/CB_recall.txt', delimiter=',')


    plt1 = plt.figure()
    precision_recall_plot = plt1.add_subplot(211)
    precision_recall_plot.plot(recall_pb, precision_pb)
    precision_recall_plot.plot(recall_cf, precision_cf)
    precision_recall_plot.plot(recall_cf_pb, precision_cf_pb)
    precision_recall_plot.plot(recall_rb_u, precision_rb_u)
    precision_recall_plot.plot(recall_rb_a, precision_rb_a)
    precision_recall_plot.plot(recall_cb, precision_cb)

    plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB'], loc='upper right')
    plt1.savefig('./plots/CB_precision-recall_2.png')

def plot_f1():
  number_recommended_artists = range(10, 200, 10)
  f1_pb = np.loadtxt('./plots/data/PB_f1.txt', delimiter=',')
  f1_cf = np.loadtxt('./plots/data/CF_f1.txt', delimiter=',')
  f1_cf_pb = np.loadtxt('./plots/data/CF_PB_f1.txt', delimiter=',')
  f1_rb_u = np.loadtxt('./plots/data/RB_U_f1.txt', delimiter=',')
  f1_rb_a = np.loadtxt('./plots/data/RB_A_f1.txt', delimiter=',')
  f1_cb = np.loadtxt('./plots/data/CB_f1.txt', delimiter=',')

  plt1 = plt.figure()
  f1_plot = plt1.add_subplot(211)
  f1_plot.plot(number_recommended_artists, f1_pb)
  f1_plot.plot(number_recommended_artists, f1_cf)
  f1_plot.plot(number_recommended_artists, f1_cf_pb)
  f1_plot.plot(number_recommended_artists, f1_rb_u)
  f1_plot.plot(number_recommended_artists, f1_rb_a)
  f1_plot.plot(number_recommended_artists, f1_cb)

  plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB'], loc='upper right')
  plt1.savefig('./plots/CB_f1_2.png')


plot_precision_recall()
plot_f1()
