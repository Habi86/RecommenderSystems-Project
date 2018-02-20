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

    precision_cf_cb = np.loadtxt('./plots/data/CF_CB_precision.txt', delimiter=',')
    recall_cf_cb = np.loadtxt('./plots/data/CF_CB_recall.txt', delimiter=',')


    plt1 = plt.figure()
    precision_recall_plot = plt1.add_subplot(211)
    precision_recall_plot.plot(recall_pb, precision_pb)
    precision_recall_plot.plot(recall_cf, precision_cf)
    precision_recall_plot.plot(recall_cf_pb, precision_cf_pb)
    precision_recall_plot.plot(recall_rb_u, precision_rb_u)
    precision_recall_plot.plot(recall_rb_a, precision_rb_a)
    precision_recall_plot.plot(recall_cb, precision_cb)
    precision_recall_plot.plot(recall_cf_cb, precision_cf_cb)

    plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB', 'CF_CB'], prop={'size': 8}, loc='upper right')
    precision_recall_plot.set_xlabel('Recall')
    precision_recall_plot.set_ylabel('Precision')
    plt1.savefig('./plots/precision-recall_3 .png')

def plot_f1():
  number_recommended_artists = range(10, 200, 10)
  f1_pb = np.loadtxt('./plots/data/PB_f1.txt', delimiter=',')
  f1_cf = np.loadtxt('./plots/data/CF_f1.txt', delimiter=',')
  f1_cf_pb = np.loadtxt('./plots/data/CF_PB_f1.txt', delimiter=',')
  f1_rb_u = np.loadtxt('./plots/data/RB_U_f1.txt', delimiter=',')
  f1_rb_a = np.loadtxt('./plots/data/RB_A_f1.txt', delimiter=',')
  f1_cb = np.loadtxt('./plots/data/CB_f1.txt', delimiter=',')
  f1_cf_cb = np.loadtxt('./plots/data/CF_CB_f1.txt', delimiter=',')

  plt1 = plt.figure()
  f1_plot = plt1.add_subplot(211)
  f1_plot.plot(number_recommended_artists, f1_pb)
  f1_plot.plot(number_recommended_artists, f1_cf)
  f1_plot.plot(number_recommended_artists, f1_cf_pb)
  f1_plot.plot(number_recommended_artists, f1_rb_u)
  f1_plot.plot(number_recommended_artists, f1_rb_a)
  f1_plot.plot(number_recommended_artists, f1_cb)
  f1_plot.plot(number_recommended_artists, f1_cf_cb)

  f1_plot.set_xlabel('number of recommended items')
  f1_plot.set_ylabel('F1')
  plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB', 'CF_CB'], prop={'size': 8}, loc='upper right')
  plt1.savefig('./plots/f1_3.png')


plot_precision_recall()
plot_f1()

# [210, 522, 235, 207, 475, 76, 650, 362, 227, 582, 396, 1052, 492, 1032, 751]

# 2
# [847, 912, 411, 1042, 896, 1056, 478, 395, 659, 642, 127, 1093, 931, 643, 983]

# 3
# [169, 611, 108, 388, 636, 889, 149, 721, 382, 237, 115, 581, 75, 1019, 391]
