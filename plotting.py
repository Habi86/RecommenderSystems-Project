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
    precision_recall_plot.set_xlabel('Recall')
    precision_recall_plot.set_ylabel('Precision')
    plt1.savefig('./plots/precision-recall_mit_cb.png')

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

  f1_plot.set_xlabel('number of recommended items')
  f1_plot.set_ylabel('F1')
  plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB'], loc='upper right')
  plt1.savefig('./plots/f1_mit_cb.png')


# plot_precision_recall()
# plot_f1()



def plot_cold_start_f1():
  f1_pb = np.loadtxt('./plots/data/cold-start/PB_f1.txt', delimiter=',')
  pb_user_playcounts = np.loadtxt('./plots/data/cold-start/PB_user_playcounts.txt', delimiter=',')
  f1_cf = np.loadtxt('./plots/data/cold-start/CF_f1.txt', delimiter=',')
  cf_user_playcounts = np.loadtxt('./plots/data/cold-start/CF_user_playcounts.txt', delimiter=',')
  f1_rb_a = np.loadtxt('./plots/data/cold-start/RB_A_f1.txt', delimiter=',')
  rb_a_user_playcounts = np.loadtxt('./plots/data/cold-start/RB_A_user_playcounts.txt', delimiter=',')

  # f1_cf_pb = np.loadtxt('./plots/data/CF_PB_f1.txt', delimiter=',')
  # f1_rb_u = np.loadtxt('./plots/data/RB_U_f1.txt', delimiter=',')
  
  # f1_cb = np.loadtxt('./plots/data/CB_f1.txt', delimiter=',')

  plt1 = plt.figure()
  f1_plot = plt1.add_subplot(211)
  f1_plot.plot(pb_user_playcounts, f1_pb, 'o')
  f1_plot.plot(cf_user_playcounts, f1_cf, 'o')
  f1_plot.plot(rb_a_user_playcounts, f1_rb_a, 'o')
  # f1_plot.plot(number_recommended_artists, f1_cf_pb)
  # f1_plot.plot(number_recommended_artists, f1_rb_u)
  # f1_plot.plot(number_recommended_artists, f1_rb_a)
  # f1_plot.plot(number_recommended_artists, f1_cb)

  f1_plot.set_xlabel('amount of playcounts')
  f1_plot.set_ylabel('F1')
  # plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB'], loc='upper right')
  plt1.savefig('./plots/cold-start-f1-pb.png')


plot_cold_start_f1()