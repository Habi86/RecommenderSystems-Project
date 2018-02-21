import numpy as np
import matplotlib.pyplot as plt

def plot_precision_recall():
    precision_pb = np.loadtxt('./plots/data/PB_precision.txt', delimiter=',')
    recall_pb = np.loadtxt('./plots/data/PB_recall.txt', delimiter=',')

    precision_cf = np.loadtxt('./plots/data/CF_precision.txt', delimiter=',')
    recall_cf = np.loadtxt('./plots/data/CF_recall.txt', delimiter=',')

    # precision_cf_pb = np.loadtxt('./plots/data/CF_PB_precision.txt', delimiter=',')
    # recall_cf_pb = np.loadtxt('./plots/data/CF_PB_recall.txt', delimiter=',')

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
    #precision_recall_plot.plot(recall_cf_pb, precision_cf_pb)
    precision_recall_plot.plot(recall_rb_u, precision_rb_u)
    precision_recall_plot.plot(recall_rb_a, precision_rb_a)
    precision_recall_plot.plot(recall_cb, precision_cb)
    precision_recall_plot.plot(recall_cf_cb, precision_cf_cb)

    #plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB', 'CF_CB'], loc='upper right')
    plt.legend(['PB', 'CF', 'RB_U', 'RB_A', 'CB', 'CF_CB'], loc='upper right')
    precision_recall_plot.set_xlabel('Recall')
    precision_recall_plot.set_ylabel('Precision')
    plt1.savefig('./plots/precision-recall.png')

def plot_f1():
  number_recommended_artists = range(10, 200, 10)
  f1_pb = np.loadtxt('./plots/data/PB_f1.txt', delimiter=',')
  f1_cf = np.loadtxt('./plots/data/CF_f1.txt', delimiter=',')
  #f1_cf_pb = np.loadtxt('./plots/data/CF_PB_f1.txt', delimiter=',')
  f1_rb_u = np.loadtxt('./plots/data/RB_U_f1.txt', delimiter=',')
  f1_rb_a = np.loadtxt('./plots/data/RB_A_f1.txt', delimiter=',')
  f1_cb = np.loadtxt('./plots/data/CB_f1.txt', delimiter=',')
  f1_cf_cb = np.loadtxt('./plots/data/CF_CB_f1.txt', delimiter=',')

  plt1 = plt.figure()
  f1_plot = plt1.add_subplot(211)
  f1_plot.plot(number_recommended_artists, f1_pb)
  f1_plot.plot(number_recommended_artists, f1_cf)
  #f1_plot.plot(number_recommended_artists, f1_cf_pb)
  f1_plot.plot(number_recommended_artists, f1_rb_u)
  f1_plot.plot(number_recommended_artists, f1_rb_a)
  f1_plot.plot(number_recommended_artists, f1_cb)
  f1_plot.plot(number_recommended_artists, f1_cf_cb)

  f1_plot.set_xlabel('number of recommended items')
  f1_plot.set_ylabel('F1')
  #plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB', 'CF_CB'], loc='upper right')
  plt.legend(['PB', 'CF', 'RB_U', 'RB_A', 'CB', 'CF_CB'], loc='upper right')
  plt1.savefig('./plots/f1.png')


# plot_precision_recall()
# plot_f1()



def plot_cold_start_f1():
  f1_pb = np.loadtxt('./plots/data/cold-start/10000/PB_f1.txt', delimiter=',')
  f1_cf_pb = np.loadtxt('./plots/data/cold-start/10000/CF_PB_f1.txt', delimiter=',')
  f1_cf = np.loadtxt('./plots/data/cold-start/10000/CF_f1.txt', delimiter=',')
  f1_cf_cb = np.loadtxt('./plots/data/cold-start/10000/CF_CB_f1.txt', delimiter=',')
  f1_cb = np.loadtxt('./plots/data/cold-start/10000/CB_f1.txt', delimiter=',')
  f1_rb_a = np.loadtxt('./plots/data/cold-start/10000/RB_A_f1.txt', delimiter=',')
  f1_rb_u = np.loadtxt('./plots/data/cold-start/10000/RB_U_f1.txt', delimiter=',')
  user_playcounts = np.loadtxt('./plots/data/cold-start/10000/user_playcounts_02.txt', delimiter=',')

  plt1 = plt.figure()
  f1_plot = plt1.add_subplot(211)
  f1_plot.plot(user_playcounts, f1_pb, 'o')
  f1_plot.plot(user_playcounts, f1_cf, 'o')
  f1_plot.plot(user_playcounts, f1_cf_pb, 'o')
  f1_plot.plot(user_playcounts, f1_rb_u, 'o')
  f1_plot.plot(user_playcounts, f1_rb_a, 'o')
  f1_plot.plot(user_playcounts, f1_cb, 'o')
  f1_plot.plot(user_playcounts, f1_cf_cb, 'o')
  
  
  

  f1_plot.set_xlabel('amount of playcounts')
  f1_plot.set_ylabel('F1')
  plt.legend(['PB', 'CF', 'CF_PB', 'RB_U', 'RB_A', 'CB', 'CF_CB'], loc='upper right')
  plt1.savefig('./plots/data/cold-start/10000/f1.png')


plot_cold_start_f1()