%% Initialize
clc; close all; clear;

%% import data
train_acc = importdata('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_48/Sub48_LOSOXV_train_acc_sorted.txt');
test_acc = importdata('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_48/Sub48_LOSOXV_test_acc_sorted.txt');
%test_acc = readNPY('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_48/np_acc.npy');
%test_acc = test_acc * 100;
%random_rbd_idx = readNPY('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_48/random_rbd_idx.npy');
%test_order = readNPY('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_48/test_idx_order.npy');

random_rbd_idx = readNPY('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_nmci_28/re/random_rbd_idx.npy');
test_order = readNPY('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_nmci_28/re/test_idx_order.npy');
test_acc2 = readNPY('D:/ANT_3D_CNN/ANT_topo/LOSOXV_random_selection_sub_nmci_28/re/np_acc.npy');
test_acc2 = test_acc2 * 100;

train_acc_mean = mean(train_acc);
test_acc_mean = mean(test_acc);
train_acc_std = std(train_acc);
test_acc_std = std(test_acc);

% plot configuration
x = 1:1:48;
x1 = 1:1:24;
x2 = 25:1:48;
w1 = 0.5;
w2 = 0.25;
x3 = 1:1:28;
x4 = 1:1:14;

figure(1)
%bar(x1, train_acc(1:24), w1, 'FaceColor', [0.2 0.2 0.5])
%hold on
%bar(x,test_acc, w2, 'FaceColor', [0 0.7 0.7])
bar(x4,test_acc2(1:14), w2, 'y')
%hold off
grid on
ylabel('Accuracy [%]', 'fontsize', 16, 'fontweight', 'bold')
xlabel('Subject number', 'fontsize', 16, 'fontweight', 'bold')
legend({'Train acc','Test acc'},'Location','northwest')
xticks(x4)

figure(2)
%bar(x1, train_acc(25:48), w1, 'FaceColor', [0.2 0.2 0.5])
%hold on
bar(x4,test_acc2(15:28), w2, 'y')
%hold off
grid on
ylabel('Accuracy [%]', 'fontsize', 16, 'fontweight', 'bold')
xlabel('Subject number', 'fontsize', 16, 'fontweight', 'bold')
legend({'Train acc','Test acc'},'Location','northwest')
xticks(x4)

figure(3)
%bar(x, train_acc, w1, 'FaceColor', [0.2 0.2 0.5])
%hold on
bar(x3,test_acc2, w2, 'y')
%hold off
grid on
ylabel('Accuracy [%]', 'fontsize', 16, 'fontweight', 'bold')
xlabel('Subject number', 'fontsize', 16, 'fontweight', 'bold')
legend({'Train acc','Test acc'},'Location','northwest')
xticks(x3)


%%
test_acc1 = [];

figure(1)
bar(x4, test_acc1(1:14), w1, 'FaceColor', [0.2 0.2 0.5])
hold on
%bar(x,test_acc, w2, 'FaceColor', [0 0.7 0.7])
bar(x4,test_acc2(1:14), w2, 'y')
hold off
grid on
ylabel('Accuracy [%]', 'fontsize', 16, 'fontweight', 'bold')
xlabel('Subject number', 'fontsize', 16, 'fontweight', 'bold')
legend({'test1 acc','test2 acc'},'Location','northwest')
xticks(x4)

figure(2)
bar(x4, test_acc1(15:28), w1, 'FaceColor', [0.2 0.2 0.5])
hold on
bar(x4,test_acc2(15:28), w2, 'y')
%hold off
grid on
ylabel('Accuracy [%]', 'fontsize', 16, 'fontweight', 'bold')
xlabel('Subject number', 'fontsize', 16, 'fontweight', 'bold')
legend({'test1 acc','test2 acc'},'Location','northwest')
xticks(x4)

figure(3)
bar(x3, test_acc1, w1, 'FaceColor', [0.2 0.2 0.5])
hold on
bar(x3,test_acc2, w2, 'y')
%hold off
grid on
ylabel('Accuracy [%]', 'fontsize', 16, 'fontweight', 'bold')
xlabel('Subject number', 'fontsize', 16, 'fontweight', 'bold')
legend({'test1 acc','test2 acc'},'Location','northwest')
xticks(x3)