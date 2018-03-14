import matplotlib.pyplot as plt
import os
import re

#log_path='18.02.22-result_original/log/TRN_jester_RGB_BNInception_TRN_segment3.csv'
log_path='log/TRN_jester_RGB_BNInception_TRNmultiscale_segment3.csv'
#log_path='log/TRN_jester_RGB_BNInception_TRN_segment3.csv'

with open(log_path) as f:
    lines = f.readlines()
'''
Epoch: [4][200/220], lr: 0.00100	Time 0.386 (0.425)	Data 0.000 (0.030)	Loss 0.7331 (0.8155)	Prec@1 68.750 (65.402)	Prec@5 100.000 (100.000)
Test: [20/27]	Time 0.122 (0.382)	Loss 0.4484 (0.5392)	Prec@1 81.250 (77.778)	Prec@5 100.000 (100.000)
Testing Results: Prec@1 77.673 Prec@5 100.000 Loss 0.54130 
'''

train_lines=[]
test_lines=[]
testing_result_lines=[]

for line in lines:
    line = re.sub('[:/,()\[\]\s+]', ' ', line)

    items = line.split(' ')

    if items[0] == 'Epoch':
        train_lines.append(line)
    elif items[0] == 'Testing':
        test_lines.append(line)
#    elif items[0] == 'Test':
#        testing_result_lines.append(line)

train_epochs = []
learning_rates = []
train_losses = []
train_top1_predes = []
train_top5_predes = []

test_epochs = []
test_losses = []
test_top1_predes = []
test_top5_predes = []


#TODO: epoch, lr, loss, top1_pred, top5_pred
for line in train_lines:
    # epoch, value, lr:, value, time, value, value, data, value, value,
    items = line.split(' ')

    # erase empty list
    while True:
        try:
            items.remove("")
        except ValueError:
            break

    train_epochs.append((float(items[1])+(float(items[2])/float(items[3]))))
    learning_rates.append(float(items[5]))
    train_losses.append(float(items[13]))
    train_top1_predes.append(float(items[16]))
    train_top5_predes.append(float(items[19]))



#TODO: x (x:test/5epoch), loss, top1_pred, top5_pred


epochs = 0
for line in test_lines:
    items = line.split(' ')
    epochs = epochs + 5

    # erase empty list
    while True:
        try:
            items.remove("")
        except ValueError:
            break

    test_epochs.append(epochs)

    test_top1_predes.append(float(items[3]))
    test_top5_predes.append(float(items[5]))
    test_losses.append(float(items[7]))



#TODO: matplotlib plot + subplots for drawing graph
max_train_top1_pred = max(train_top1_predes)
max_train_top5_pred = max(train_top5_predes)
best_train_epoch = train_epochs[train_top1_predes.index(max_train_top1_pred)]
test_epoch_index = test_epochs.index(int(best_train_epoch/5)*5)
best_test_top1_pred = test_top1_predes[test_epoch_index]

print('train_best:', best_train_epoch)
print('top1:', max_train_top1_pred)
print('top5:', max_train_top5_pred)
print('test_top1', best_test_top1_pred)

plt.subplot(211)
plt.plot(train_epochs, train_losses, 'r')
plt.plot(test_epochs, test_losses, 'b')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(212)
plt.plot(train_epochs, train_top1_predes, 'r')
plt.plot(train_epochs, train_top5_predes, 'b')
plt.plot(test_epochs, test_top1_predes, 'g')
plt.plot(test_epochs, test_top5_predes, 'y')
plt.xlabel('epochs')
plt.ylabel('prediction')
#plt.subplot(311)
#plt.plot(train_epochs, learning_rates)
plt.show()
