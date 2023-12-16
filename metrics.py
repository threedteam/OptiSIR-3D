import glob
import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc, recall_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
root = "./wsd/"

test_dir = root + "sensor_data/test"
writein = root + "results_metics.txt"
modelsave = root + 'results/model'
p = "c3dL_"
# p = "3c3d_"
print("P:", p)
output_re = open(writein, 'a')
output_re.write('\r\n')


def load_lc_data(path, indices=0, split=False):
    txt_Paths = glob.glob(os.path.join(path, '*.txt'))
    txt_Paths.sort()
    sensor_data = []
    sensor_label = []

    for txt_item in txt_Paths:
        label = txt_item.split('-')[1]
        data_z = pd.read_csv(txt_item, delimiter=',', header=None)
        data_z = np.asarray(data_z)
        # print(data_z.shape)
        data = data_z.reshape((1, 22, 181, 180))

        # sensor array optimization
        # if indices:
        if indices.any():
            data = data[0][indices]
            data = data.reshape((1, len(indices), 181, 180))

        # print(data.shape, type(data))
        if (label == 'health'):
            label_c = 0
        else:
            label_c = 1

        sensor_data.append(data)
        sensor_label.append(label_c)

    _data = np.array(sensor_data, dtype="float")
    _label = np.array(sensor_label, dtype="float")
    print(_data.shape)

    if split == True:
        # train,val-array,list
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            _data, _label,
            test_size=0.1,
            random_state=20,
            shuffle=True,
            stratify=_label
        )
        return x_train, x_val, y_train, y_val
    else:
        return _data, _label


args = {'batch': 8, 'epochs': 200, 'lr': 0.0005, 'momentum': 0.9, 'nw': 8}
N = 5
avg_acc = 0
auc_v = 0
sen = 0
spe = 0
lis = []
op = 8

# for op in (8, 9, 11, 13, 14):
for i in range(0, N):
    print("Turn {} for training.".format(i))
    args['optim_param'] = op

    ps = "m" + str(args['momentum']) + "[b" + str(args['batch']) + "e" + str(args['epochs']) + "i" + str(
        i) + "lr" + str(args['lr']) + p + str(args['optim_param']) + "]"
    output_re.write(ps + '\n')
    args['ps'] = ps
    ## ---------------------- Dataload ---------------------- ##
    # SAO
    optim_param = args['optim_param']
    indices = [3, 21, 9, 10, 14, 6, 0, 20, 7, 2, 13, 15, 12, 18, 4, 1, 16, 19, 8, 11, 17, 5]  # MAX
    indices = indices[:optim_param]
    indices.sort()

    #SIR
    index = [0, 6, 4, 2, 1, 3, 7, 5]  # Max
    # index = [6, 7, 2, 0, 5, 3, 1, 4]  # Min
    # sensor array recombination
    if index:
        indices = np.array(indices)  # [ 0  1  4  7  8 12 13 16 20 21]
        re_indices = indices[index]  # [ 7  4 12  0  8  1 16 20 21 13]
    print('re_indices: ', re_indices)
    output_re.write('re_indices: ' + str(re_indices) + '\t')
    indices = re_indices

    print('indices: ',indices)
    output_re.write('indices: ' + str(indices) + '\t')

    # test_data, test_label = load_lc_data(test_dir, indices=0)
    test_data, test_label = load_lc_data(test_dir, indices)

    torch_test = torch.from_numpy(test_data)
    test_num = len(torch_test)
    torch_testL = torch.tensor(test_label)
    test_ids = TensorDataset(torch_test, torch_testL)
    test_loader = DataLoader(dataset=test_ids, batch_size=args['batch'], shuffle=False, num_workers=args['nw'])
    ## ---------------------- test ---------------------- ##
    y_pre = []
    y_label = []
    y_pre_2 = []

    # end model
    print(modelsave + '/' + p + '-i' + str(i + 1) + 'end.pt')
    model2 = torch.load(modelsave + '/' + p + '-i' + str(i + 1) + 'end.pt')
    print("==========")
    print(model2)

    model2.eval()
    correct2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model2(data)  # tensor([[-0.5464,  0.4531],[ 7.3246, -7.4213],...
            m = nn.Softmax(dim=1)
            prob = m(output)
            print(prob)
            pre = prob[:,1]
            print(pre)
            predict_2 = torch.max(output, dim=1)[1]  # tensor([1, 0, 0, 1, 1, 0, 1, 1])

            y_pre.extend(pre)
            y_label.extend(target)
            y_pre_2.extend(predict_2)

            correct2 += torch.eq(predict_2, target.to(device)).sum().item()

    print('\nEndmodel Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct2, len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))
    args['Acc'] = correct2 / len(test_loader.dataset)

    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    output_re.write('time:' + str(t) + '\n')

    # args['auc_value'], args['sensitivity'], args['specificity'] = metrics(y_pre,y_label,i+1,ps)
    # auc_value,sensitivity,specificity = metrics(y_pre,y_label,i+1,ps)

    y_pre = torch.tensor(y_pre, device='cpu').numpy()
    y_label = torch.tensor(y_label, device='cpu').numpy()
    y_pre_2 = torch.tensor(y_pre_2, device='cpu').numpy()

    fpr, tpr, thresholds = roc_curve(y_label, y_pre)
    plt.plot(fpr, tpr, c="b", clip_on=False)
    # plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], ls='--', c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.xlabel('False Positivite Rate (1 - specificity)')
    plt.ylabel('True Positivite Rate (sensitivite)')
    plt.grid(True)
    plt.savefig(str(ps) + str(i + 1) + 'roc.png')
    plt.show()

    auc_value = auc(fpr, tpr)
    sensitivity = recall_score(y_label, y_pre_2)
    specificity = recall_score(np.logical_not(y_label), np.logical_not(y_pre_2))
    print(auc_value)
    print(sensitivity)
    print(specificity)

    print('\nauc, sen, spe: {},{},{}\n'.format(auc_value, sensitivity, specificity))
    output_re.write('auc, sen, spe: ' + str(auc_value) + '---' + str(sensitivity) + '---' + str(specificity) + '\t')
    print(args['Acc'])
    lis.append(args['Acc'])
    avg_acc += args['Acc']
    auc_v += auc_value
    sen += sensitivity
    spe += specificity

avg_acc = avg_acc / N
auc_v = auc_v / N
sen = sen / N
spe = spe / N

print(lis)
output_re.write('The average accuracy of ' + str(N) + ' times is ' + str(avg_acc) + '\n')
output_re.write(
    'The average auc,sen,spe of ' + str(N) + ' times is ' + str(auc_v) + '---' + str(sen) + '---' + str(spe) + '\n')
output_re.write('The best of ps' + args['ps'] + str(N) + ' are' + str(lis) + '\n')
