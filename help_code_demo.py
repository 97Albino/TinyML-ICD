import csv, torch, os
import random

import numpy as np




SPLIT_TEST = 16
SPLIT_TRAIN = 60



def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1

def stats_report(mylist):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print("F-1 = ", F1(mylist))
    print("F-B = ", FB(mylist))
    print("SEN = ", Sensitivity(mylist))
    print("SPE = ", Specificity(mylist))
    print("BAC = ", BAC(mylist))
    print("ACC = ", ACC(mylist))
    print("PPV = ", PPV(mylist))
    print("NPV = ", NPV(mylist))

    return output

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }

class IEGM_DataSET_analyze():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))
        if(mode == 'train'):
            csvdata_alltest = loadCSV(os.path.join(self.indice_dir, 'test' + '_indice.csv'))

        #################################
        #i=0
        subjects_train = []
        subjects_test = []
        num_train=0
        num_test=0
        ###################################

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))
            ###################################
            if mode=="train":
                sub = k.split('-')[0][1:]
                if sub not in subjects_train:
                    num_train+=1
                    subjects_train.append(sub)
            ###################################
        ###################################
        for i, (k, v) in enumerate(csvdata_alltest.items()):
            if mode=="train":
                sub = k.split('-')[0][1:]
                if sub not in subjects_test:
                    num_test+=1
                    subjects_test.append(sub)
        print(subjects_train)
        print(num_train)
        print("\n\n\n############################################################################################################################################\n\n\n")
        print(subjects_test)
        print(num_test)
        all_subs = (subjects_test + subjects_train)
        #for i in range(0, len(all_subs)):
        #    all_subs[i] = int(all_subs[i])
        #print(torch.sort(torch.tensor(all_subs)).values)
        #print(torch.sort(torch.tensor(all_subs)).values.shape)

        train_ids =  []

        for i in range(SPLIT_TRAIN):
            randchooser = random.randint(0,SPLIT_TRAIN-1-i)

            subid = all_subs[randchooser]
            all_subs.remove(subid)
            train_ids.append(subid)

        print(all_subs)
        print(len(all_subs))
        print(train_ids)
        print(len(train_ids))

        verify_overlap = True

        for subs in all_subs:
            if subs in train_ids:
                verify_overlap = False
        if verify_overlap:
            print('no overlap')

        ###################################

        #print(i)

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample

Classes = {'VFb':0,
            'VFt':1,
            'VT':2,
            'AFb':3,
            'AFt':4,
            'SR':5,
            'SVT':6,
            'VPD':7}

class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, ids, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []

        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, 'train' + '_indice.csv'))

        csvdata_alltest = loadCSV(os.path.join(self.indice_dir, 'test' + '_indice.csv'))

        #################################
        #i=0
        subjects_train = []
        subjects_test = []
        num_train=0
        num_test=0


        for i, (k, v) in enumerate(csvdata_all.items()):
            #self.names_list.append(str(k) + ' ' + str(v[0]))
            sub = int(k.split('-')[0][1:])
            if sub in ids:
                self.names_list.append(str(k) + ' ' + str(v[0]))

        for i, (k, v) in enumerate(csvdata_alltest.items()):
                sub = int(k.split('-')[0][1:])
                if sub in ids:
                    self.names_list.append(str(k) + ' ' + str(v[0]))





    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        Class = torch.zeros(8)

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        name_id = int(self.names_list[idx].split('-')[0][1:])
        #print(names)
        Class[Classes[self.names_list[idx].split('-')[1]] ]= 1

        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label,'class':Class, 'subject': name_id}

        return sample





class IEGM_DataSETorg():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))





        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample


def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path).cpu()
    #net = torch.load(net_path, map_location=torch.device('cpu')).float()
    #, map_location=torch.device('cpu')).float()
    dummy_input = torch.rand(1,1,size,1)
    #print(dummy_input.size())

    optName = str(net_name)+'.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True,opset_version=11)
