import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import torchvision.transforms as transforms

SHUFFLE= True

if SHUFFLE:
    from help_code_demo import ToTensor, IEGM_DataSET, stats_report
else:
    from help_code_demo1 import ToTensor, IEGM_DataSET, stats_report

IDs = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21,
       23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 40, 41, 43, 44,
       45, 46, 48, 49, 50, 51, 53, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 85, 86, 88, 89, 90,
       91, 93, 94, 95]
train_ids = []
SPLIT_TRAIN=56
for i in range(10):
    random.shuffle(IDs)
for i in range(SPLIT_TRAIN):
    randchooser = random.randint(0, SPLIT_TRAIN - 1 - i)

    subid = IDs[randchooser]
    IDs.remove(subid)
    train_ids.append(subid)



test_ids = IDs

def fb(y_true, y_pred,beta=2):
    eps = 1e-10
    y_pred = torch.round(y_pred)
    tp =torch.sum(y_true*y_pred, axis=0)
    tn =torch.sum((1-y_true)*(1-y_pred), axis=0)
    fp = torch.sum((1-y_true)*y_pred, axis=0)
    fn = torch.sum(y_true*(1-y_pred),  axis=0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)


    fb = (1+(beta**2))*((p*r) / (((beta**2)*p)+r+eps))
    #f1 = torch.where(torch.is_nan(f1), torch.zeros_like(f1), f1)
    return fb.mean()

def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    BATCH_SIZE_TEST = 512
    SIZE = args.size
    path_data = args.path_data
    path_records = args.path_record
    path_net = args.path_net
    path_indices = args.path_indices
    #stats_file = open(path_records + 'seg_stat.txt', 'w')

    # load trained network
    net = torch.load(path_net, map_location='cuda:0')
    net.eval()
    net.cuda()
    device = torch.device('cuda:0')

    if SHUFFLE:
        testset = IEGM_DataSET(path_data,
                               path_indices,
                               test_ids,
                               SIZE,
                               transform=transforms.Compose([ToTensor()]))
    else:
        testset = IEGM_DataSET(root_dir=path_data,
                               indice_dir=path_indices,
                               mode='test',
                               size=SIZE,
                               transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    fbscore=0
    i=0
    for data_test in testloader:
        IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
        seg_label = deepcopy(labels_test)

        IEGM_test = IEGM_test.float().to(device)
        labels_test = labels_test.to(device)



        outputs_test = net(IEGM_test.reshape(IEGM_test.shape[0],1,1,1250))
        outputs_test = torch.nn.functional.softmax(outputs_test)
        _, predicted_test = torch.max(outputs_test.data, 1)

        #print(predicted_test)
        fbscore += fb(labels_test,predicted_test.float())

        #print(fbscore)
        i+=1
    print(fbscore/i)
    #print(f1score.mean())

    '''

    if seg_label == 0:
        segs_FP += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
        segs_TN += (predicted_test == labels_test).sum().item()
    elif seg_label == 1:
        segs_FN += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
        segs_TP += (predicted_test == labels_test).sum().item()
    '''
    # report metrics
    #stats_file.write('segments: TP, FN, FP, TN\n')
    #output_segs = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    #stats_file.write(output_segs + '\n')

    del net


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='/home/faaiz/Work/datasets/tinyml_contest_data_training/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/IEGM_net.pkl')
    argparser.add_argument('--path_record', type=str, default='./records/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
    print("device is --------------", device)

    main()
