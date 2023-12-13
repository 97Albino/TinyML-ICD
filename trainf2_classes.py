import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import random
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.autograd import Variable
from model import NetworkIEGM as Network
from model_onnx import NetworkIEGM as Network_onnx
from help_code_demo import ToTensor, IEGM_DataSET, FB
from profile import profile



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-2, help='weight decay')
parser.add_argument('--size', type=int, default=1250)
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--trained_path', type=str, default='saved_models', help='path to the trained model for onnx conversion')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--get_ops', action='store_true', default=False, help='calculate ops')
parser.add_argument('--onnx', action='store_true', default=False, help='calculate ops')
parser.add_argument('--reverse_data_split', action='store_true', default=False, help='use test data as training and training data as test')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--analyze_data', action='store_true', default=False, help='analyze data')
parser.add_argument('--diverse_train_split', action='store_true', default=False, help='Generate a third hidden split to test performance')
parser.add_argument('--stable_metric', action='store_true', default=False, help='Use a stability metric that chosses model with best validation performance but also similar training acc to ensure stable results across dataset.')
##PLEASE IGNORE THIS ARG AS IT IS NOT FULLY TESTED YET. NETWORK SHOULD TRAIN FINE WITHOUT THIS
parser.add_argument('--hidden_split', action='store_true', default=False, help='Generate a third hidden split to test performance')


args = parser.parse_args()

DIVERSE_TRAIN_SPLIT = args.diverse_train_split
analyze_labels = args.analyze_data
USE_STABLE_METRIC = args.stable_metric
HIDDEN =args.hidden_split

SPLIT_HIDDEN = 20
SPLIT_TEST = 20

SPLIT_RATIO=2
if not HIDDEN:
  SPLIT_HIDDEN=0
SPLIT_TRAIN = 76-SPLIT_HIDDEN-SPLIT_TEST


all_data = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21,
       23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 40, 41, 43, 44,
       45, 46, 48, 49, 50, 51, 53, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 85, 86, 88, 89, 90,
       91, 93, 94, 95]

def prepare_data_indices():
  IDs = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21,
         23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 40, 41, 43, 44,
         45, 46, 48, 49, 50, 51, 53, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68,
         69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 85, 86, 88, 89, 90,
         91, 93, 94, 95]
  train_ids = []
  hidden_ids = []

  #Suffle Ids mutiple times
  for i in range(10):
    random.shuffle(IDs)

  for i in range(SPLIT_HIDDEN):
    randchooser = random.randint(0, SPLIT_TRAIN - 1 - i)

    subid = IDs[randchooser]
    IDs.remove(subid)
    hidden_ids.append(subid)

  for i in range(SPLIT_TRAIN):
    randchooser = random.randint(0, SPLIT_TRAIN - 1 - i)

    subid = IDs[randchooser]
    IDs.remove(subid)
    train_ids.append(subid)

  verify_overlap = True

  for subs in IDs:
    if subs in train_ids or subs in hidden_ids:
      verify_overlap = False
  for subs in hidden_ids:
    if subs in train_ids :
      verify_overlap = False
  assert verify_overlap

  test_ids = IDs

  return train_ids, test_ids, hidden_ids


beta =2
eps = 1e-10
def fb(y_true, y_pred, beta=2):
  eps = 1e-10
  y_pred = torch.round(y_pred.float())
  tp = torch.sum(y_true * y_pred, axis=0)
  tn = torch.sum((1 - y_true) * (1 - y_pred), axis=0)
  fp = torch.sum((1 - y_true) * y_pred, axis=0)
  fn = torch.sum(y_true * (1 - y_pred), axis=0)


  # f1 = torch.where(torch.is_nan(f1), torch.zeros_like(f1), f1)
  return tp,tn,fp,fn

def fb_loss(y_true, y_pred, beta=2):
  y_pred = torch.nn.functional.softmax(y_pred)

  eps = 1e-10
  #y_pred = torch.round(y_pred)


  tp = torch.sum(y_true * y_pred[:,1], axis=0)
  tn = torch.sum((1 - y_true) * (y_pred[:,0]), axis=0)
  fp = torch.sum((1 - y_true) * y_pred[:,1], axis=0)
  fn = torch.sum(y_true * y_pred[:,0], axis=0)

  p = tp / (tp + fp + eps)
  r = tp / (tp + fn + eps)

  fb = (1 + (beta ** 2)) * ((p * r) / (((beta ** 2) * p) + r + eps))
  #f1 = torch.where(torch.is_nan(f1), torch.zeros_like(f1), f1)
  return 1 - fb



if not args.get_ops:
  args.save = 'eval-{}-{}-layers-{}'.format(args.save, args.layers,time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)



CLASSES = 2
path_data = args.data
path_indices = './data_indices'
SIZE =  args.size
BATCH_SIZE = args.batch_size
BATCH_SIZE_TEST = args.batch_size


def main():
  if args.get_ops:
    model2 = Network_onnx()
    print(model2)
    model2.eval()
    num_ops, num_params = profile(model2, [1, 1, 1,1250])
    print("========================================")
    print("#OPS: " + str(8*num_ops/10**3))
    print("#parmas: " + str(num_params))
    print("========================================")
    return
  if args.onnx:
    model3 = Network_onnx()
    cp = torch.load(args.trained_path)

    model3.load_state_dict(cp.state_dict(), strict=True)
    dummy_in = torch.rand([1,1,1,1250])

    torch.onnx.export(model3, dummy_in, 'modelv3.onnx', verbose=True)
    print(model3)


  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  good_split_found = False
  while (not good_split_found):

    train_ids, test_ids, hidden_ids = prepare_data_indices()

    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            ids=train_ids,
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))
    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           ids=test_ids,
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))
    if HIDDEN:
      hiddenset = IEGM_DataSET(root_dir=path_data,
                             indice_dir=path_indices,
                             ids=hidden_ids,
                             size=SIZE,
                             transform=transforms.Compose([ToTensor()]))
    fulldataset = IEGM_DataSET(root_dir=path_data,
                               indice_dir=path_indices,
                               ids=all_data,
                               size=SIZE,
                               transform=transforms.Compose([ToTensor()]))

    train_queue = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=2)

    if HIDDEN:
      hidden_queue = torch.utils.data.DataLoader(hiddenset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=2)

    alldata_queue = torch.utils.data.DataLoader(fulldataset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=2)

    test_total = analyzelabels(valid_queue, test_ids)

    train_total = analyzelabels(train_queue, train_ids)

    good_split_found = True
    if DIVERSE_TRAIN_SPLIT:
      for i in range(train_total.shape[1]):
        if train_total[0,i]/test_total[0,i]<1.8:
          good_split_found = False
          print("\n\n\n================\nBAD SPLIT FOUND\n===========================\n\n\n")



  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)


  model = Network()
  model_best = Network()
  model_best_hidden = Network()


  model = model.cuda()
  model_best = model_best.cuda()
  model_best_hidden = model_best_hidden.cuda()



  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  print(model)

  best = 0
  best_hidden = 0

  if args.onnx:
    #print(cp)
    print(model3)
    #return
    infer(valid_queue,model3.cuda(), criterion)
    return

  for epoch in range(args.epochs):

    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs





    train_acc, fbscore, train_obj = train(train_queue, model, criterion, optimizer, epoch)

    full_f2, full_acc, full_obj,fullstd = infer(alldata_queue, model, criterion)
    train_f2, train_acc, train_obj, trainstd = infer(train_queue, model, criterion)
    valid_f2,valid_sc, valid_obj, validstd = infer(valid_queue, model, criterion)
    if HIDDEN:
      hidden_f2, hidden_acc, hidden_obj = inferorg(hidden_queue, model, criterion)
    logging.info('valid_acc %f', valid_f2)



    if(((train_f2-valid_f2).abs()<=0.02)or((train_f2>0.96) and (valid_f2>train_f2)) or (not USE_STABLE_METRIC)):
      if valid_f2>best:
        best = valid_f2
        logging.info('BEST_acc %f', best)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        utils.save_model(model, os.path.join(args.save, 'model.pt'))
        model_best.load_state_dict(model.state_dict())
        #if ((fbscore-valid_acc).abs()<=0.03):
        #  print("FOUND")
        #  model_best2.load_state_dict(model.state_dict())
    if HIDDEN:
      if hidden_f2 > best_hidden:
          model_best_hidden.load_state_dict(model.state_dict())
          best_hidden = hidden_f2



    print("BEST:  "+str(best))
    print("BEST HIDDEN:  "+str(best_hidden))
    print('=============================================')
    print(f"FULL DATA F2: {full_f2.item():.4f}\tFULL DATA PPA: {full_obj:.4f} +- {fullstd:.4f} \tFULL DATA ACC: {full_acc.item():.4f}")
    print(f"TRAIN DATA F2: {train_f2.item():.4f}\tTRAIN DATA PPA: {train_obj:.4f} +- {trainstd:.4f}\tTRAIN DATA ACC: {train_acc.item():.4f}")
    print(f"VALID DATA F2: {valid_f2.item():.4f}\tVALID DATA PPA: {valid_obj:.4f} +- {validstd :.4f}\tVALID DATA ACC: {valid_sc.item():.4f}")
    if HIDDEN:
      print(f"HIDDEN F2: {hidden_f2.item():.4f}\tHIDDEN ACC: {hidden_acc.item():.4f}")
    print('=============================================')
    #print("PARAMS: " + str(pytorch_total_params))
  if best==0:
      print("MODEL NOT FOUND")
  else:
    full_f2, full_acc, full_obj,fullstd = infer(alldata_queue, model_best, criterion)
    train_f2, train_acc, train_obj, trainstd = infer(train_queue, model_best, criterion)
    valid_f2, valid_sc, valid_obj, validstd = infer(valid_queue, model_best, criterion)
    if HIDDEN:
      hidden_f2, hidden_acc, hidden_obj = infer(hidden_queue, model_best, criterion)

    if HIDDEN:
      full_f2h, full_acch, full_obj = infer(alldata_queue, model_best_hidden, criterion)
      train_f2h, train_acch, valid_obj = infer(train_queue, model_best_hidden, criterion)
      valid_f2h, valid_sch, valid_obj = infer(valid_queue, model_best_hidden, criterion)
      hidden_f2h, hidden_acch, hidden_obj = infer(hidden_queue, model_best_hidden, criterion)



    print("BEST MODEL")

    logging.info("==============================================")
    logging.info(f"FULL DATA F2: {full_f2.item():.4f}\tFULL DATA PPA: {full_obj:.4f} +- {fullstd:.4f} \tFULL DATA ACC: {full_acc.item():.4f}")
    logging.info(f"TRAIN DATA F2: {train_f2.item():.4f}\tTRAIN DATA PPA: {train_obj:.4f} +- {trainstd:.4f}\tTRAIN DATA ACC: {train_acc.item():.4f}")
    logging.info(f"VALID DATA F2: {valid_f2.item():.4f}\tVALID DATA PPA: {valid_obj:.4f} +- {validstd :.4f}\tVALID DATA ACC: {valid_sc.item():.4f}")
    if HIDDEN:
      logging.info(f"HIDDEN F2: {hidden_f2.item():.4f}\tHIDDEN ACC: {hidden_acc.item():.4f}")
    logging.info("==============================================")

    if HIDDEN:
      print("BEST HIDDEN MODEL")
      print("==============================================")
      print(f"FULL DATA F2: {full_f2h.item():.4f}\tFULL DATA ACC: {full_acch.item():.4f}")
      print(f"TRAIN DATA F2: {train_f2h.item():.4f}\tTRAIN DATA ACC: {train_acch.item():.4f}")
      print(f"VALID DATA F2: {valid_f2h.item():.4f}\tVALID DATA ACC: {valid_sch.item():.4f}")
      print(f"HIDDEN F2: {hidden_f2h.item():.4f}\tHIDDEN ACC: {hidden_acch.item():.4f}")
      print("==============================================")


  if analyze_labels:
    analyzelabels(valid_queue, test_ids)

    analyzelabels(train_queue, train_ids)
    if HIDDEN:
      analyzelabels(hidden_queue, hidden_ids)

    return






def train(train_queue, model, criterion, optimizer, epoch):

  model.train()
  running_loss = 0.0
  correct = 0.0
  total_correct = 0
  accuracy = 0.0
  total = 0

  i=0
  tp = 0
  tn = 0
  fn = 0
  fp = 0

  for step, data in enumerate(train_queue):

    input, target, clas = data['IEGM_seg'], data['label'], data['class']

    input = Variable(input).float().cuda()
    target = Variable(target).cuda()
    clas = Variable(clas).cuda()


    #if input.size(0) < args.batch_size:
    #  return 0, 0
    BATCH_SIZE = target.size(0)
    total+=BATCH_SIZE
    optimizer.zero_grad()

    input = input.reshape(input.shape[0],1,1,1250)


    logits_cls = model(input)




    loss = criterion(logits_cls, target) + 25 * fb_loss(target, logits_cls)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()


    _, predicted = torch.max(logits_cls.data, 1)


    correct += (predicted == target).sum()
    total_correct += correct



    tpt, tnt, fpt, fnt = fb(target, predicted)
    tp += tpt
    fp += fpt
    tn += tnt
    fn += fnt





    correct = 0.0
    running_loss += loss.item()
    i += 1

    if step % 10 == 0:
      print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
            (epoch + 1, i, total_correct/total, running_loss / i))
  p = tp / (tp + fp + eps)
  r = tp / (tp + fn + eps)

  fbacc = (1 + (beta ** 2)) * ((p * r) / (((beta ** 2) * p) + r + eps))

  print("TOTAL TRAiN DATA: "+str(total))

  return total_correct / total, fbacc, running_loss / i


def inferorg(valid_queue, model, criterion):

  model.eval()
  model.eval()
  running_loss = 0.0
  correct = 0.0
  accuracy = 0.0
  total = 0.0
  i = 0
  fbacc= 0
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  for step, data in enumerate(valid_queue):
    input, target, clas, subject = data['IEGM_seg'], data['label'], data['class'], data['subject']

    with torch.no_grad():
      input = Variable(input).float().cuda()
      target = Variable(target).cuda()
    input = input.reshape(input.shape[0], 1, 1, 1250)
    logits=  model(input)



    loss = criterion(logits, target)


    _, predicted = torch.max(logits.data, 1)


    total += target.size(0)


    tpt,tnt,fpt,fnt = fb(target,predicted)
    tp+=tpt
    fp+=fpt
    tn+=tnt
    fn+=fnt

    correct += (predicted == target).sum()

    accuracy += correct / BATCH_SIZE

    running_loss += loss#.item()
    i += 1
  print('--------------------------------------------')
  p = tp / (tp + fp + eps)
  r = tp / (tp + fn + eps)

  fbacc = (1 + (beta ** 2)) * ((p * r) / (((beta ** 2) * p) + r + eps))
  print('FB: %.5f ACC: %.5f Test Loss: %.5f' % (fbacc, correct/total, running_loss / i))

  print("TOTAL TEST DATA: "+str(total))


  return fbacc, correct/total, running_loss / i


def infer(valid_queue, model, criterion, printmode=False):

  model.eval()
  model.eval()
  running_loss = 0.0
  correct = 0.0
  accuracy = 0.0
  total = 0.0
  i = 0
  fbacc= 0
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  acc_dict = {}
  for step, data in enumerate(valid_queue):
    input, target, clas, subject = data['IEGM_seg'], data['label'], data['class'], data['subject']

    with torch.no_grad():
      input = Variable(input).float().cuda()
      target = Variable(target).cuda()
    input = input.reshape(input.shape[0], 1, 1, 1250)
    logits=  model(input)


    _, predicted = torch.max(logits.data, 1)

    loss = criterion(logits, target)

    total += target.size(0)


    tpt,tnt,fpt,fnt = fb(target,predicted)
    tp+=tpt
    fp+=fpt
    tn+=tnt
    fn+=fnt

    correct += (predicted == target).sum()

    accuracy += correct / BATCH_SIZE

    running_loss += loss.item()

    dict = {}
    corr = (predicted == target)


    for i in range(len(subject)):
      if int(subject[i]) not in dict:
        dict[int(subject[i])] = 0
      dict[int(subject[i])] = [int(corr[i]), 1]

    for key, value in dict.items():
      if key in acc_dict:
        acc_dict[key][0] += value[0]
        acc_dict[key][1] += 1
      else:
        acc_dict[key] = value

    i += 1
  if printmode:
    print('--------------------------------------------')
  p = tp / (tp + fp + eps)
  r = tp / (tp + fn + eps)

  fbacc = (1 + (beta ** 2)) * ((p * r) / (((beta ** 2) * p) + r + eps))
  print('FB: %.5f ACC: %.5f Test Loss: %.5f' % (fbacc, correct/total, running_loss / i))

  print("TOTAL TEST DATA: "+str(total))

  for key, value in acc_dict.items():
    acc_dict[key] = value[0] / value[1]
  if printmode:
    print(len(acc_dict))
    for i in sorted(acc_dict):
      print("{}: {}".format(i, acc_dict[i]))

  PPA = 0
  Subs = 0
  allacc = []
  for key, value in acc_dict.items():
    PPA+=value
    Subs+=1
    allacc.append(value)

  PPmean = torch.mean(torch.tensor(allacc))
  PPstd = torch.std(torch.tensor(allacc))




  return fbacc, correct/total, PPA/Subs, PPstd

Classes = {'VFb':0,
            'VFt':1,
            'VT':2,
            'AFb':3,
            'AFt':4,
            'SR':5,
            'SVT':6,
            'VPD':7}
def analyzelabels(valid_queue, ids):

  size = len(ids)
  map = {}
  for i in range(size):
    map[str(ids[i])] = i
  out = torch.zeros((size,8))


  running_loss = 0.0
  correct = 0.0
  accuracy = 0.0
  total = 0.0
  i = 0
  fbacc= 0
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  print(ids)
  total = torch.zeros(1,8)
  for step, data in enumerate(valid_queue):
    input, target, clas, subject = data['IEGM_seg'], data['label'], data['class'], data['subject']

    with torch.no_grad():
      input = Variable(input).float().cuda()
      target = Variable(target).cuda()
      clas = Variable(clas).cuda()
      #print(torch.argmax(clas))
      #[1]

      #print(subject)
      #print(map)
      for i in range(subject.shape[0]):
        clas_id = torch.argmax(clas[i])
        total[0,clas_id]+=1
        out[map[str(subject[i].item())], clas_id]+=1


  for i in range(size):
    ones = out[i,0].item()+out[i,1].item()+out[i,2].item()
    onest = total[0,0].item()+total[0,1].item()+total[0,2].item()
    zeros = out[i,3].item()+out[i,4].item()+out[i,5].item()+out[i,6].item()+out[i,7].item()
    zerost = total[0,3].item()+total[0,4].item()+total[0,5].item()+total[0,6].item()+total[0,7].item()
    print("SUBJECT#"+str(i)+'\t'+str(out[i,0].item())+'\t'+str(out[i,1].item())+'\t'+str(out[i,2].item())+'\t||\t'+str(out[i,3].item())+'\t'+str(out[i,4].item())+'\t'+str(out[i,5].item())+'\t'+str(out[i,6].item())+'\t'+str(out[i,7].item())+"\t||\t"+str(ones)+"\t||\t"+str(zeros))
  i=0
  print("TOTAL#"+'\t\t'+str(total[i,0].item())+'\t'+str(total[i,1].item())+'\t'+str(total[i,2].item())+'\t||\t'+str(total[i,3].item())+'\t'+str(total[i,4].item())+'\t'+str(total[i,5].item())+'\t'+str(total[i,6].item())+'\t'+str(total[i,7].item())+"\t||\t"+str(onest)+"\t||\t"+str(zerost))



  print("\n\n\n =========================================================== \n\n\n")
  return total



if __name__ == '__main__':
  main() 


