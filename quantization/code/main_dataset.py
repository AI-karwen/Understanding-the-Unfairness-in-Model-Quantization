from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../../../..")
import os
import pandas as pd
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader
from models import nin_gc, nin, resnet
from CustomUTK import UTKDataset
from MultNN import TridentNN
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.utils as utils
from MyLoss import MyCrossEntropyLoss


import quantize
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_state(model, best_acc):
    print("==> Saving model ...")
    state = {
        "best_acc": best_acc,
        "state_dict": model.state_dict(),
    }
    state_copy = state["state_dict"].copy()
    for key in state_copy.keys():
        if "module" in key:
            state["state_dict"][key.replace("module.", "")] = state["state_dict"].pop(
                key
            )
    if args.model_type == 0:
        if args.bn_fuse:
            if args.prune_quant or args.prune_qaft:
                torch.save(
                    {
                        "cfg": cfg,
                        "best_acc": best_acc,
                        "state_dict": state["state_dict"],
                    },
                    "models_save/nin_bn_fused.pth",
                )
            else:
                torch.save(state, "models_save/nin_bn_fused.pth")
        else:
            if args.prune_quant or args.prune_qaft:
                torch.save(
                    {
                        "cfg": cfg,
                        "best_acc": best_acc,
                        "state_dict": state["state_dict"],
                    },
                    "models_save/nin.pth",
                )
            else:
                torch.save(state, "models_save/nin.pth")
    elif args.model_type == 1:
        if args.bn_fuse:
            if args.prune_quant or args.prune_qaft:
                torch.save(
                    {
                        "cfg": cfg,
                        "best_acc": best_acc,
                        "state_dict": state["state_dict"],
                    },
                    "models_save/nin_gc_bn_fused.pth",
                )
            else:
                torch.save(state, "models_save/nin_gc_bn_fused.pth")
        else:
            if args.prune_quant or args.prune_qaft:
                torch.save(
                    {
                        "cfg": cfg,
                        "best_acc": best_acc,
                        "state_dict": state["state_dict"],
                    },
                    "models_save/nin_gc.pth",
                )
            else:
                torch.save(state, "models_save/trident_float.pth")
    else:
        if args.bn_fuse:
            torch.save(state, "models_save/resnet_bn_fused.pth")
        else:
            torch.save(state, "models_save/resnet.pth")


def adjust_learning_rate(optimizer, epoch):
    update_list = [40,70]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1
    # print("args.lr",args.lr)
    return


def train(epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_gen.train()

    batch_num = 0
    # loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
    loop = enumerate(trainloader)
    grad_norms = []
    for batch_idx,(data,target) in loop:
        target_gen = target[:,1]
        if not args.cpu:
            data, target_gen = data.cuda(), target_gen.cuda()
        data, target_gen = Variable(data), Variable(target_gen)

        output = model_gen(data)
        # print(target_gen)
        output = output[1]

        loss = criterion_gen(output, target_gen)
        
        # PTQ doesn't need backward
        if not args.ptq_control:
            optimizer.zero_grad()
            loss.backward()

            '''
            print("参数.......",model_gen.parameters())
            for p in model_gen.parameters():
                print(p)
            grad_norm = torch.norm(torch.stack([p.grad.detach().norm() for p in model_gen.parameters()]))
            grad_norms.append(grad_norm.item())
            '''
            optimizer.step()

            
            '''
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}".format(
                        epoch,
                        batch_idx * len(data),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.data.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )'''
        else:
            return
            '''batch_num += 1
            if batch_num > args.ptq_batch:
                break
            print("Batch:", batch_num)'''
    return


def test():
    global best_acc
    model_gen.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_gen.to(device)
    test_loss = 0
    correct = 0
  
    for data, target in testloader:
        target_gen = target[:,1].to(device)
        if not args.cpu:
            data, target_gen = data.cuda(), target_gen.cuda()
        data, target_gen = Variable(data), Variable(target_gen)
        output = model_gen(data)
        output = output[1]

        

        test_loss += criterion_gen(output, target_gen).data.item()
        pred = output.data.max(1, keepdim=True)[1]

        correct_tensor = pred.eq(target_gen.data.view_as(pred))
        correct += pred.eq(target_gen.data.view_as(pred)).cpu().sum()
        correct_numpy = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())


        for i in range(target_gen.data.__len__()):
            label = target_gen.data[i]
            class_correct[label] += correct_numpy[i].item()
            class_total[label] += 1

        
    acc = 100.0 * float(correct) / len(testloader.dataset)


    if acc > best_acc:
        best_acc = acc
        save_state(model_gen, best_acc)
    average_test_loss = test_loss / (len(testloader.dataset) / args.eval_batch_size)

    print("epochs",epoch)

    for i in range(5):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            average_test_loss,
            correct,
            len(testloader.dataset),
            100.0 * float(correct) / len(testloader.dataset),
        )
    )

    save_state(model_gen, best_acc)
    print("Best Accuracy: {:.2f}%\n".format(best_acc))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--age_range",default="",help="年龄范围",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="set if only CPU is available"
    )
    parser.add_argument("--gpu_id", action="store", default="", help="gpu_id")
    parser.add_argument(
        "--data", action="store", default="../../../../data", help="dataset path"
    )
    parser.add_argument(
        "--lr", action="store", default=0.001, help="the intial learning rate"
    )
    parser.add_argument(
        "--wd", action="store", default=1e-5, help="the intial learning rate"
    )
    # prune_quant
    parser.add_argument(
        "--prune_quant",
        default="",
        type=str,
        metavar="PATH",
        help="the path to the prune_quant model",
    )
    # refine
    parser.add_argument(
        "--refine",
        default="",
        type=str,
        metavar="PATH",
        help="the path to the float_refine model",
    )
    # resume
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="the path to the resume model",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--start_epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train_start",
    )
    parser.add_argument(
        "--end_epochs",
        type=int,
        default=61,
        metavar="N",
        help="number of epochs to train_end",
    )
    # W/A — bits
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    # bn融合标志位
    parser.add_argument(
        "--bn_fuse", action="store_true", help="batch-normalization fuse"
    )
    # bn融合校准标志位
    parser.add_argument(
        "--bn_fuse_calib",
        action="store_true",
        help="batch-normalization fuse calibration",
    )
    # 量化方法选择
    parser.add_argument(
        "--q_type", type=int, default=0, help="quant_type:0-symmetric, 1-asymmetric"
    )
    # 量化级别选择
    parser.add_argument(
        "--q_level", type=int, default=0, help="quant_level:0-per_channel, 1-per_layer"
    )
    # weight_observer选择
    parser.add_argument(
        "--weight_observer",
        type=int,
        default=0,
        help="quant_weight_observer:0-MinMaxObserver, 1-MovingAverageMinMaxObserver",
    )
    # pretrained_model标志位
    parser.add_argument(
        "--pretrained_model", action="store_true", help="pretrained_model"
    )
    # qaft标志位
    parser.add_argument(
        "--qaft", action="store_true", help="quantization-aware-finetune"
    )
    # prune_qaft
    parser.add_argument(
        "--prune_qaft",
        default="",
        type=str,
        metavar="PATH",
        help="the path to the prune_qaft model",
    )
    # ptq_observer
    parser.add_argument("--ptq", action="store_true", help="post-training-quantization")
    # ptq_control
    parser.add_argument("--ptq_control", action="store_true", help="ptq control flag")
    # ptq_percentile
    parser.add_argument(
        "--percentile", type=float, default=0.999999, help="the percentile of ptq"
    )
    # ptq_batch
    parser.add_argument("--ptq_batch", type=int, default=41, help="the batch of ptq")
    parser.add_argument(
        "--model_type", type=int, default=1, help="model type:0-nin,1-nin_gc,2-resnet"
    )
    args = parser.parse_args()
    print("==> Options:", args)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    setup_seed(1)

    print("==> Preparing data..")

    # Read in the dataframe

    input_file = "../../../../data/age_gender.csv"
    dataFrame = pd.read_csv(input_file)

    # Construct age bins
    age_bins = [0,10,15,20,25,30,40,50,60,120]
    age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    dataFrame['bins'] = pd.cut(dataFrame.age, bins=age_bins, labels=age_labels)

    # Split into training and testing
    train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)

    # get the number of unique classes for each group
    class_nums = {'age_num':9, 'eth_num':5,
                  'gen_num':len(dataFrame['gender'].unique())}

    # Define train and test transforms
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    # Construct the custom pytorch datasets
    train_set = UTKDataset(train_dataFrame, transform=transform_train)
    test_set = UTKDataset(test_dataFrame, transform=transform_test)

    # Load the datasets into dataloaders
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Sanity Check
    for X, y in trainloader:
        print(f'Shape of training X: {X.shape}')
        print(f'Shape of y: {y.shape}')
        break

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    classes =(
        "white",
        "black",
        "Asian",
        "Indian",
        "Others"
    )
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    loss_class = list(0. for i in range(5))


    if args.prune_quant:
        print("******Prune Quant model******")
        # checkpoint = torch.load('../prune/models_save/nin_refine.pth')
        checkpoint = torch.load(args.prune_quant)
        cfg = checkpoint["cfg"]
        if args.model_type == 0:
            model = nin.Net(cfg=checkpoint["cfg"])
        else:
            model = nin_gc.Net(cfg=checkpoint["cfg"])
        model.load_state_dict(checkpoint["state_dict"])
        best_acc = 0
        print("***ori_model***\n", model)
        quantize.prepare(
            model,
            inplace=True,
            a_bits=args.a_bits,
            w_bits=args.w_bits,
            q_type=args.q_type,
            q_level=args.q_level,
            weight_observer=args.weight_observer,
            bn_fuse=args.bn_fuse,
            bn_fuse_calib=args.bn_fuse_calib,
            pretrained_model=args.pretrained_model,
            qaft=args.qaft,
            ptq=args.ptq,
            percentile=args.percentile,
        )
        print("\n***quant_model***\n", model)
    elif args.prune_qaft:
        print("******Prune QAFT model******")
        # checkpoint = torch.load('models_save/nin_bn_fused.pth')
        checkpoint = torch.load(args.prune_qaft)
        cfg = checkpoint["cfg"]
        if args.model_type == 0:
            model = nin.Net(cfg=checkpoint["cfg"])
        else:
            model = nin_gc.Net(cfg=checkpoint["cfg"])
        print("***ori_model***\n", model)
        quantize.prepare(
            model,
            inplace=True,
            a_bits=args.a_bits,
            w_bits=args.w_bits,
            q_type=args.q_type,
            q_level=args.q_level,
            weight_observer=args.weight_observer,
            bn_fuse=args.bn_fuse,
            bn_fuse_calib=args.bn_fuse_calib,
            pretrained_model=args.pretrained_model,
            qaft=args.qaft,
            ptq=args.ptq,
            percentile=args.percentile,
        )
        print("\n***quant_model***\n", model)
        model.load_state_dict(checkpoint["state_dict"])
        best_acc = checkpoint["best_acc"]
    elif args.refine:

        print("******Float Refine model******")
        # checkpoint = torch.load('models_save/nin.pth')
        
        checkpoint = torch.load(args.refine)
        if args.model_type == 0:
            model = nin.Net()
        elif args.model_type == 1:
            model_gen = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
        else:
            model = resnet.resnet18()
        best_acc = 0
        model_gen.load_state_dict(checkpoint["state_dict"])
        print("***ori_model***\n", model_gen)
        quantize.prepare(
            model_gen,
            inplace=True,
            a_bits=args.a_bits,
            w_bits=args.w_bits,
            q_type=args.q_type,
            q_level=args.q_level,
            weight_observer=args.weight_observer,
            bn_fuse=args.bn_fuse,
            bn_fuse_calib=args.bn_fuse_calib,
            pretrained_model=args.pretrained_model,
            qaft=args.qaft,
            ptq=args.ptq,
            percentile=args.percentile,
        )

        
        print("\n***quant_model***\n", model_gen)

        
    elif args.resume:
        print("******Reume model******")
        # checkpoint = torch.load('models_save/nin.pth')
        checkpoint = torch.load(args.resume)
        if args.model_type == 0:
            model = nin.Net()
        elif args.model_type == 1:
            model = nin_gc.Net()
        else:
            model = resnet.resnet18()
        
        print("***ori_model***\n", model)
        quantize.prepare(
            model,
            inplace=True,
            a_bits=args.a_bits,
            w_bits=args.w_bits,
            q_type=args.q_type,
            q_level=args.q_level,
            weight_observer=args.weight_observer,
            bn_fuse=args.bn_fuse,
            bn_fuse_calib=args.bn_fuse_calib,
            pretrained_model=args.pretrained_model,
            qaft=args.qaft,
            ptq=args.ptq,
            percentile=args.percentile,
        )
        print("\n***quant_model***\n", model)
        
        model.load_state_dict(checkpoint["state_dict"])
        best_acc = checkpoint["best_acc"]
    else:
        print("******Initializing model******")
        if args.model_type == 0:
            model = nin.Net()
        elif args.model_type == 1:
            model_gen= TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
        else:
            model = resnet.resnet18()

        best_acc = 0

        for m in model_gen.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)

        print("***ori_model***\n", model_gen)

        '''
        quantize.prepare(
            model_gen,
            inplace=True,
            a_bits=args.a_bits,
            w_bits=args.w_bits,
            q_type=args.q_type,
            q_level=args.q_level,
            weight_observer=args.weight_observer,
            bn_fuse=args.bn_fuse,
            bn_fuse_calib=args.bn_fuse_calib,
            pretrained_model=args.pretrained_model,
            qaft=args.qaft,
            ptq=args.ptq,
            percentile=args.percentile,
        )

        print("\n***quant_model***\n", model_gen)
        '''

    if not args.cpu:
        model_gen.cuda()
        model_gen = torch.nn.DataParallel(
            model_gen, device_ids=range(torch.cuda.device_count())
        )

    
    '''
    base_lr = float(args.lr)
    param_dict = dict(model_gen.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{"params": [value], "lr": base_lr, "weight_decay": args.wd}]
    
    '''
    criterion_gen = nn.CrossEntropyLoss()
    #criterion_train = MyCrossEntropyLoss()
    optimizer = optim.Adam(model_gen.parameters(), lr=args.lr)

    if args.ptq_control:
        args.end_epochs = 11
        print("ptq is doing...")

    start= time.perf_counter()


    for epoch in range(args.start_epochs, args.end_epochs):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
    
    
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))

    for epoch in range(args.start_epochs, args.end_epochs):
        #adjust_learning_rate(optimizer, epoch)
        #train(epoch)
        test()

    save_state(model_gen, best_acc)
    end = time.perf_counter()
    runTime = end - start

    print("运行时间：",runTime,"秒")

    if args.ptq_control:
        print("ptq is done")
