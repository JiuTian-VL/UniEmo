from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
import sys
import warnings
sys.path.append("../")
import argparse
import logging
import math
import os
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.checkpoint
import transformers
import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
import yaml
import json
from EmoSet import EmoSet
import Emotion_clip
from optimizer import build_optimizer, build_scheduler
from utils import *
import clip
import pdb
def parse_args(learning_rate,test_only, model_resume, output_dir, data_dir,model_arch, seed, num_train_epochs, train_batch_size, weight_decay, warmup_epochs, fix_text, query_len, query_layer, cal_emotion_space):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_arch",
        type=str,
        default=model_arch,
        help="choose the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir,
        help="The train directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_emotion_classes",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--model_resume",
        type=str,
        default=model_resume,
        help=(
            'pretrained checkpoint.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--seed", type=int, default=seed, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=train_batch_size, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=weight_decay,
        help="weight_decay.",
    )
    parser.add_argument(
        "--fix_text",
        type=bool,
        default=fix_text,
        help="fix the text encoder.",
    )
    parser.add_argument(
        "--test_only",
        type=bool,
        default=test_only,
        help="only test.",
    )
    parser.add_argument(
        "--cal_emotion_space",
        type=bool,
        default=cal_emotion_space,
        help="cal_emotion_space during inference.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=warmup_epochs,
        help="warmup_epochs.",
    )
    parser.add_argument(
        "--query_len",
        type=int,
        default=query_len,
        help="The length of prompt tokens.",
    )
    parser.add_argument(
        "--query_layer",
        type=int,
        default=query_layer,
        help="The position for prompt to insert.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs)
    args = parser.parse_args()
    return args

class emo_classifier(nn.Module):
    def __init__(self,vision_width ):
        super(emo_classifier, self).__init__()
        self.fc0 = nn.Linear(vision_width, 768)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(768, 8)


    def forward(self, x):
        x = self.fc0(x)
        x = self.drop_out(self.relu(x))
        x = self.fc1(x)

        return x
class Scene_classifier(nn.Module):
    def __init__(self,vision_width ):
        super(Scene_classifier, self).__init__()
        self.fc0 = nn.Linear(vision_width, 254)
        # self.relu = nn.ReLU()
        # self.drop_out = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(768, 254)


    def forward(self, x):
        x = self.fc0(x)
        # x = self.drop_out(self.relu(x))
        # x = self.fc1(x)

        return x
    
class object_classifier(nn.Module):
    def __init__(self,vision_width ):
        super(object_classifier, self).__init__()
        self.fc0 = nn.Linear(vision_width, 409)
        # self.relu = nn.ReLU()
        # self.drop_out = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(768, 409)


    def forward(self, x):
        x = self.fc0(x)
        # x = self.drop_out(self.relu(x))
        # x = self.fc1(x)

        return x
def generate_text(data):
    text_aug = f"{{}}"
    text = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data])

    return text
def cross_entropy(preds, target, reduciton = 'mean'):
    log_softmax = nn.LogSoftmax(dim = -1)
    loss = (-target * log_softmax(preds)).sum(1)
    return loss.mean()
def main(args):
    # pdb.set_trace()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    # writer = SummaryWriter(log_dir=args.output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True
        )

 
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
        
    logger = create_logger(output_dir=args.output_dir, accelerator=accelerator, name=f"{args.model_arch}")
    
    logger.info(f"working dir: {args.output_dir}")
    # pdb.set_trace()
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    train_dataset = EmoSet(
        data_root=args.data_dir,
        num_emotion_classes=args.num_emotion_classes,
        phase='train',
    )
    val_dataset = EmoSet(
        data_root=args.data_dir,
        num_emotion_classes=args.num_emotion_classes,
        phase='val',
    )
    test_dataset = EmoSet(
        data_root=args.data_dir,
        num_emotion_classes=args.num_emotion_classes,
        phase='test',
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    model, _ , vision_width= Emotion_clip.load(model_path= None, name = args.model_arch,
                         device="cpu", jit=False, query_len=args.query_len, query_layer=args.query_layer, logger = logger
                        )
    # vision_width = model.state_dict["visual.conv1.weight"].shape[0]
    Emo_classifier = emo_classifier(vision_width)
    Object_classifier = object_classifier(vision_width)
    scene_classifier = Scene_classifier(vision_width)
    optimizer = build_optimizer(args, model, Emo_classifier, Object_classifier, scene_classifier)
    
    lr_scheduler = build_scheduler(args, optimizer, len(train_dataloader))
    start_epoch, max_accuracy = 0, 0.0
    # pdb.set_trace()

    # pdb.set_trace()
    optimizer, train_dataloader, lr_scheduler, model, Emo_classifier, val_dataloader, test_dataloader, Object_classifier, scene_classifier= accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, model, Emo_classifier, val_dataloader, test_dataloader, Object_classifier, scene_classifier
    )
    # 
    if args.model_resume:
        model = model.module if hasattr(model, 'module') else model
        Emo_classifier = Emo_classifier.module if hasattr(Emo_classifier, 'module') else Emo_classifier
        start_epoch, max_accuracy = load_checkpoint(args, model, Emo_classifier, optimizer, lr_scheduler, logger)
                
                
    weight_dtype = torch.float32
    info = json.load(open(os.path.join(args.data_dir, 'info.json')))
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    model.to(accelerator.device, dtype = weight_dtype)
    Emo_classifier.to(accelerator.device, dtype = weight_dtype)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    is_best = False
    # pdb.set_trace()
    if args.cal_emotion_space:
        save_dir = Path("UniEmo_space")
        save_dir.mkdir(parents=True, exist_ok=True)
        _ = cal_emotion_space(accelerator, train_dataloader, model, Emo_classifier, args, logger)
    if args.test_only:
        acc1 = validate(accelerator, test_dataloader, model, Emo_classifier, args, logger)
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {acc1:.1f}%")
        return
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_object = nn.BCELoss()
    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()
        # optimizer.zero_grad()
        num_steps = len(train_dataloader)
        start = time.time()
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.accumulate(Emo_classifier):
                    
                    contrast_scene_loss_1 = torch.tensor(0).to(accelerator.device)
                    contrast_object_loss_1 = torch.tensor(0).to(accelerator.device)
                    image = batch['image']
                    label = batch['emotion_label_idx']
                    scene_label = batch['scene_label_idx']#####scene_label [2]
                    object_one_hot_label = batch['object_label_idx']#######2x409
                    values, object_label = torch.topk(object_one_hot_label, 1)
                    object_label[values==0] = -1
                    object_label = object_label.squeeze()
                    # pdb.set_trace()
                    object_text, scene_text = clip.tokenize(batch['object_text'], context_length=77).to(accelerator.device), clip.tokenize(batch['scene_text'], context_length=77).to(accelerator.device)
                

                    cls_fea, visual_query_list, logits_per_image_scene, logits_per_image_object, logits_per_image_scene_last, logits_per_image_object_last = model(image, scene_text, object_text)
                    mask_scene = scene_label == -1
                    mask_object = object_label == -1
                    label_contrast_scene = torch.arange(image.shape[0]).to(accelerator.device)
                    label_contrast_scene[mask_scene] = -1
                    label_contrast_object = torch.arange(image.shape[0]).to(accelerator.device)
                    label_contrast_object[mask_object] = -1
                    
                    pred = Emo_classifier(cls_fea)###2x8

                    emotion_query_loss = criterion(pred, label)
                    
                    valid_scene_target = (scene_label.squeeze()!=-1).float().sum()
                    valid_object_target = (object_label.squeeze()!=-1).float().sum()
                    # pdb.set_trace()
                    
                    if valid_scene_target!=0:
                        pred_scene = scene_classifier(visual_query_list[0])
                        # scene_query_loss = criterion(pred_scene, scene_label.squeeze())
                        contrast_scene_loss_1 = (criterion(logits_per_image_scene, label_contrast_scene) + criterion(logits_per_image_scene.transpose(0, 1), label_contrast_scene))/2
                    if valid_object_target!=0:
                        # pdb.set_trace()
                        pred_object = Object_classifier(visual_query_list[1])
                        # object_query_loss = criterion_object(F.sigmoid(pred_object), object_one_hot_label)
                        contrast_object_loss_1 = (criterion(logits_per_image_object, label_contrast_object) + criterion(logits_per_image_object.transpose(0, 1), label_contrast_object))/2

                    total_loss = contrast_scene_loss_1*0.05 + contrast_object_loss_1*0.05 + 0.9*emotion_query_loss
                    accelerator.backward(total_loss)
                    optimizer.step()
                    # lr_scheduler.step_update(epoch*num_steps + step)
                    lr_scheduler.step()
                    optimizer.zero_grad()
            if accelerator.sync_gradients:
                batch_time = time.time() -end
                end = time.time()
                if step % 50 ==0:
                    # pdb.set_trace()
                    lr = optimizer.param_groups[0]['lr']
                    
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    etas = batch_time * (num_steps - step)
                    logger.info(
                    f'Train: [{epoch}/{args.num_train_epochs}][{step}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                    f'time {batch_time:.4f}\t'
                    f'total_loss {total_loss:.4f}\t'
                    f'emotion_cls_loss {emotion_query_loss:.4f}\t'
                    # f'scene_query_loss {scene_query_loss:.4f}\t'
                    # f'object_query_loss {object_query_loss:.4f}\t'
                    f'contrast_scene_loss {contrast_scene_loss_1:.4f}\t'
                    f'contrast_object_loss {contrast_object_loss_1:.4f}\t'
                    f'mem {memory_used:.0f}MB')
            # acc1 = validate(accelerator, val_dataloader, model,Emo_classifier, args, logger)
        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        acc1 = validate(accelerator, test_dataloader, model, Emo_classifier, args, logger)
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test emotion images: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if accelerator.is_main_process:
            
            epoch_saving(args, epoch, Emo_classifier, model, max_accuracy, optimizer, lr_scheduler, logger,args.output_dir, is_best, accelerator)

    logger.info(f"training finish")
import time
import torch

@torch.inference_mode()
def benchmark_model_forward_speed(accelerator, val_loader, model, Emo_classifier, logger,
                                  warmup_steps=10, measure_steps=200):
    model.eval()
    Emo_classifier.eval()

    device = accelerator.device
    use_cuda = (device.type == "cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # warmup
    for idx, batch_data in enumerate(val_loader):
        if idx >= warmup_steps:
            break
        _image = batch_data["image"]
        cls_fea, _, _ = model(_image, scene_text=None, object_text=None)
        _ = Emo_classifier(cls_fea)

    accelerator.wait_for_everyone()
    if use_cuda:
        torch.cuda.synchronize()

    total_images_local = 0
    t0 = time.perf_counter()
    for idx, batch_data in enumerate(val_loader):
        if idx >= measure_steps:
            break
        _image = batch_data["image"]
        cls_fea, _, _ = model(_image, scene_text=None, object_text=None)
        _ = Emo_classifier(cls_fea)
        total_images_local += _image.shape[0]

    accelerator.wait_for_everyone()
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_images_tensor = torch.tensor([total_images_local], device=device, dtype=torch.long)
    total_images_all = accelerator.gather(total_images_tensor).sum().item()

    elapsed_tensor = torch.tensor([elapsed], device=device, dtype=torch.float64)
    elapsed_all = accelerator.gather(elapsed_tensor)
    elapsed_global = elapsed_all.max().item()

    throughput_global = total_images_all / elapsed_global
    latency_ms = (elapsed_global / total_images_all) * 1000.0

    logger.info(
        f"[Speed][ForwardOnly] total_images={total_images_all}, "
        f"elapsed={elapsed_global:.4f}s, "
        f"throughput={throughput_global:.2f} img/s, "
        f"latency={latency_ms:.3f} ms/img"
    )
    return {
        "total_images": total_images_all,
        "elapsed_s": elapsed_global,
        "throughput_img_s": throughput_global,
        "latency_ms_img": latency_ms,
    }
Emotion = ["amusement", "awe", "contentment",
               "excitement",
               "anger",
               "disgust",
               "fear",
               "sadness"
               ]
@torch.no_grad()
def cal_emotion_space(accelerator, val_loader,  model, Emo_classifier, config, logger):
    model.eval()
    Emo_classifier.eval()
    Emo = [0] * 8
    Emo_num = [0] * 8
    # acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter = 0
    acc5_meter = 0
    total_dataset_num = 0

    emotion_features = {emotion: {"cls_fea": [], "scene": [], "object": []} for emotion in Emotion}


    with torch.no_grad():
        logger.info(f"start calculate emotion space")
        for idx, batch_data in enumerate(val_loader):
            
            _image = batch_data['image']#####8x3x224x224
            label_id = batch_data['emotion_label_idx']####8
            label_id = label_id.reshape(-1)
            # _image = _image.view(b, n, t, c, h, w)
            
            cls_fea, patch_token, visual_query_list = model(_image, scene_text=None, object_text = None)
            ########calcaulate_mean std
            # pdb.set_trace()
            for i, emotion_idx in enumerate(label_id):
                emotion_name = Emotion[emotion_idx.item()]
                emotion_features[emotion_name]["cls_fea"].append(cls_fea[i].cpu())  # 存在 CPU 上
                emotion_features[emotion_name]["scene"].append(visual_query_list[0][i].cpu())  # 存在 CPU 上
                emotion_features[emotion_name]["object"].append(visual_query_list[1][i].cpu())
            

        #####计算情感空间
        for emotion, features in emotion_features.items():
            if features["cls_fea"]:  # 确保该情感类别有数据
                # 计算并保存 cls_fea 的均值和方差
                cls_fea_tensor = torch.stack(features["cls_fea"])
                cls_fea_mean = cls_fea_tensor.mean(dim=0)
                cls_fea_std = cls_fea_tensor.std(dim=0)
                torch.save(cls_fea_mean, f"./UniEmo_space/{emotion}_cls_fea_mean.pt")
                torch.save(cls_fea_std, f"./UniEmo_space/{emotion}_cls_fea_std.pt")
                print(f"Saved cls_fea mean and std for {emotion} to UniEmo_space/{emotion}_cls_fea_mean.pt and UniEmo_space/{emotion}_cls_fea_std.pt")

            if features["scene"]:  # 计算并保存 scene 的均值和方差
                scene_tensor = torch.stack(features["scene"])
                scene_mean = scene_tensor.mean(dim=0)
                scene_std = scene_tensor.std(dim=0)
                torch.save(scene_mean, f"./UniEmo_space/{emotion}_scene_mean.pt")
                torch.save(scene_std, f"./UniEmo_space/{emotion}_scene_std.pt")
                print(f"Saved scene mean and std for {emotion} to UniEmo_space/{emotion}_scene_mean.pt and UniEmo_space/{emotion}_scene_std.pt")

            if features["object"]:  # 计算并保存 object 的均值和方差
                object_tensor = torch.stack(features["object"])
                object_mean = object_tensor.mean(dim=0)
                object_std = object_tensor.std(dim=0)
                torch.save(object_mean, f"./UniEmo_space/{emotion}_object_mean.pt")
                torch.save(object_std, f"./UniEmo_space/{emotion}_object_std.pt")
                print(f"Saved object mean and std for {emotion} to UniEmo_space/{emotion}_object_mean.pt and UniEmo_space/{emotion}_object_std.pt")
        ######################### 
    return 1
@torch.no_grad()
def validate(accelerator, val_loader,  model, Emo_classifier, config, logger):
    model.eval()
    Emo_classifier.eval()
    # acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter = 0
    acc5_meter = 0
    total_dataset_num = 0
    with torch.no_grad():
        logger.info(f"start classification evaluation")
        for idx, batch_data in enumerate(val_loader):
            
            _image = batch_data['image']#####8x3x224x224
            label_id = batch_data['emotion_label_idx']####8
            label_id = label_id.reshape(-1)
            # _image = _image.view(b, n, t, c, h, w)
            cls_fea, _, _= model(_image, scene_text=None, object_text = None)
            cls_fea = Emo_classifier(cls_fea)
            # 
            
            # output = output  + logits_list[0] + logits_list[1] + logits_list[2]
            similarity = cls_fea.view(_image.shape[0], -1).softmax(dim=-1)

            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            # pdb.set_trace()
            indices_1 = accelerator.gather_for_metrics(indices_1)
            indices_5 = accelerator.gather_for_metrics(indices_5)
            label_id = accelerator.gather_for_metrics(label_id)
            acc1, acc5 = 0, 0
            total_bs = indices_1.shape[0]
            for i in range(total_bs):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1
            
            acc1_meter += (float(acc1))
            acc5_meter += (float(acc5))
            total_dataset_num += total_bs
            if idx % 50 == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {(acc1_meter/total_dataset_num)*100:.3f}\t'
                )

            # acc1_meter.update(float(acc1) / b * 100, b)
            # acc5_meter.update(float(acc5) / b * 100, b)
        acc1_all = (acc1_meter/total_dataset_num) *100
        acc5_all = (acc5_meter/total_dataset_num) * 100
        # print(1)
    logger.info(f' * Acc@1 {acc1_all:.3f} Acc@5 {acc5_all:.3f}')
    return acc1_all                  
            
if __name__ == "__main__":

    def parameter(file_name):
        with open(file_name, 'r') as file:
            params = yaml.safe_load(file)
        
        args = parse_args(**params)
        params["project_name"] = os.path.basename(__file__)
        params_json = json.dumps(params)
        os.makedirs(f'{params["output_dir"]}',exist_ok=True)
        with open(f'{params["output_dir"]}/params.json', 'w') as f:
            f.write(params_json)
        return args

    # Choose your config file
    file_name = 'config/config_vit.yaml'
    args = parameter(file_name)
    main(args)