import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist
import torch

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    # pdb.set_trace()
    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]


def fix_text(model):
    for name, param in model.named_parameters():
        if "visual." in name or "mit" in name or "prompts" in name:
            continue
        else:
            param.requires_grad=False
import pdb
def build_optimizer(config, model, Emo_classifier, Object_classifier, scene_classifier):
    
    model = model.module if hasattr(model, 'module') else model
    
    # fix text
    if config.fix_text:
        fix_text(model)
    
    # set decay and lr
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    ##lr=8.e-6, 
    clip_parameters = set_weight_decay(model, skip, skip_keywords, 
        weight_decay=config.weight_decay, lr=8.e-6, 
        have=(), not_have=("visual_query_scene", "visual_query_object", "logit_scale_object", "visual_query_interact")
    )
    
    # pdb.set_trace()
    # msg_parameters = set_weight_decay(model, skip, skip_keywords,
    #     weight_decay=config.weight_decay, lr=config.learning_rate*10, 
    #     have=("prompts",), not_have=()
    # )


    # learning_rate_mit = 4e-4
    # mit_parameters = set_weight_decay(model, skip, skip_keywords,
    #     weight_decay=3*config.weight_decay, lr=learning_rate_mit, 
    #     have=("mit",), not_have=()
    # )
    learning_rate_prompts = 5.e-4  ##0.0001
    weight_decay_prompts = 1.e-2
    prompts_parameters = set_weight_decay(model, skip, skip_keywords, 
        weight_decay=weight_decay_prompts, lr = learning_rate_prompts, 
        have=("visual_query_scene", "visual_query_object", "logit_scale_object", "visual_query_interact"), not_have=()
    )
    
    # optimizer = optim.AdamW(clip_parameters + mit_parameters + prompts_parameters + msg_parameters,
    #                     betas=(0.9, 0.98), eps=1e-8,)
    Emo_classifier_parameters = set_weight_decay(Emo_classifier, skip, skip_keywords, weight_decay = 0.01, lr=config.learning_rate)
    Object_classifier_parameters = set_weight_decay(Object_classifier, skip, skip_keywords, weight_decay = 0.01, lr=config.learning_rate)
    scene_classifier_parameters = set_weight_decay(scene_classifier, skip, skip_keywords, weight_decay = 0.01, lr=config.learning_rate)
    # optimizer = optim.AdamW(Emo_classifier_parameters + clip_parameters,
    #                     betas=(0.9, 0.999), eps=1e-8)
    # optimizer = optim.AdamW(Emo_classifier_parameters + clip_parameters + prompts_parameters,
    #                     betas=(0.9, 0.999), eps=1e-8)
    optimizer = optim.AdamW(Emo_classifier_parameters + prompts_parameters + clip_parameters + Object_classifier_parameters + scene_classifier_parameters,
                        betas=(0.9, 0.999), eps=1e-8)
    # pdb.set_trace()
    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    # pdb.set_trace()
    num_steps = int(config.num_train_epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)
    
    # lr_scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=num_steps,
    #     lr_min=8.e-6/100,
    #     warmup_lr_init=0,
    #     warmup_t=warmup_steps,
    #     cycle_limit=1,
    #     t_in_epochs=False,
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max= 32
    )
    

    return lr_scheduler