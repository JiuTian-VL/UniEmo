import numpy
import torch.distributed as dist
import torch
import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, accelerator, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if accelerator.is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{accelerator.local_process_index}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
import pdb
def epoch_saving(config, epoch, Emo_classifier, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best, accelerator):
    # model = model.module if hasattr(model, 'module') else model
    # pdb.set_trace()
    save_state = {'model': accelerator.unwrap_model(model).state_dict(),
                  'Emo_classifier': accelerator.unwrap_model(Emo_classifier).state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    os.makedirs(os.path.join(working_dir, 'ckpt'),exist_ok=True)
    save_path = os.path.join(os.path.join(working_dir, 'ckpt'), f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    # torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, Emo_classifier, optimizer, lr_scheduler, logger):
    # start_epoch, max_accuracy=0
    if os.path.isfile(config.model_resume): 
        logger.info(f"==============> Resuming form {config.model_resume}....................")
        checkpoint = torch.load(config.model_resume, map_location='cpu')
        load_state_dict = checkpoint['model']
        Emo_classifier_stat = checkpoint['Emo_classifier']
        msg_emo = Emo_classifier.load_state_dict(Emo_classifier_stat, strict=False)
        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")
        logger.info(f"resume emo_cls model: {msg_emo}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']
            logger.info(f"=> loaded successfully '{config.model_resume}' (epoch {checkpoint['epoch']})")
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         # pdb.set_trace()
        #         state[k] = v.cuda()
        except:
            logger.info("w/o optimizer")
            return 0, 0
        return start_epoch, max_accuracy
     

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.model_resume)))
        return 0, 0