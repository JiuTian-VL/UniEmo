import sys
import torch
import os
from model import *
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
from tqdm.auto import tqdm
import argparse
import random
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import torch.nn.functional as F
import numpy as np
import sys
from model import *
sys.path.append("/data/Users/zyj/EmoGen")
# print(sys.path)
from train_vit_class import Emotion_clip
from random import choice
from main import *
import pdb

@torch.no_grad()
def count_relate(img, model, processor):
    with open(f'dataset_balance/all_attribute_object_scene.pkl', 'rb') as f:
        attribute_total = pickle.load(f)
    data_pro = processor(images=img, text=attribute_total, return_tensors="pt", padding=True).to(model.device)
    data_pro = model(**data_pro)
    score = data_pro.logits_per_image.squeeze(0)
    indice = torch.argmax(score, dim=0)
    relate_semantic = attribute_total[indice.item()]
    relate_score = score[indice.item()].item()

    return relate_semantic, relate_score

class Emotion_classifier(nn.Module):
    def __init__(self,vision_width ):
        super(Emotion_classifier, self).__init__()
        self.fc0 = nn.Linear(vision_width, 768)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(768, 8)


    def forward(self, x):
        x = self.fc0(x)
        x = self.drop_out(self.relu(x))
        x = self.fc1(x)

        return x
@torch.no_grad()
def inference(arg, emotion):
    working_path = arg.working_path
    device = torch.device(arg.device if torch.cuda.is_available() else "cpu")  # TODO
    placeholder_token = f"<{emotion}>"
    # placeholder_token = "amusement</w>"
    save_dir = f"{working_path}/img/{emotion}"
    prompt = [placeholder_token]
    
    
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 50  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    # generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
    batch_size = len(prompt)
    num_picture = arg.num_picture
    repo_id = arg.repo_id
    model = CLIPModel.from_pretrained("model/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("model/clip-vit-large-patch14")
    vae = AutoencoderKL.from_pretrained('model/stable-diffusion-v1-5', subfolder="vae")
    vae.to(device)

    tokenizer = CLIPTokenizer.from_pretrained('model/stable-diffusion-v1-5', subfolder="tokenizer")

    class image_encoder(nn.Module):
        def __init__(self):
            super(image_encoder, self).__init__()
            self.resnet = BackBone()
            self.resnet = torch.nn.Sequential(*list(self.resnet.children())[1:-1])

        def forward(self, x):
            out = self.resnet(x)
            return out

    model_dict = {
        "FC": lambda args: FC(),
        "MLP": lambda args: MLP(args.num_fc_layers, args.need_ReLU, args.need_LN, args.need_Dropout),
        "SimpleMLP": lambda args: SimpleMLP(args.need_ReLU, args.need_Dropout),
    }
    mapper = model_dict[arg.mapper_name](arg)
    state = torch.load(os.path.join(working_path, "mapper.pth"))
    mapper.load_state_dict(state)
    mapper.eval()
    mapper.to(device)

    # CTI = MultitokenIntegrationTransformer(T = 3, embed_dim=1024, layers = 1)
    # state = torch.load(os.path.join(working_path, "CTI.pth"))
    # CTI.load_state_dict(state)
    # CTI.eval()
    # CTI.to(device)

    mapper_scene = model_dict[arg.mapper_name](arg)
    state = torch.load(os.path.join(working_path, "mapper_scene.pth"))
    mapper_scene.load_state_dict(state)
    mapper_scene.eval()
    mapper_scene.to(device)

    mapper_object = model_dict[arg.mapper_name](arg)
    state = torch.load(os.path.join(working_path, "mapper_object.pth"))
    mapper_object.load_state_dict(state)
    mapper_object.eval()
    mapper_object.to(device)


    # pdb.set_trace()
    # sce_coefficient = scene_alpha()
    # obj_coefficient = object_alpha()
    # state_sce = torch.load(os.path.join(working_path, "sce_coefficient.pth"))
    # state_obj = torch.load(os.path.join(working_path, "obj_coefficient.pth"))
    # sce_coefficient.load_state_dict(state_sce)
    # obj_coefficient.load_state_dict(state_obj)
    # sce_coefficient.to(device)
    # obj_coefficient.to(device)

    
    
    # pdb.set_trace()

    e_mean = torch.load(f"emo_space/{emotion}_mean_v2.pt")
    e_std = torch.load(f"emo_space/{emotion}_std_v2.pt")
    # 
    normal = torch.distributions.Normal(e_mean, e_std)
    encoder, _ , vision_width= Emotion_clip.load(model_path= None, name = 'ViT-L/14',
                         device="cpu", jit=False, query_len=32, query_layer=31, logger = None
                        )
    # pdb.set_trace()
    checkpoint = torch.load('runs/train_vit/best.pth', map_location='cpu')
    load_state_dict = checkpoint['model']
    msg = encoder.load_state_dict(load_state_dict, strict=False)

    Emo_classifier = Emotion_classifier(vision_width)
    Emo_classifier_stat = checkpoint['Emo_classifier']
    msg_emo = Emo_classifier.load_state_dict(Emo_classifier_stat, strict=False)



    def save_pic(emotion, img, path, semantic, score):
        if path is not None:
            os.makedirs(path, exist_ok=True)
        try:
            files = sorted([x for x in os.listdir(path) if x.endswith("_v2.jpg")], key=lambda x: int(x.split("_")[1]))
            num = int(files[-1].split("_")[1])
            img.save(f"{path}/{emotion}_{num + 1}_{semantic}_{score:.2f}_v2.jpg")
        except:
            img.save(f"{path}/{emotion}_0_{semantic}_{score:.2f}_v2.jpg")

    text_encoder = CLIPTextModel.from_pretrained('model/stable-diffusion-v1-5', subfolder="text_encoder")
    text_encoder.to(device)
    
    num_added_tokens = tokenizer.add_tokens(prompt)
    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode("cat", add_special_tokens=False)
    # pdb.set_trace()
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    placeholder_token_ids = tokenizer.convert_tokens_to_ids([placeholder_token])
    # pdb.set_trace()
    # pdb.set_trace()
    token_id = placeholder_token_ids[0]
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data

    text = prompt
    templates = [
        "{} bag",
    ]
  
    # templates = [
    #   "{} bag", "{} cup", "{} room", "{} street",
    # ]
    if not arg.use_prompt:
        text = random.choice(templates).format(prompt[0])
        # text = random.choice(templates)
    text_input = tokenizer(
        text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
   
    # text_input2 = tokenizer(
    #     placeholder_token, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    # )
    # with torch.no_grad():
    #     text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    # print(tokenizer.get_vocab())
    # print(text_embeddings.shape)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    unet = UNet2DConditionModel.from_pretrained(
        'model/stable-diffusion-v1-5', subfolder="unet"
    )
    unet.to(device)
    encoder.to(device)
    encoder.eval()
    Emo_classifier.to(device)

    scheduler = UniPCMultistepScheduler.from_pretrained('model/stable-diffusion-v1-5', subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    
    
    
    tfm = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
    label2idx = {
      "amusement": 0,
      "awe": 1,
      "contentment": 2,
      "excitement": 3,
      "anger": 4,
      "disgust": 5,
      "fear": 6,
      "sadness": 7
    }
    root_dir = f"/data/Users/zyj/EmoGen/data/EmoSet/image/{emotion}"
    root_list = os.listdir(root_dir)
    total_attr, _ = read_attr()
    total_attr_embed = generate_attr(processor, model, total_attr).detach()
    attr_coefficient = get_coefficient()
    # 
    for _ in range(num_picture):
        # e_vec = normal.sample((1,)).to(device)
       
        index = np.random.randint(0, len(root_list), size=1)
        path = os.path.join(root_dir, root_list[index[0]])
        image = Image.open(path)
        img_feat = tfm(image.copy())
        img_feat = img_feat.unsqueeze(0).to(device)

  
        emo_v, patch_token, visual_query_list = encoder(img_feat, scene_text=None, object_text=None)
        scene_query, object_query = visual_query_list[0], visual_query_list[1]
    
        pred_emd_emo = mapper(emo_v)
      
        pred_emd_scene  = mapper_scene(scene_query)
        pred_emd_object  = mapper_object(object_query)

        score_scene = F.cosine_similarity(total_attr_embed, pred_emd_scene).unsqueeze(0)
        score_object = F.cosine_similarity(total_attr_embed, pred_emd_object).unsqueeze(0)

        index_scene = torch.argmax(score_scene, dim = 1)
        index_object = torch.argmax(score_object, dim = 1)
        attr_rate_scene = 0
        attr_rate_object = 0
        if total_attr[index_scene] in attr_coefficient.keys():
            attr_rate_scene = attr_coefficient[total_attr[index_scene]][label2idx[emotion]].item()
        if total_attr[index_object] in  attr_coefficient.keys():
            attr_rate_object = attr_coefficient[total_attr[index_object]][label2idx[emotion]].item()
        
        
        
       
        # pred_emd = pred_emd_emo + attr_rate_scene * pred_emd_scene + attr_rate_object * pred_emd_object 
        pred_emd = pred_emd_emo + attr_rate_scene * pred_emd_scene + attr_rate_object * pred_emd_object 

        token_embeds[token_id] = pred_emd
      
   
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            # generator=generator,
        )
        latents = latents.to(device)
        latents = latents * scheduler.init_noise_sigma

        with torch.no_grad():
            # pdb.set_trace()
            hiddenstate = text_encoder(text_input.input_ids.to(device))
            text_embeddings = hiddenstate[0]
            # 
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            # pdb.set_trace()
            image = vae.decode(latents).sample
        # pdb.set_trace()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        semantic, score = count_relate(pil_images[0], model, processor)
        save_pic(emotion, pil_images[0], save_dir, semantic, score)


@torch.no_grad()
def emo_cls(cur_dir, device, weight):
    classifier = emo_classifier().to(device)
    state = torch.load(weight, map_location=device)
    classifier.load_state_dict(state)
    classifier.eval()

    CLIPmodel = CLIPModel.from_pretrained("model/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("model/clip-vit-large-patch14")

    class EmoDataset(Dataset):
        def __init__(self, data_root, processor):
            self.emotion_list_8 = {"amusement": 0,
                                   "awe": 1,
                                   "contentment": 2,
                                   "excitement": 3,
                                   "anger": 4,
                                   "disgust": 5,
                                   "fear": 6,
                                   "sadness": 7}
            self.emotion_list_2 = {"amusement": 0,
                                   "awe": 0,
                                   "contentment": 0,
                                   "excitement": 0,
                                   "anger": 1,
                                   "disgust": 1,
                                   "fear": 1,
                                   "sadness": 1
                                   }
            self.tfm = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            self.image_paths = []
            self.processor = processor
            self.data_root = data_root
            for root, _, file_path in os.walk(self.data_root):
                for file in file_path:
                    if file.endswith("jpg"):
                        self.image_paths.append(os.path.join(root, file))
            self._length = len(self.image_paths)

        def __len__(self):
            return self._length

        def __getitem__(self, i):
            path = self.image_paths[i]
            example = {}
            image = Image.open(path).convert('RGB')
            data = self.processor(images=image, return_tensors="pt", padding=True)
            data['pixel_values'] = data['pixel_values'].squeeze(0)
            example['image'] = data
            # data = self.model.get_image_features(**data)
            example['emotion_8'] = self.emotion_list_8[path.split('/')[-2]]
            example['emotion_2'] = self.emotion_list_2[path.split('/')[-2]]
            return example

    val_dataset = EmoDataset(cur_dir, processor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    picture_num = len(val_dataset)
    val_loader = tqdm(val_loader, file=sys.stdout)
    score_8 = 0
    score_2 = 0
    acc_num_2 = 0
    acc_num_8 = 0

    def eightemotion(Emo, Emo_num, Emo_score, pre, label, correct):

        for i in range(label.shape[0]):
            emo_label = label[i][0].item()
            Emo[emo_label] += correct[i].item()
            Emo_num[emo_label] += 1
            Emo_score[emo_label] += pre[i][emo_label]
        return Emo, Emo_num, Emo_score

    Emo = [0] * 8
    Emo_num = [0] * 8
    Emo_score = [0] * 8
    Emotion = ["amusement", "awe", "contentment",
               "excitement",
               "anger",
               "disgust",
               "fear",
               "sadness"
               ]
    for step, data in enumerate(val_loader):
        images = data['image'].to(device)
        clip = CLIPmodel.get_image_features(**images)
        pred = classifier(clip.to(device))
        labels_8 = data['emotion_8'].to(device).unsqueeze(1)
        labels_2 = data['emotion_2'].to(device).unsqueeze(1)
        pred_emotion_8 = torch.argmax(pred, dim=1, keepdim=True)
        p_8 = F.softmax(pred)
        p_2 = p_8.reshape((p_8.shape[0], 2, 4))
        p_2 = torch.sum(p_2, dim=2)
        p_2 = p_2.reshape((p_8.shape[0], -1))

        pred_emotion_2 = torch.argmax(p_2, dim=1, keepdim=True)

        pred_score_8 = torch.gather(p_8, dim=1, index=labels_8)
        pred_score_2 = torch.gather(p_2, dim=1, index=labels_2)

        acc_num_2 += (labels_2 == pred_emotion_2).sum().item()
        score_2 += torch.sum(pred_score_2).item()
        acc_num_8 += (labels_8 == pred_emotion_8).sum().item()
        score_8 += torch.sum(pred_score_8).item()
        eightemotion(Emo, Emo_num, Emo_score, p_8, labels_8, (labels_8 == pred_emotion_8))
    acc_8 = (acc_num_8 / picture_num) * 100
    total_score_8 = score_8 / picture_num
    acc_2 = (acc_num_2 / picture_num) * 100
    total_score_2 = score_2 / picture_num
    with open(os.path.join(cur_dir, 'evaluation.txt'), "a") as f:
        f.write(f'emo_score (8 class): {total_score_8:.2f}' + '\n')
        f.write(f'accuracy (8 class): {acc_8:.2f}%' + '\n')
        f.write(f'emo_score (2 class): {total_score_2:.2f}' + '\n')
        f.write(f'accuracy (2 class): {acc_2:.2f}%' + '\n')
        pdb.set_trace()
        for i in range(8):
            tmp = Emo[i] / Emo_num[i] * 100
            f.write(f'{Emotion[i]} accuracy:{tmp:.2f}% score:{(Emo_score[i]/Emo_num[i]):.2f} \n')


def generate(cur_dir, device,model, num_fc_layers=1, need_LN=False, need_ReLU=False, need_Dropout=False, use_prompt=False):
    emotion_list = ["amusement", "excitement", "awe", "contentment", "fear", "disgust", "anger", "sadness"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_picture', type=int, default=1000)
    parser.add_argument('--repo_id', type=str, default="stable-diffusion-v1-5/")
    parser.add_argument('--device', type=str, default=device)
    #####################################################################################
    parser.add_argument('--working_path', type=str, default=cur_dir)
    parser.add_argument('--mapper_name', type=str, default=model)
    parser.add_argument('--num_fc_layers', type=int, default=num_fc_layers)
    parser.add_argument("--need_LN", type=bool, default=need_LN)
    parser.add_argument("--need_ReLU", type=bool, default=need_ReLU)
    parser.add_argument("--need_Dropout", type=bool, default=need_Dropout)
    parser.add_argument("--use_prompt", type=bool, default=use_prompt)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--emotion_idx", type=int, default=None, help="A seed for reproducible training.")
    opt = parser.parse_args()
    emotion = emotion_list[opt.emotion_idx]
    # for emo in emotion_list:
    inference(opt, emotion)

import pdb
if __name__ == "__main__":
    import json

    file = [
        "runs/transfer",
    ]
    # choose which epoch do you want to generate

    epochs = [1]
    device = "cuda:0"

    # emotion_classifier's weight
    weight = "weights/Clip_emotion_classifier/time_2023-11-12_03-29-best.pth"

    # 从 JSON 文件中读取参数
    # pdb.set_trace()
    for f in file:
        with open(f'{f}/params.json', 'r') as f:
            params_json = f.read()

        params = json.loads(params_json)
        globals().update(params)
        origin = output_dir
        try:
            for i in epochs:
                output_dir = os.path.join(origin, str(i))
                # use_prompt = True
                # generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout, use_prompt)
                generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout)
                # emo_cls(output_dir, device, weight)
        except:
            output_dir = origin
            # use_prompt = True
            # generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout, use_prompt)
            generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout)
            # emo_cls(output_dir, device, weight)