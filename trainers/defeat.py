import os
import os.path as osp
import numpy as np
import json
from tqdm import tqdm
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torchattacks import PGD, TPGD
from autoattack import AutoAttack
from torch.autograd import grad, Variable
import pdb

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'DEFEAT',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'DEFEAT',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    if cfg.MODEL.BACKBONE.ROBUST:
        ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
        eps = int(cfg.AT.EPS * 255)
        ckp_name += f'_eps{eps}.pth.tar'
        ckp = torch.load(osp.join('/backbone', ckp_name))
        missing_keys_4_robust_clip, _ = model.visual.load_state_dict(ckp['vision_encoder_state_dict'], strict=False)
        # print('Weights not found for some missing keys_for robust clip: ', missing_keys_4_robust_clip)
        print('Load Robust Clip TeCoA')
    return model

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        ##### original clip text features ############################################
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(cfg,True)
        clip_model_.cuda()
        
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp_hard = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp_hard.format(c.replace("_", " ")) for c in classnames] #hand-cafted prompt
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_]) 
        prompts_ = prompts_.cuda()
        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_) 
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features #originial text_features
        #############################################################################

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts

class LinearProjection(nn.Module):
    def __init__(self, dim):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.squeeze(-1)
        return x
    
class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp

class VQVAE_NET(nn.Module):

    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim))
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            nn.ConvTranspose2d(dim, input_dim, 3, 1, 1))
        self.n_downsample = 2

    def forward(self, x):
        # encode
        x = x.permute(0, 2, 1)
        x = x.view(-1, 512, 7, 7)
        ze = self.encoder(x)
        
        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data #[n_embedding, dim]
        N, C, H, W = ze.shape
        K, _ = embedding.shape #[K=n_embedding]
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()
        
        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq


class ImageNormalizer(nn.Module):

    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device='cuda'):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.cls_projection = LinearProjection(49).to(clip_model.dtype)
        self.ori_embedding = self.prompt_learner.text_features
        self.vqvae = VQVAE_NET(512, 256, 512).to(clip_model.dtype) # input_dim, dim, n_embedding
        
        self.normalize = ImageNormalizer(cfg.INPUT.PIXEL_MEAN,
                                         cfg.INPUT.PIXEL_STD).to(device)
        self.alpha = cfg.TRAINER.COOP.ALPHA 
    def forward(self, image, attack=False):
        image_features, local_ima_fea = self.image_encoder(self.normalize(image).type(self.dtype))

        x_hat, ze, zq = self.vqvae(local_ima_fea)
        ima_fea_vae = x_hat.view(x_hat.shape[0], 512, -1)
        ima_fea_vae = ima_fea_vae.permute(0, 2, 1)

        local_ima_fea__ = self.cls_projection(ima_fea_vae)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_ima_fea__ = local_ima_fea__ / local_ima_fea__.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_ima_fea__ @ text_features.t()
        logits_mix = self.alpha * logits + (1-self.alpha) *logits_local

        if self.prompt_learner.training:
            return logits_mix, image_features, local_ima_fea__, text_features, self.ori_embedding,  x_hat, ze, zq, local_ima_fea
        else:
            return logits_mix


@TRAINER_REGISTRY.register()
class DEFEAT(TrainerX):
    def check_cfg(self, cfg):
        print(cfg.TRAINER.COOP.PREC)
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)


        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                elif "cls_projection" in name:
                    param.requires_grad_(True)
                elif "vqvae" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        optim = self.optim

        eps=self.cfg.AT.EPS
        alpha=self.cfg.AT.ALPHA
        steps=self.cfg.AT.STEPS
        attack = PGD(self.model,
                     eps=self.cfg.AT.EPS,
                     alpha=self.cfg.AT.ALPHA,
                     steps=self.cfg.AT.STEPS)


        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:

            delta = torch.zeros_like(image).uniform_(-eps, eps)
            for _ in range(steps):
                adv = torch.clamp(image+delta, 0, 1).requires_grad_(True)
                output, _, _, _, _, _, _, _, _ = self.model(adv, attack=True)
                loss_ce = -F.cross_entropy(output, label)
                loss_ce.backward()
                delta -= alpha * torch.sign(adv.grad)
                delta = torch.clamp(delta, -eps, eps).detach()
                
            adv = torch.clamp(image+delta, 0, 1).detach()
            output, image_features, local_ima_fea__, text_features, ori_embedding, x_hat, ze, zq, x = self.model(adv)
            loss_ce = F.cross_entropy(output, label)
            loss_scl_image = F.l1_loss(image_features, local_ima_fea__.cuda(),
                                reduction='mean')
            loss_scl_text = F.l1_loss(text_features, ori_embedding.cuda(),
                                      reduction='mean')

            mse_loss = nn.MSELoss()
            x = x.permute(0,2,1)
            x = x.view(x_hat.shape)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss_vae = self.cfg.TRAINER.COOP.W3 * l_reconstruct + self.cfg.TRAINER.COOP.W4 * l_embedding + 0.25 * self.cfg.TRAINER.COOP.W4 * l_commitment
            loss_scl = self.cfg.TRAINER.COOP.W1 * loss_scl_image + self.cfg.TRAINER.COOP.W2 * loss_scl_text
            loss = loss_ce + loss_scl + loss_vae

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss_ce": loss_ce.item(),
            "loss_scl": loss_scl.item(),
            "l_reconstruct": l_reconstruct.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if self.cfg.TRAINER.DEFEAT.ATK_TEST:
            self.evaluator_adv = build_evaluator(self.cfg, lab2cname=self.lab2cname)
            self.evaluator_adv.reset()
            torch.cuda.empty_cache()
            model = self.model
            eps = self.cfg.AT.EPS
            alpha = eps / 4.0
            steps = 100
            if self.cfg.TRAINER.DEFEAT.ATK == 'aa':
                attack = AutoAttack(model,
                                    norm='Linf',
                                    eps=eps,
                                    version='standard',
                                    verbose=False)
            elif self.cfg.TRAINER.DEFEAT.ATK == 'pgd':
                attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
            elif self.cfg.TRAINER.DEFEAT.ATK == 'tpgd':
                attack = TPGD(model, eps=eps, alpha=alpha, steps=steps)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                output = self.model_inference(input).detach()
            self.evaluator.process(output, label)
            torch.cuda.empty_cache()
            if self.cfg.TRAINER.DEFEAT.ATK_TEST:
                bs = input.size(0)
                model.mode = 'attack'
                if self.cfg.TRAINER.DEFEAT.ATK == 'aa':
                    model = model.float()
                    adv = attack.run_standard_evaluation(input, label, bs=bs)
                elif self.cfg.TRAINER.DEFEAT.ATK in ['pgd', 'tpgd']:
                    adv = attack(input, label)
                else:
                    adv, _ = pgd(input, label, model, CWLoss, eps, alpha, steps)
                model.mode = 'classification'
                with torch.no_grad():
                    output_adv  = self.model_inference(adv).detach()
                self.evaluator_adv.process(output_adv, label)
        results = self.evaluator.evaluate()
        results_adv = {}
        if self.cfg.TRAINER.DEFEAT.ATK_TEST:
            results_adv = self.evaluator_adv.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        if self.cfg.TRAINER.DEFEAT.ATK_TEST:
            for k, v in results_adv.items():
                tag = f"{split}/{k}_adv"
                self.write_scalar(tag, v, self.epoch)
        if self.cfg.TRAINER.DEFEAT.ATK_TEST:
            with open(osp.join(self.output_dir, 'results.json'), 'w') as fp:
                json.dump(results, fp) 
                fp.write('\n')
                json.dump(results_adv, fp) 
                fp.write('\n')
            return list(results.values())[0], list(results_adv.values())[0]
        else:
            return list(results.values())[0]

    
def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else torch.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*torch.sign(ig)
    else:
        pert += eps_step*torch.sign(ig)
    pert.clamp_(-eps, eps)
    adv = torch.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert