import os
import os.path as osp
import json
from tqdm import tqdm
import numpy as np
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

from torchattacks import PGD, TPGD, FAB, PGDL2, APGD
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
        design_details = {"trainer": "MaPLe",
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": cfg.TRAINER.FAP.N_CTX}
    else:
        # Return original CLIP model for generating frozen VL features.
        design_details = {"trainer": "CoOp",
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    if cfg.MODEL.BACKBONE.ROBUST:
        eps = int(cfg.AT.EPS * 255)
        if cfg.MODEL.BACKBONE.FARE:
            ckp_name = 'vitb32_fare'
            ckp_name += f'_eps_{eps}.pt'
        else:
            ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
            ckp_name += f'_eps{eps}.pth.tar'
        ckp = torch.load(osp.join('/backbone', ckp_name))
        if cfg.MODEL.BACKBONE.FARE:
            missing_keys_4_robust_clip, _ = model.visual.load_state_dict(ckp, strict=False)
            print('Load Robust Clip FARE')
        else:
            missing_keys_4_robust_clip, _ = model.visual.load_state_dict(ckp['vision_encoder_state_dict'], strict=False)
        # print('Weights not found for some missing keys_for robust clip: ', missing_keys_4_robust_clip)
            print('Load Robust Clip TeCoA')
    return model


_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
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

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype) #first extend dim then add. Just add positional information to 77 tokens
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FAP.N_CTX
        ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.FAP.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.FAP.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        self.proj.to(dtype)
        self.ctx = nn.Parameter(ctx_vectors)
  
        # compound prompts
        self.compound_prompts_image = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_image:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(768,ctx_dim )
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):


        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) #copy for every input classes

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        text_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            text_deep_prompts.append(layer(self.compound_prompts_image[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), text_deep_prompts ,self.compound_prompts_image  # pass here original, as for visual 768 is required



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
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.normalize = ImageNormalizer(cfg.INPUT.PIXEL_MEAN,
                                         cfg.INPUT.PIXEL_STD).to(device)

    def forward(self, image,return_features=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner() #shared takes from the first layer, and deep takes from second and later layer
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(self.normalize(image).type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)
        if return_features:
            return logits, image_features, text_features
        else:
            return logits
        
    # def inference(self, image):
    #     tokenized_prompts = self.tokenized_prompts
    #     logit_scale = self.logit_scale.exp()

    #     prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner() #shared takes from the first layer, and deep takes from second and later layer
    #     text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
    #     image_features = self.image_encoder(self.normalize(image).type(self.dtype), shared_ctx, deep_compound_prompts_vision)

    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #     logits = logit_scale * image_features @ text_features.t()


    #     return logits, image_features


def calculate_elementwise_kl_div(output1, output2):
    instance_wise_kl=F.kl_div(F.log_softmax(output1, dim=1), F.softmax(output2, dim=1), reduction='none')
    kl_divs=torch.sum(instance_wise_kl,dim=-1)
    return kl_divs

def calculate_cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2)+1


def calculate_adv_loss(output_clean, output_adv, clean_image_features, adv_image_features, adv_term="cos"):
    kl_divs = calculate_elementwise_kl_div(output_adv, output_clean)
    if adv_term=="cos":
        cosine_sims = calculate_cosine_similarity(clean_image_features, adv_image_features)
        loss = torch.mean(kl_divs * cosine_sims)
    else:
        raise NotImplementedError
    return loss

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class FAP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)#.to(self.device)


        if cfg.TRAINER.FAP.PREC == "fp32" or cfg.TRAINER.FAP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
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
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FAP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        optim = self.optim

        eps = self.cfg.AT.EPS
        alpha = self.cfg.AT.ALPHA
        steps = self.cfg.AT.STEPS

        prec = self.cfg.TRAINER.FAP.PREC
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
                output = self.model(adv)
                loss_ce = -F.cross_entropy(output, label)
                loss_ce.backward()
                delta -= alpha * torch.sign(adv.grad)
                delta = torch.clamp(delta, -eps, eps).detach()
                
            adv = torch.clamp(image+delta, 0, 1).detach()

            nat_loss = torch.tensor(0.0).to(self.device)
            adv_loss = torch.tensor(0.0).to(self.device)
            output_adv, adv_image_features, _ = self.model(adv, return_features=True)
            output_clean, clean_image_features, _ = self.model(image, return_features=True)

            nat_loss = F.cross_entropy(output_clean, label)
            adv_loss = calculate_adv_loss(
                output_clean,
                output_adv,
                clean_image_features,
                adv_image_features,
                self.cfg.ATTACK.PGD.ADV_TERM,
            )

            loss = nat_loss + self.cfg.ATTACK.PGD.LAMBDA_1 * adv_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}
        if prec != "amp":
            loss_summary.update({
                "Nat_loss": nat_loss.item(),
                "Adv_loss": adv_loss.item(),
            })

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
            attack_name = self.cfg.TRAINER.DEFEAT.ATK.strip()

            n_classes = len(self.dm.dataset.classnames)
            # Calculate the number of target classes for AA
            # The attack tries to find 9 wrong classes by default.
            # We must limit this to the number of available wrong classes (n_classes - 1).
            # We also ensure it's at least 1.
            n_target_classes = max(1, min(9, n_classes - 1))

            if attack_name == 'aa':
                print("-----AutoAttack-------")
                attack = AutoAttack(model,
                                    norm='Linf',
                                    eps=eps,
                                    version='standard',
                                    verbose=False)
                if (n_classes - 1) < 9:
                    print(f"AutoAttack: Overwriting n_target_classes to {n_target_classes}")
                    attack.apgd_targeted.n_target_classes = n_target_classes
                    attack.fab.n_target_classes = n_target_classes
            elif attack_name == 'pgd':
                print("-----PGD-------")
                attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
            elif attack_name == 'tpgd':
                attack = TPGD(model, eps=eps, alpha=alpha, steps=steps)
            elif attack_name == 'fab':
                print("-----FAB-------")
                attack = FAB(model, norm='Linf', steps=steps, eps=eps, n_restarts=1, alpha_max=alpha, eta=1.05, beta=0.9, verbose=False, seed=0, n_classes=n_classes)
            elif attack_name == 'pgdl2':
                print("-----PGDL2-------")
                attack = PGDL2(model, eps=0.5, alpha=0.5/4.0, steps=steps, random_start=True)
            elif attack_name == 'apgd':
                print("-----APGD-------")
                attack = APGD(model, norm='Linf', eps=eps, steps=steps, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
            elif attack_name == 'cw':
                print("-----CW-------")
            else:
                raise ValueError(f"Unsupported attack: {attack_name}")

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
                if attack_name == 'aa':
                    adv = attack.run_standard_evaluation(input, label, bs=bs)
                elif attack_name in ['pgd', 'tpgd']:
                    adv = attack(input, label)
                elif attack_name == 'cw':
                    adv, _ = pgd(input, label, model, CWLoss, eps, alpha, steps)
                elif attack_name == 'fab':
                    adv = attack(input, label)
                elif attack_name == 'pgdl2':
                    adv = attack(input, label)
                elif attack_name == 'apgd':
                    adv = attack(input, label)
                model.mode = 'classification'
                with torch.no_grad():
                    output_adv = self.model_inference(adv).detach()


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
