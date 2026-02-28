# DEFEAT: Discrete Latent Feature based Adversarial Training [ICLR 2026, Poster]
> [**Discrete Latent Features Ablate Adversarial Attack: A Robust Prompt Tuning Framework for VLMs**](https://openreview.net/forum?id=lZgORA63ew)<br>
> Yang Chen, Yanbin Wei, James Kwok, Yu Zhang

## Preparation
### Code

This code is built on top of [CoOp](https://github.com/KaiyangZhou/CoOp) which extensively uses the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl`. After that, run `pip install -r requirements.txt` to install a few more packages (this should be done when `dassl` is activated). Then, you are ready to go.

### Data

Follow [DATASETS.md](DATASETS.md) to install the datasets. After successfully set up the datasets, the data directory variables, `DATA`, in each script under `/DEFEAT_code/scripts` MUST be updated with the root path of those datasets.  

### Pre-trained Robust CLIP Backbone

We adopt as backbone the pre-trained adversarially-robust CLIP models from [TeCoA](https://github.com/cvlab-columbia/ZSRobust4FoundationModel). The used pre-trained weights are provided [here](https://emckclac-my.sharepoint.com/:f:/g/personal/k19010102_kcl_ac_uk/EmZ98eFLv71FqQyqPLvWNTkBYNAKPyx_wYEDjNPx7smKCA?e=8AB51S). (provided by [APT (CVPR 2024)](https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning)) To run the code, the pre-trained backbone models should be placed under the directory `/backbone`.  The code currently supports one architecture: ViT-B/32 (named `vitb32`). Taking an example of tuning ViT-B/32 at epsilon=4/255, the path to the checkpoint is `/DEFEAT_code/backbone/vitb32_eps4.pth.tar`. Note that our code can be easily adapted to load other pre-trained models as backbone.

## Adversarial few-shot classification

The configureation is provided in config file at `/DEFEAT_code/configs/trainers/DEFEAT/vit_b32.yaml`:

Run the command below to train and test DEFEAT on Caltech101.
```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# Adversarial training
#DEFEAT_code\scripts
bash train_vae.sh caltech101 vit_b32 16 16 4 2.67 3 10.0 20.0 0.5 0.1 0.5
```

The above arguments correspond to in order:

1. dataset.
2. training configuration identifier.
3. the number of context vectors, `M`
4. the number of shots, `N`
5. the training perturbation budget, `\epsilon`
6. the step size of training adversary
7. the number of steps for training adversary
8. `\lambda`
9. `\mu`
10. `\alpha`
11. `\beta`
12. weight for logits fusion.


```bash
# Test accuracy and robustness
bash test_vae.sh caltech101 vit_b32 200 16 16 4 2.67 3 10.0 20.0 0.5 0.1 0.5
```
1. dataset.
2. training configuration identifier.
3. load epoch
4. ...


Run the command below to train and test APT on Caltech101.
```bash
# Adversarial training
bash apt_train.sh caltech101 vit_b32 16 16 4 2.67 3
# Test accuracy and robustness
bash apt_test.sh caltech101 vit_b32 200 16 16 4 2.67 3
```
## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation
If you find our paper of codebase useful, please consider citing us as:
```bibtex
  @inproceedings{chen2026discrete,
    title={Discrete Latent Features Ablate Adversarial Attack: A Robust Prompt Tuning Framework for VLMs},
    author={Yang Chen and Yanbin Wei and James Kwok and Yu Zhang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=lZgORA63ew}
  }

```

