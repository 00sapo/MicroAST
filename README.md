# MicroAST

This is just a fork of the official code from the paper ["MicroAST: Towards Super-Fast Ultra-Resolution Arbitrary Style Transfer"](https://arxiv.org/pdf/2211.15313.pdf)

See original code: https://github.com/EndyWon/MicroAST

This fork helps using it in other projects.

## Installation

`pip install git+https://github.com/00sapo/MicroAST.it`

## Usage

### CLI

You need to download the models from the original repo.

- Help: `microast -h`
- Stylize a single image: `microast -s <style_image> -c <content_image> -o <output_image> -m <model_dir>`
- Stylize all images in a directory: `microast --style_dir <style_dir> --content_dir <content_dir> --output_dir <output_dir> -model_dir <model_dir>`

### API

```python
from microast import load_image, load_models, save_output, stylize, transform_image

# Input paths
style_path = 'style.png'
content_path = 'content.png'
output_path = 'output.png'
content_encoder_path = 'content_encoder.pth'
style_encoder_path = 'style_encoder.pth'
modulator_path = 'modulator.pth'
decoder_path = 'decoder.pth'
alpha = 1.0
device = 'cuda:0'

already_loaded_image = np.array(Image.open(content_path))

# The code
# N.B. The handling of devices is all up to you
network = load_models(content_encoder_path, style_encoder_path, modulator_path, decoder_path).to(device) # a model
content = load_image(content_path).to(device) # a tensor
# or
content = transform_image(already_loaded_image).to(device) # a tensor

style = load_image(style_path).to(device) # a tensor
output_array = stylize(network, content, style, alpha).cpu()
# output_array is a tensor of shape (1, 3, H, W) and dtype float32
save_output(output_array, output_path)
```

```

## Citation:

If you find the ideas and codes useful for your research, please cite the paper:

```

@inproceedings{wang2023microast,
title={MicroAST: Towards Super-Fast Ultra-Resolution Arbitrary Style Transfer},
author={Wang, Zhizhong and Zhao, Lei and Zuo, Zhiwen and Li, Ailin and Chen, Haibo and Xing, Wei and Lu, Dongming},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
year={2023}
}

```

## Acknowledgement:

We refer to some codes and ideas from [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [DIN](https://ojs.aaai.org/index.php/AAAI/article/view/5862). Great thanks to them!
```
