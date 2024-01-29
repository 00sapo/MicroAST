import argparse
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from . import net_microAST as net

THIS_DIR = Path(__file__).parent
WEIGHTS_DIR = THIS_DIR / "weights"


def transform_image(image, size: Optional[int] = None, crop: bool = False):
    """Create a transform function that resizes, crops, reshape, and normalizes the image"""
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform(image).float()


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument("--content", "-c", type=str, help="File path to the content image")
parser.add_argument(
    "--content_dir", type=str, help="Directory path to a batch of content images"
)
parser.add_argument("--style", "-s", type=str, help="File path to the style image")
parser.add_argument(
    "--style_dir", type=str, help="Directory path to a batch of style images"
)
parser.add_argument(
    "--models_dir",
    "-m",
    help="Directory where the model checkpoints are",
    type=str,
    default=WEIGHTS_DIR,
)
parser.add_argument(
    "--content_encoder",
    type=str,
    help="Filename (without directory) of the content encoder checkpoint",
    default="content_encoder_iter_160000.pth.tar",
)
parser.add_argument(
    "--style_encoder",
    type=str,
    default="style_encoder_iter_160000.pth.tar",
    help="Filename (without directory) of the style encoder checkpoint",
)
parser.add_argument(
    "--modulator",
    type=str,
    default="modulator_iter_160000.pth.tar",
    help="Filename (without directory) of the modulator checkpoint",
)
parser.add_argument(
    "--decoder",
    type=str,
    default="decoder_iter_160000.pth.tar",
    help="Filename (without directory) of the decoder checkpoint",
)

# Additional options
parser.add_argument(
    "--content_size",
    type=int,
    default=0,
    help="New (minimum) size for the content image, \
                    keeping the original size if set to 0",
)
parser.add_argument(
    "--style_size",
    type=int,
    default=0,
    help="New (minimum) size for the style image, \
                    keeping the original size if set to 0",
)
parser.add_argument(
    "--crop", action="store_true", help="do center crop to create squared image"
)
parser.add_argument(
    "--save_ext", default=".jpg", help="The extension name of the output image"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
    help="Directory to save the output image(s)",
)
parser.add_argument(
    "--output", "-o", type=str, default="output.jpg", help="Path to save the output"
)
parser.add_argument("--gpu_id", type=int, default=0)

# Advanced options
parser.add_argument(
    "--alpha",
    type=float,
    default=1.0,
    help="The weight that controls the degree of \
                             stylization. Should be between 0 and 1",
)


def stylize(network, content, style, alpha=1.0, _print_elapsed_time=False):
    """
    Stylize the content image with the style image using the model network and the given alpha.

    `content` and `style` should be 4-dim tensors with shape [BATCH, 3, H, W], but [3, H, W] also works for batch size 1.

    The input data tpe should be float. In general, use `microast.transform_image` before of using this function.
    """

    # Add batch dimension if not there
    if content.dim() < 4:
        content = content.unsqueeze(0)
    if style.dim() < 4:
        style = style.unsqueeze(0)

    # Put channel dim before H and W
    if content.shape[3] <= 4:
        content = content.transpose(1, 3)
    if style.shape[3] <= 4:
        style = style.transpose(1, 3)

    torch.cuda.synchronize()
    if _print_elapsed_time:
        tic = time.time()

    with torch.no_grad():
        output = network(content, style, alpha)

    torch.cuda.synchronize()
    if _print_elapsed_time:
        print("Elapsed time: %.4f seconds" % (time.time() - tic))
    return output


def save_output(output_array, output_path):
    """Save the output image; just a wrapper of torchvision.utils.save_image"""
    save_image(output_array, str(output_path))


def load_image(path, resize=None, crop=False):
    """Load the image from the given path, optionally cropping and resizing it. You still need to move it to the proper device"""
    image = Image.open(str(path))
    image = transform_image(image, resize, crop)
    return image


def load_models(content_encoder_path, style_encoder_path, modulator_path, decoder_path):
    """Load the models from the given paths; you still need to move the built network to the proper device"""
    content_encoder = net.Encoder()
    style_encoder = net.Encoder()
    modulator = net.Modulator()
    decoder = net.Decoder()

    content_encoder.eval()
    style_encoder.eval()
    modulator.eval()
    decoder.eval()

    content_encoder.load_state_dict(torch.load(content_encoder_path))
    style_encoder.load_state_dict(torch.load(style_encoder_path))
    modulator.load_state_dict(torch.load(modulator_path))
    decoder.load_state_dict(torch.load(decoder_path))
    network = net.TestNet(content_encoder, style_encoder, modulator, decoder)

    return network


def main(args):
    device = torch.device("cuda:%d" % args.gpu_id)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    assert args.content or args.content_dir
    content_paths = (
        [Path(args.content)] if args.content else list(Path(args.content_dir).glob("*"))
    )

    assert args.style or args.style_dir
    style_paths = (
        [Path(args.style)] if args.style else list(Path(args.style_dir).glob("*"))
    )

    assert args.output or args.output_dir

    network = load_models(
        Path(args.models_dir) / args.content_encoder,
        Path(args.models_dir) / args.style_encoder,
        Path(args.models_dir) / args.modulator,
        Path(args.models_dir) / args.decoder,
    )
    network.to(device)

    for content_path in content_paths:
        for style_path in style_paths:
            try:
                content = load_image(content_path).to(device)
                style = load_image(style_path).to(device)
                output_array = stylize(
                    network, content, style, args.alpha, _print_elapsed_time=True
                )
                if args.output:
                    output_path = Path(args.output)
                else:
                    output_path = output_dir / "{:s}_stylized_{:s}{:s}".format(
                        content_path.stem, style_path.stem, args.save_ext
                    )
                save_output(output_array, output_path)
            except Exception as e:
                print(f"Error processing {content_path} and {style_path}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
