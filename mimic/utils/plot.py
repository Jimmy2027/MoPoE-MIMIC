import textwrap
from typing import Optional

import torch
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from mimic.utils import text as text
from mimic import log


def create_fig(fn, img_data, num_img_row, save_figure=False):
    if save_figure:
        save_image(img_data.data.cpu(), fn, nrow=num_img_row)
    grid = make_grid(img_data, nrow=num_img_row)
    return (
        grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to('cpu', torch.uint8)
            .numpy()
    )


def text_to_pil(exp, t, imgsize, font, w=128, h=128, linewidth=8, log_tag: Optional[str] = None):
    blank_img = torch.ones([imgsize[0], w, h])
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    one_hot = len(t.shape) > 2
    sep = ' ' if exp.flags.text_encoding == 'word' else ''
    text_sample = text.tensor_to_text(exp, t, one_hot=one_hot)[0]

    text_sample = sep.join(text_sample).translate({ord('*'): None})
    if log_tag:
        log.info(f"logging to {log_tag}: {text_sample}")
        exp.tb_logger.write_text(log_tag, text_sample)
    lines = textwrap.wrap(''.join(text_sample), width=linewidth)
    y_text = h
    num_lines = len(lines)
    for l, line in enumerate(lines):
        width, height = font.getsize(line)
        draw.text((0, (h / 2) - (num_lines / 2 - l) * height), line, (0, 0, 0), font=font)
        y_text += height
    if imgsize[0] == 3:
        return transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                    Image.ANTIALIAS))
    else:
        return transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                    Image.ANTIALIAS).convert('L'))
