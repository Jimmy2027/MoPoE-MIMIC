from torchvision import transforms
import torch
import PIL.Image as Image


def get_transform_img(args):
    if args.img_clf_type == 'cheXnet' or args.feature_extractor_img == 'densenet':
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        crops_transform = get_crops_transform(args)
        transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(args.img_size),
            crops_transform,
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    else:
        transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(args.img_size, args.img_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    return transform_img


def get_crops_transform(args):
    if args.n_crops == 10:
        crops_transform = transforms.TenCrop(224)
    elif args.n_crops == 5:
        crops_transform = transforms.FiveCrop(224)
    else:
        # ugly...
        crops_transform = transforms.Lambda(lambda x: x)
    return crops_transform
