from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import PIL.Image as Image


def get_transform_img(args: any):
    if args.img_clf_type == 'cheXnet' or args.feature_extractor_img == 'densenet':
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        crops_transform = get_crops_transform(args)
        transformation_list = [
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(args.img_size)]
        if args.n_crops not in [10, 5]:
            transformation_list.extend([transforms.ToTensor(), normalize])
        else:
            if args.n_crops == 10:
                transformation_list.append(transforms.TenCrop(224))
            elif args.n_crops == 5:
                crops_transform = transforms.FiveCrop(224)

            transformation_list.extend([crops_transform, transforms.Lambda(
                lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda(
                                            lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        transform_img = transforms.Compose(transformation_list)

    else:
        transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(args.img_size, args.img_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    return transform_img


def get_crops_transform(args) -> transforms:
    if args.n_crops == 10:
        crops_transform = transforms.TenCrop(224)
    elif args.n_crops == 5:
        crops_transform = transforms.FiveCrop(224)
    else:
        # ugly...
        crops_transform = transforms.Lambda(lambda x: x)
    return crops_transform


def get_data_loaders(args, dataset):
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        num_workers = args.dataloader_workers // args.world_size
    else:
        sampler = None
        num_workers = args.dataloader_workers
    d_loader = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=(sampler is None),
                          num_workers=num_workers, drop_last=True,
                          sampler=sampler, pin_memory=True)

    return sampler, d_loader


def samplers_set_epoch(args, train_sampler, test_sampler, epoch: int) -> None:
    if args.distributed:
        for sampler in [train_sampler, test_sampler]:
            sampler.set_epoch(epoch)
