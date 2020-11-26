from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import PIL.Image as Image


class CustomTransforms:
    def to_RGB(self, x: Image) -> Image:
        return x.convert('RGB')

    def crops_to_tensor(self, crops: torch.Tensor) -> torch.Tensor:
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

    def normalize_crops(self, crops: torch.Tensor, normalize: transforms.Normalize) -> torch.Tensor:
        return torch.stack([normalize(crop) for crop in crops])

    def foo(self, x):
        return x


def get_transform_img(args: any):
    """
    densenet needs RGB images and normalization.
    """

    # When running the classifier training, need to make sure that feature_extractor_img is not set.
    # need to make sure that img_clf and feature_extractor_img need the same transformations
    if (
            args.img_clf_type != 'densenet'
            and args.feature_extractor_img != 'densenet'
    ):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(args.img_size, args.img_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
    custom_transforms = CustomTransforms()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    crops_transform = get_crops_transform(args)
    transformation_list = [
        transforms.ToPILImage(),
        transforms.Lambda(custom_transforms.to_RGB),
        transforms.Resize(args.img_size)]
    if args.n_crops not in [10, 5]:
        transformation_list.extend([transforms.ToTensor(), normalize])
    else:
        # image is split in n_crops number of crops, stacked and every crop is then normalized
        if args.n_crops == 10:
            transformation_list.append(transforms.TenCrop(224))
        elif args.n_crops == 5:
            crops_transform = transforms.FiveCrop(224)

        transformation_list.extend([crops_transform, transforms.Lambda(
            custom_transforms.crops_to_tensor),
                                    transforms.Lambda(
                                        custom_transforms.normalize_crops)])
    return transforms.Compose(transformation_list)


def get_crops_transform(args) -> transforms:
    if args.n_crops == 10:
        return transforms.TenCrop(224)
    elif args.n_crops == 5:
        return transforms.FiveCrop(224)
    else:
        # ugly...
        return transforms.Lambda(lambda x: x)


def get_data_loaders(args, dataset):
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        num_workers = args.dataloader_workers // args.world_size
    else:
        sampler = None
        num_workers = args.dataloader_workers
    d_loader = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=(sampler is None),
                          num_workers=num_workers,
                          sampler=sampler, pin_memory=True)

    assert len(d_loader), f'length of the dataloader needs to be at least 1, it is {len(d_loader)}'

    return sampler, d_loader


def samplers_set_epoch(args, train_sampler, test_sampler, epoch: int) -> None:
    if args.distributed:
        for sampler in [train_sampler, test_sampler]:
            sampler.set_epoch(epoch)
