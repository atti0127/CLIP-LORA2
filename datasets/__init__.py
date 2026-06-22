from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .base_to_novel import configure_class_split


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                }


def build_dataset(dataset, root_path, shots, preprocess, setting='standard'):
    if dataset == 'imagenet':
        output = dataset_list[dataset](root_path, shots, preprocess)
    else:
        output = dataset_list[dataset](root_path, shots)
    return configure_class_split(output, setting)
