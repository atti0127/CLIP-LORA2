import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from datasets.base_to_novel import (
    BaseToNovelDataset,
    RemappedSubset,
    configure_class_split,
)
from datasets.utils import DatasetBase, Datum
from lora import harmonic_mean, resolve_test_loaders
from loralib.utils import get_adapter_save_dir


def make_items(num_classes, samples_per_class=2):
    return [
        Datum(
            impath=f'/tmp/class_{label}_{index}.jpg',
            label=label,
            classname=f'class_{label}',
        )
        for label in range(num_classes)
        for index in range(samples_per_class)
    ]


class FakeImageDataset(Dataset):
    def __init__(self, labels):
        self.targets = list(labels)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), self.targets[index]


class BaseToNovelTest(unittest.TestCase):
    def test_datum_dataset_uses_coop_half_split_and_relabels(self):
        train = make_items(5)
        source = DatasetBase(
            train_x=train,
            val=make_items(5),
            test=make_items(5),
        )
        source.template = ['a photo of a {}.']

        split = BaseToNovelDataset(source)

        self.assertEqual(split.classnames, ['class_0', 'class_1', 'class_2'])
        self.assertEqual(split.test_new_classnames, ['class_3', 'class_4'])
        self.assertEqual(sorted({item.label for item in split.train_x}), [0, 1, 2])
        self.assertEqual(sorted({item.label for item in split.test_new}), [0, 1])
        self.assertEqual(
            sorted({item.classname for item in split.test_new}),
            ['class_3', 'class_4'])
        self.assertEqual(sorted({item.label for item in source.test}), [0, 1, 2, 3, 4])

    def test_torchvision_style_dataset_is_subsampled_and_relabelled(self):
        source = SimpleNamespace(
            classnames=['zero', 'one', 'two', 'three'],
            template=['a photo of a {}.'],
            train_x=FakeImageDataset([0, 1, 2, 3]),
            val=FakeImageDataset([0, 1, 2, 3]),
            test=FakeImageDataset([0, 1, 2, 3]),
        )

        split = configure_class_split(source, 'base2new')

        self.assertIsInstance(split.test_new, RemappedSubset)
        _, first_novel_label = split.test_new[0]
        _, second_novel_label = split.test_new[1]
        self.assertEqual((first_novel_label, second_novel_label), (0, 1))

    def test_standard_setting_preserves_all_classes(self):
        source = SimpleNamespace(
            classnames=['zero', 'one'], test=object())
        configured = configure_class_split(source, 'standard')
        self.assertIs(configured, source)
        self.assertIsNone(configured.test_new)
        self.assertEqual(configured.test_classnames, source.classnames)

    def test_harmonic_mean_and_loader_contract(self):
        self.assertAlmostEqual(harmonic_mean(80., 60.), 68.5714285714)
        self.assertEqual(harmonic_mean(0., 0.), 0.)
        args = SimpleNamespace(setting='base2new')
        self.assertEqual(resolve_test_loaders(args, ('base', 'novel')), ('base', 'novel'))
        with self.assertRaisesRegex(ValueError, 'base_loader'):
            resolve_test_loaders(args, 'base')

    def test_base_to_novel_checkpoint_path_is_isolated(self):
        args = SimpleNamespace(
            adaptation='hydra', backbone='ViT-B/16', dataset='dtd',
            save_path='/outputs', seed=1, setting='base2new', shots=16,
            router_temperature_schedule='fixed', hydra_diversity_weight=0.,
            hydra_balance_weight=0., image_anchor_weight=0.,
            text_anchor_weight=0.,
        )
        self.assertEqual(
            get_adapter_save_dir(args),
            '/outputs/vitb16/dtd/16shots/seed1/base2new/hydra')


if __name__ == '__main__':
    unittest.main()
