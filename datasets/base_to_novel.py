import math

from torch.utils.data import Dataset

from .utils import Datum


VALID_SETTINGS = ('standard', 'base2new')


class RemappedSubset(Dataset):
    """A dataset subset whose labels are contiguous within the selected split."""

    def __init__(self, dataset, selected_labels):
        self.dataset = dataset
        self.selected_labels = tuple(selected_labels)
        self.label_mapping = {
            label: new_label
            for new_label, label in enumerate(self.selected_labels)
        }
        selected = set(self.selected_labels)
        self.indices = [
            index for index, label in enumerate(dataset.targets)
            if int(label) in selected
        ]
        if not self.indices:
            raise ValueError('The selected class split contains no samples')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample, label = self.dataset[self.indices[index]]
        return sample, self.label_mapping[int(label)]


def _subsample_datum_list(data_source, selected_labels):
    selected = set(selected_labels)
    relabeler = {
        label: new_label for new_label, label in enumerate(selected_labels)
    }
    output = []
    for item in data_source:
        if item.label not in selected:
            continue
        output.append(Datum(
            impath=item.impath,
            label=relabeler[item.label],
            domain=item.domain,
            classname=item.classname,
        ))
    if not output:
        raise ValueError('The selected class split contains no samples')
    return output


def subsample_classes(data_source, selected_labels):
    """Subsample and relabel Datum lists or torchvision-style datasets."""

    if hasattr(data_source, 'targets'):
        return RemappedSubset(data_source, selected_labels)
    return _subsample_datum_list(data_source, selected_labels)


class BaseToNovelDataset:
    """A base/novel view following the CoOp, MMA, and 2SFS split protocol."""

    def __init__(self, dataset):
        classnames = list(dataset.classnames)
        if len(classnames) < 2:
            raise ValueError('Base-to-novel evaluation requires at least two classes')

        split_index = math.ceil(len(classnames) / 2)
        base_labels = tuple(range(split_index))
        novel_labels = tuple(range(split_index, len(classnames)))

        self._source_dataset = dataset
        self.setting = 'base2new'
        self.template = dataset.template
        self.train_x = subsample_classes(dataset.train_x, base_labels)
        self.val = subsample_classes(dataset.val, base_labels)
        self.test = subsample_classes(dataset.test, base_labels)
        self.test_new = subsample_classes(dataset.test, novel_labels)

        self.classnames = classnames[:split_index]
        self.val_classnames = list(self.classnames)
        self.test_classnames = list(self.classnames)
        self.test_new_classnames = classnames[split_index:]
        self.num_classes = len(self.classnames)

        if set(self.classnames) & set(self.test_new_classnames):
            raise AssertionError('Base and novel class names must be disjoint')

    def __getattr__(self, name):
        return getattr(self._source_dataset, name)


def configure_class_split(dataset, setting):
    if setting not in VALID_SETTINGS:
        raise ValueError(
            f'Unknown evaluation setting {setting!r}; expected one of {VALID_SETTINGS}')
    if setting == 'base2new':
        return BaseToNovelDataset(dataset)

    dataset.setting = 'standard'
    dataset.test_new = None
    dataset.test_classnames = list(dataset.classnames)
    dataset.test_new_classnames = None
    return dataset
