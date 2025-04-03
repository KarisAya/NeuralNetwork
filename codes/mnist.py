from pathlib import Path
from PIL import Image


def save_mnist(path: Path):
    from torchvision.datasets import MNIST
    from hashlib import md5

    train_path = path / "train"

    for n in range(10):
        (train_path / str(n)).mkdir(exist_ok=True, parents=True)

    test_path = path / "test"

    for n in range(10):
        (test_path / str(n)).mkdir(exist_ok=True, parents=True)

    train_data = MNIST("", True, download=True)

    image: Image.Image

    for image, label in train_data:
        image.save(train_path / str(label) / f"{md5(image.tobytes()).hexdigest()}.png")

    test_data = MNIST("", False, download=True)

    for image, label in test_data:
        image.save(test_path / str(label) / f"{md5(image.tobytes()).hexdigest()}.png")


def load_mnist(path: Path | str):
    """
    路径中的子文件夹是label名称
    label文件夹下的文件是png图片,lable名称是数字
    图片会使用PIL打开然后使用transforms.ToTensor()直接转化为tensor
    """
    from torchvision.transforms.functional import to_tensor

    dataset = []
    if not isinstance(path, Path):
        path = Path(path)
    for label in path.iterdir():
        if label.is_dir():
            for image in label.iterdir():
                if image.suffix == ".png":
                    dataset.append((to_tensor(Image.open(image)), int(label.name)))
    return dataset


def load_mnist_to_tensor():
    from torchvision.datasets import MNIST
    from torchvision.transforms.functional import to_tensor

    train_data = MNIST("", True, download=True)

    train_dataset = [(to_tensor(image), label) for image, label in train_data]

    test_data = MNIST("", False, download=True)

    test_dataset = [(to_tensor(image), label) for image, label in test_data]

    return train_dataset, test_dataset
