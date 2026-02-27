from pathlib import Path
from nda.data_loader import DataLoader, Partition


def main() -> None:
    loader = DataLoader(Path(__file__).parent / "static" / "data")

    partitions: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")
    train, val, test = [loader.load(partition) for partition in partitions]

    for partition, data in zip(partitions, (train, val, test)):
        print(len(data))


if __name__ == "__main__":
    main()
