from .random_access_dataset import TorchDataset


def show_info(name):
    print(f"\n\n--- {name.capitalize()} ---")
    dataset = TorchDataset(name, is_train=True)
    print(dataset.class_names)
    counter = dict((cls, val) for cls, val
                   in zip(dataset.class_names, dataset.class_counts))
    print(f"==> Total image: {sum(counter.values())}")
    for idx, (cls, num) in enumerate(counter.items(), 1):
        print(f"{cls+':':<47} {num:<7} | ", end='')
        if idx % 3 == 0:
            print()
    print('\n\n')
    return


def main():
    show_info('caltech101')
    show_info('food101')
    show_info('stanford_dogs')


if __name__ == "__main__":
    main()
