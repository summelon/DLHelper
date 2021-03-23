import os
import torch
from colored import fg, attr

import dlhelper.utils.preprocess as preprocess
import dlhelper.utils.arguments as arguments
from dlhelper.model import ResNet


def main():
    args = arguments.get_args()
    patience = 13
    patience_counter = 0
    whole_model = False
    check_threshold = 0.01  # 1%

    train_loader = preprocess.prepare_loader(
            args.dataset, base_dir=args.base_dir,
            batch_size=args.batch_size, is_train=True,
            num_workers=args.num_workers)
    eval_loader = preprocess.prepare_loader(
            args.dataset, base_dir=args.base_dir,
            batch_size=args.batch_size, is_train=False,
            num_workers=args.num_workers)
    num_class = len(train_loader.dataset.classes)

    model = ResNet(
            num_class,
            arch='resnet50',
            learning_rate=args.lr,
            finetune=args.finetune,
            pretrained=True if args.pretrained is None else False,
            deepmind_byol=args.deepmind_byol)

    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)
        if state_dict.get('state_dict') is not None:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        print(f"[ INFO ] Load weights from {args.pretrained}")

    if not args.test:
        print("[ INFO ] Show trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"\t=> {name}")

    if args.test:
        eval_loss, eval_acc = model.eval_model(eval_loader)
        return

    # --- Train loop ---
    best_acc = 0.0
    print(f"[ INFO ] Totally {args.epochs} epoch")
    for e in range(args.epochs):
        print(f"\n\n=== Epoch: {e} ===")
        train_loss, train_acc = model.train_model(train_loader)
        eval_loss, eval_acc = model.eval_model(eval_loader)

        if (best_acc + check_threshold) < eval_acc:
            best_acc = eval_acc
            torch.save(model if whole_model else model.state_dict(),
                       os.path.join(args.checkpoint, "best.pt"))
            print(f"=> Saved best model"
                  f"({fg(208)}eval acc: {best_acc:.2%}{attr(0)}) "
                  f"in {args.checkpoint}")
            patience_counter = 0

        patience_counter += 1
        if patience_counter >= patience:
            break
    return


if __name__ == "__main__":
    main()
