import os
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class TSNEPainter:
    def __init__(
            self,
            num_class: int,
            seed=666):
        self._fix_seed(seed)
        self.num_class = num_class

        self.labels = np.array([])
        self.outputs = None

    def run_test_data(self, net, dataloader):
        handle = self._register_hook(net)
        eval_loss, eval_acc = self.eval_model(dataloader, net)
        handle.remove()
        return

    def fit_and_draw(self, file_name):
        print("==> Calculating T-SNE vectors...")
        tsne_result = TSNE(2, init='pca').fit_transform(self.outputs)
        x_axis = self._normalize(tsne_result[:, 0])
        y_axis = self._normalize(tsne_result[:, 1])
        self._draw_scatter(x_axis, y_axis, file_name)
        return

    @torch.no_grad()
    def eval_model(self, dataloader, net):
        """
        Return loss, accuracy
        """
        net._reset_status()
        net.model = net.model.eval()

        pbar = tqdm.tqdm(dataloader, ncols=77, desc="==> Validating")
        for image, label in pbar:
            self.labels = np.concatenate((self.labels, label))

            image = image.to(net.device)
            label = label.to(net.device)

            output = net.model(image)
            pred = torch.argmax(output, dim=1)
            loss = net.criterion(output, label)

            net._update_status(pred, label, loss)

        return net._cal_metric()

    def _register_hook(self, net):
        def hook_collect_data(module_, input_, output_):
            outputs_now = output_.clone().cpu().numpy()
            outputs_now = outputs_now.reshape(outputs_now.shape[0], -1)
            if self.outputs is None:
                self.outputs = outputs_now
            else:
                self.outputs = np.concatenate((self.outputs, outputs_now))

        global_avg = net.model._modules.get('avgpool')
        handle = global_avg.register_forward_hook(hook_collect_data)
        return handle

    def _draw_scatter(self, x_axis, y_axis, file_name):
        color_map = plt.cm.rainbow(np.linspace(0, 1, self.num_class))
        pbar = tqdm.tqdm(
                range(self.num_class), ncols=77, desc="==> Painting T-SNE")
        for i_th in pbar:
            indices = [i for i, lbl in enumerate(self.labels) if lbl == i_th]
            i_th_x = np.take(x_axis, indices)
            i_th_y = np.take(y_axis, indices)
            plt.scatter(
                    i_th_x, i_th_y, label=i_th,
                    color=color_map[i_th], s=5)

        if self.num_class < 15:
            plt.legend(loc='best')
        plt.savefig(file_name)
        return

    def _fix_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return

    def _normalize(self, data):
        val_range = np.max(data) - np.min(data)
        norm_data = (data - np.min(data)) / val_range
        return norm_data


def main():
    import sys
    sys.path.append('/home/project/ssl/moco/')
    from model import ResNet
    from utils import arguments
    from utils import preprocess

    args = arguments.get_args()

    eval_loader = preprocess.prepare_loader(
            args.dataset, batch_size=args.batch_size,
            is_train=False, num_workers=args.num_workers)
    num_class = len(eval_loader.dataset.classes)

    model = ResNet(
            num_class,
            arch='resnet50',
            learning_rate=0,
            finetune=args.finetune,
            pretrained=True if args.pretrained is None else False)

    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)
        if state_dict.get('state_dict') is not None:
            state_dict = state_dict['state_dict']
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            remove_keys = list()
            for key in state_dict.keys():
                if 'fc' in key:
                    remove_keys.append(key)
            for key in remove_keys:
                del state_dict[key]
            model.load_state_dict(state_dict)

        print(f"[ INFO ] Load weights from {args.pretrained}")

    tsne_painter = TSNEPainter(num_class)
    tsne_painter.run_test_data(model, eval_loader)
    tsne_painter.fit_and_draw(args.result)

    return


if __name__ == "__main__":
    main()
