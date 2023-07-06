import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from .utils import add_colorbar


class CKA:
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        model1_name: str = None,
        model2_name: str = None,
        model1_layers: List[str] = None,
        model2_layers: List[str] = None,
        device: str = "cpu",
    ):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info["Name"] = model1.__repr__().split("(")[0]
        else:
            self.model1_info["Name"] = model1_name

        if model2_name is None:
            self.model2_info["Name"] = model2.__repr__().split("(")[0]
        else:
            self.model2_info["Name"] = model2_name

        if self.model1_info["Name"] == self.model2_info["Name"]:
            warn(
                f"Both model have identical names - {self.model2_info['Name']}. "
                "It may cause confusion when interpreting the results. "
                "Consider giving unique names to the models :)"
            )

        self.model1_info["Layers"] = []
        self.model2_info["Layers"] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn(
                "Model 1 seems to have a lot of layers. "
                "Consider giving a list of layers whose features you are concerned with "
                "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)"
            )

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn(
                "Model 2 seems to have a lot of layers. "
                "Consider giving a list of layers whose features you are concerned with "
                "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)"
            )

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(
        self,
        model: str,
        name: str,
        layer: nn.Module,
        inp: torch.Tensor,
        out: torch.Tensor,
    ):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _register_layers(self, model, model_layers, model_info, model_name):
        for name, layer in model.named_modules():
            # Don't register any embedding or dropout layers
            if "embedding" in name or "dropout" in name:
                continue
            if model_layers is not None:
                if name in model_layers:
                    model_info["Layers"] += [name]
                    layer.register_forward_hook(
                        partial(self._log_layer, model_name, name)
                    )
            else:
                model_info["Layers"] += [name]
                layer.register_forward_hook(partial(self._log_layer, model_name, name))

    def _insert_hooks(self):
        # Model 1
        self._register_layers(
            self.model1, self.model1_layers, self.model1_info, "model1"
        )

        # Model 2
        self._register_layers(
            self.model2, self.model2_layers, self.model2_info, "model2"
        )

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        if N <= 3:
            warn(f"N <= 3! N = {N}")
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        inc_result = (ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))
        result += (inc_result).item()
        to_subtract = (ones.t() @ K @ L @ ones) * 2 / (N - 2)
        result -= (to_subtract).item()
        final_result = (1 / (N * (N - 3)) * result).item()
        return final_result

    def compare(self, dataloader1: DataLoader, dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn(
                "Dataloader for Model 2 is not given. Using the same dataloader for both models."
            )
            dataloader2 = dataloader1

        self.model1_info["Dataset"] = dataloader1.dataset.__repr__().split("\n")[0]
        self.model2_info["Dataset"] = dataloader2.dataset.__repr__().split("\n")[0]

        N = (
            len(self.model1_layers)
            if self.model1_layers is not None
            else len(list(self.model1.modules()))
        )
        M = (
            len(self.model2_layers)
            if self.model2_layers is not None
            else len(list(self.model2.modules()))
        )

        # self.hsic_matrix = torch.zeros(N, M, 3)
        self.hsic_matrix = None

        num_batches = min(len(dataloader1), len(dataloader1))

        for x1, x2 in tqdm(
            zip(dataloader1, dataloader2),
            desc="| Comparing features |",
            total=num_batches,
        ):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))
            if self.hsic_matrix is None:
                self.hsic_matrix = torch.zeros(
                    len(self.model1_features), len(self.model2_features), 3
                )

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                kk_hsic = self._HSIC(K, K)
                self.hsic_matrix[i, :, 0] += kk_hsic / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert (
                        K.shape == L.shape
                    ), f"Feature shape mismatch! {K.shape}, {L.shape} for feature1 named {name1} and feature2 named {name2}"

                    kl_hsic = self._HSIC(K, L)
                    ll_hsic = self._HSIC(L, L)
                    self.hsic_matrix[i, j, 1] += kl_hsic / num_batches
                    self.hsic_matrix[i, j, 2] += ll_hsic / num_batches

        denom = self.hsic_matrix[:, :, 0].sqrt() * self.hsic_matrix[:, :, 2].sqrt()
        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (denom)
        if torch.isnan(self.hsic_matrix).any():
            warn("Found NANs in HSIC matrix.")

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info["Name"],
            "model2_name": self.model2_info["Name"],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info["Layers"],
            "model2_layers": self.model2_info["Layers"],
            "dataset1_name": self.model1_info["Dataset"],
            "dataset2_name": self.model2_info["Dataset"],
        }

    def plot_results(self, save_path: str = None, title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin="lower", cmap="magma")
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(
                f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18
            )

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()
