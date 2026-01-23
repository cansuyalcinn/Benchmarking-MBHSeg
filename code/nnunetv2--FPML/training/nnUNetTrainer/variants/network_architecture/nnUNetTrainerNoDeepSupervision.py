from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        percentage: float = 0.05,
        seed_name: str = "4",
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, percentage, seed_name)
        self.enable_deep_supervision = False