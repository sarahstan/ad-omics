from torch.utils.data import DataLoader
from data.dataset import ADOmicsDataset


class ADOmicsDataLoader(DataLoader):
    def __init__(self, subset: str, data_path: str):
        self.data_path = data_path
        self.dataset = ADOmicsDataset(data_path=self.data_path, subset=subset)

    def get_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, **kwargs)
