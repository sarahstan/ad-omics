import lightning as ltn
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from trainer.lightning.mlp_classifier import ADClassifierLightning
from data import scDATA, ADOmicsDataset


def main():
    ### Load datasets
    data_path = "/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"
    scdata = scDATA(data_path=data_path)
    scdata.split_patient_level()
    training_dataset = ADOmicsDataset(scDATA=scdata, subset="train")
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=8,
        num_workers=31,
        shuffle=True,
    )
    validation_dataset = ADOmicsDataset(scDATA=scdata, subset="val")
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=2,
        num_workers=31,
        shuffle=False,
    )

    ### Get example size
    gene_count_vector_input_size = training_dataset[0][0].shape[0]

    ### Cell type input size
    cell_type_input_size = training_dataset[0][1].shape[0]

    ### Initialize model
    model = ADClassifierLightning(
        gene_input_dim=gene_count_vector_input_size,
        cell_type_input_dim=cell_type_input_size,
        hidden_dims=[5000, 500, 50, 5],
    )

    ### Set up loggers
    tb_logger = TensorBoardLogger(save_dir="tblogs", name="adomics_mlp")

    ### Set up trainer
    trainer = ltn.Trainer(logger=tb_logger, max_epochs=10)
    trainer.fit(model, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
