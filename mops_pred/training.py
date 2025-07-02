import lightning as L
import torch
import wandb
from absl import app, flags
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ml_collections.config_flags import config_flags

from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models import model_factory

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "./configs/resnet50_config.py",
    "Experiment Configuration",
)
flags.DEFINE_enum(
    "mode", "train", ["train", "debug", "test"], "Running mode: train or debug"
)

flags.DEFINE_string("checkpoint", None, "Checkpoint Path")


def run_training(cfg, model, debug=False):
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        offline=debug,
        config=cfg.to_dict(),
    )

    train_dl, test_dl = create_dataloader(cfg)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=20,
        filename="{epoch:03d}",
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        fast_dev_run=debug,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=test_dl,
    )


def main(argv):
    cfg = FLAGS.config
    debug = FLAGS.mode == "debug"

    torch.manual_seed(cfg.seed)
    model = model_factory.create_model(cfg.to_dict()["model"])
    model.cuda()

    if FLAGS.checkpoint is not None:
        model.load_state_dict(torch.load(FLAGS.checkpoint)["model_state_dict"])

    run_training(cfg, model, debug=debug)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
