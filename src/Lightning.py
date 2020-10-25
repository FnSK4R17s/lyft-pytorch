import pytorch_lightning as pl
import torch
import model_dispatcher
import config
from dataset import DATADataModule
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm

class MODEL_NAME_Lightning(pl.LightningModule):
    def __init__(self, hparams):
        super(MODEL_NAME_Lightning, self).__init__()
        self.loss_fn = torch.nn.CTCLoss(blank=77)
        self.hparams = hparams
        # import model from model dispatcher
        self.model = model_dispatcher.MODELS[self.hparams.model_name]


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, cooldown=1, min_lr=1e-08, verbose=True),
            'monitor': 'val_checkpoint_on',
            'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler, default
            'interval': 'epoch',
            'frequency': 1 
        }
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_nb):
        x = batch['image']
        y = batch['label']
        y_hat = self(x)
        target_lengths = batch['length']
        input_lengths = batch['bucket']
        loss = self.loss_fn(y_hat, y, input_lengths, target_lengths)
        result = pl.TrainResult(minimize=loss)
        result.loss = loss
        return result


    def validation_step(self, batch, batch_nb):
        x = batch['image']
        y = batch['label']
        y_hat = self(x)
        target_lengths = batch['length']
        input_lengths = batch['bucket']
        loss = self.loss_fn(y_hat, y, input_lengths, target_lengths)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.loss = loss
        result.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return result


    def test_step(self, batch, batch_nb):
        x = batch['image']
        y = batch['label']
        y_hat = self(x)
        target_lengths = batch['length']
        input_lengths = batch['bucket']
        loss = self.loss_fn(y_hat, y, input_lengths, target_lengths)
        result = pl.EvalResult(checkpoint_on=loss)
        result.loss = loss
        result.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return result

def find_ckpt():
    val = float('inf')
    path = None
    try:
        print('Looking for ckpts')
        for ckpt in tqdm(glob(os.path.join(config.save_path,'*.ckpt'))):
            basename = os.path.basename(ckpt)
            c = basename.split('.ckpt')[0]
            v = (float)((c.split('-')[-1]).split('=')[-1])
            if v< val:
                val=v
                path=ckpt
    except:
        print('No ckpt found in directory')

    return path

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning.loggers import TensorBoardLogger
    import os

    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--batch_size', type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument('--model_name', type=str, default=config.MODEL_NAME)
    parser.add_argument('--accumulate', type=int, default=config.ACCUMULATE)
    parser.add_argument('--aws', type=bool, default=True)
    parser.add_argument('--gpus', type=str, default=config.GPUS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dev', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    hparams = parser.parse_args()

    dm = DATADataModule(batch_size=hparams.batch_size, num_workers=hparams.num_workers)

    lit_model = MODEL_NAME_Lightning(hparams)

    MODEL_SAVE = os.path.join(config.save_path, f'{hparams.model_name}-'+'model_ckpt-{epoch:02d}-{val_loss:.2f}')

    early_stopping = pl.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=5)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=MODEL_SAVE, save_weights_only=False, mode='min', monitor='val_loss', verbose=False)

    resume_from_checkpoint = find_ckpt()

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        accumulate_grad_batches=hparams.accumulate,
        profiler=True,
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        gradient_clip_val=0.5,
        max_epochs=hparams.epochs,
        distributed_backend='ddp',
        resume_from_checkpoint=resume_from_checkpoint,
        val_check_interval=0.25,
        logger=logger,
        fast_dev_run=hparams.dev,
    )

    trainer.fit(lit_model, dm)

    torch.save(lit_model.model.state_dict(), config.MODEL_LATEST)

    trainer.test()

    # lit_model.test_df.to_csv(config.SUB_FILE, index=False)
    # print(lit_model.test_df.head())