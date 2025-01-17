import pytorch_lightning as pl
import torch
import model_dispatcher
import config
# from dataset import DATADataModule
from glob import glob
from tqdm import tqdm

from torch import nn, optim

from losses import MAE, MSE, LyftLoss

from dataset_pointnet import LyftDataModule

class LyftNet(pl.LightningModule):   
    def __init__(self, hparams, criterion=LyftLoss):
        super().__init__()

        self.hparams = hparams
        self.save_hyperparameters(self.hparams)
        self.criterion = criterion

        self.model = model_dispatcher.MODELS[config.MODEL_NAME]

        # MODEL Definition
                

    def forward(self, x):
        return self.model(x)

    
    def configure_optimizers(self):
        # optimizer =  optim.Adam(self.parameters(), lr= self.hparams.lr, betas= (0.9,0.999), 
        #                   weight_decay= self.hparams.weight_decay, amsgrad=False)
        
        optimizer =  optim.Adam(self.parameters(), lr= config.LR, betas= (0.9,0.999), 
                          weight_decay= self.hparams.weight_decay, amsgrad=False)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=1e-5,
        )

        lr_plateau = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=self.hparams.lr_patience, cooldown=1, min_lr=1e-5, verbose=True),
            'monitor': 'checkpoint_on',  # Default: val_loss
            'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler, default
            'interval': 'epoch',
            'frequency': 1 
        }

        return [optimizer], [scheduler, lr_plateau]

    
    def training_step(self, batch, batch_idx):
        # x, y, y_av = [b.to(self.device) for b in batch]

        x = batch['x']
        y = batch['y']
        y_av = batch['y_av']

        c, preds = self(x)
        loss = self.criterion(c,preds,y, y_av)
        
        mse = MSE(preds, y, y_av)
        mae = MAE(preds, y, y_av)
        rmse = torch.sqrt(mse)
        
        # metric = (self.hparams.alpha*loss) + ((1-self.hparams.alpha)*rmse)
        # result = pl.TrainResult(minimize=metric)
        # result.loss = metric
        # result.log('metric', metric, prog_bar=True, logger=True, sync_dist=True)

        result = pl.TrainResult(minimize=loss)
        result.loss = loss
        result.log('loss', loss, prog_bar=False, logger=True, sync_dist=True)
        result.log('mse', mse, prog_bar=False, logger=True, sync_dist=True)
        result.log('mae', mae, prog_bar=False, logger=True, sync_dist=True)
        result.log('rmse', rmse, prog_bar=True, logger=True, sync_dist=True)

        return result
    
    def validation_step(self, batch, batch_idx):
        # x, y, y_av = [b.to(self.device) for b in batch]

        x = batch['x']
        y = batch['y']
        y_av = batch['y_av']

        c,preds = self(x)
        loss = self.criterion(c, preds, y, y_av)
        
        mse = MSE(preds, y, y_av)
        mae = MAE(preds, y, y_av)
        rmse = torch.sqrt(mse)
        
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.loss = loss
        result.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        result.log('val_mse', mse, prog_bar=False, logger=True, sync_dist=True)
        result.log('val_mae', mae, prog_bar=False, logger=True, sync_dist=True)
        result.log('val_rmse', rmse, prog_bar=True, logger=True, sync_dist=True)

        return result


    def test_step(self, batch, batch_idx):
        # x, y, y_av = [b.to(self.device) for b in batch]

        x = batch['x']
        y = batch['y']
        y_av = batch['y_av']

        c,preds = self(x)
        test_loss = self.criterion(c, preds, y, y_av)
        
        mse = MSE(preds, y, y_av)
        mae = MAE(preds, y, y_av)
        rmse = torch.sqrt(mse)
    
        result = pl.EvalResult(checkpoint_on=loss)
        result.loss = loss
        result.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        result.log('test_mse', mse, prog_bar=False, logger=True, sync_dist=True)
        result.log('test_mae', mae, prog_bar=False, logger=True, sync_dist=True)
        result.log('test_rmse', rmse, prog_bar=True, logger=True, sync_dist=True)

        return result

def find_ckpt():
    val = float('inf')
    path = None
    try:
        print('Looking for ckpts')
        for ckpt in tqdm(glob(os.path.join(config.save_path,f'{config.MODEL_NAME}-*.ckpt'))):
            basename = os.path.basename(ckpt)
            c = basename.split('.ckpt')[0]
            v = (float)((c.split('-')[-1]).split('=')[-1])
            if v< val:
                val=v
                path=ckpt
        if path != None:
            print('CKPT found, val =', val, ' Loading..')
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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dev', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=4)
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--frame_stride', type=int, default=config.FRAME_STRIDE)
    hparams = parser.parse_args()

    dm = LyftDataModule(hparams)

    lit_model = LyftNet(hparams)

    MODEL_SAVE = os.path.join(config.save_path, f'{hparams.model_name}-'+'model_ckpt-{epoch:02d}-{val_loss:.2f}')

    early_stopping = pl.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=hparams.es_patience)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=MODEL_SAVE, save_weights_only=False, mode='min', monitor='val_loss', verbose=False)

    resume_from_checkpoint = find_ckpt()

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=2,
        name='lightning_logs'
    )

    callbacks = [early_stopping]

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        accumulate_grad_batches=hparams.accumulate,
        profiler=True,
        callbacks=callbacks,
        checkpoint_callback=model_checkpoint,
        gradient_clip_val=1.0,
        max_epochs=hparams.epochs,
        # distributed_backend='ddp',
        resume_from_checkpoint=resume_from_checkpoint,
        logger=logger,
        fast_dev_run=hparams.dev,
        # val_check_interval=0.5,
        precision=16
    )

    trainer.fit(lit_model, dm)

    torch.save(lit_model.model.state_dict(), config.MODEL_LATEST)

    trainer.test()

    # lit_model.test_df.to_csv(config.SUB_FILE, index=False)
    # print(lit_model.test_df.head())