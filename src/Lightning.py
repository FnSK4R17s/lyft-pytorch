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
    def __init__(self, batch_size=32, lr=5e-4, weight_decay=1e-8, num_workers=0, 
                 criterion=LyftLoss, data_root=config.DATA_ROOT,  epochs=1):
        super().__init__()

       
        self.save_hyperparameters(
            dict(
                HBACKWARD = config.HBACKWARD,
                HFORWARD = config.HFORWARD,
                NFRAMES = config.NFRAMES,
                FRAME_STRIDE = config.FRAME_STRIDE,
                AGENT_FEATURE_DIM = config.AGENT_FEATURE_DIM,
                MAX_AGENTS = config.MAX_AGENTS,
                TRAIN_ZARR = config.TRAIN_ZARR,
                VALID_ZARR = config.VALID_ZARR,
                batch_size = batch_size,
                lr=lr,
                weight_decay=weight_decay,
                num_workers=num_workers,
                criterion=criterion,
                epochs=epochs,
            )
        )
        
        self._train_data = None
        self._collate_fn = None
        self._train_loader = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
        self.lr = lr
        self.epochs=epochs
        
        self.weight_decay = weight_decay
        self.criterion = criterion
        
        self.data_root = data_root

        # MODEL Definition
                
        self.pnet = model_dispatcher.MODELS[config.MODEL_NAME]

        self.fc0 = nn.Sequential(
            nn.Linear(2048+256, 1024), nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 300),
        )

        self.c_net = nn.Sequential(
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        bsize, npoints, hb, nf = x.shape 
        
        # Push points to the last  dim
        x = x.transpose(1, 3)

        # Merge time with features
        x = x.reshape(bsize, hb*nf, npoints)

        x, trans, trans_feat = self.pnet(x)

        # Push featuresxtime to the last dim
        x = x.transpose(1,2)

        x = self.fc0(x)

        c = self.c_net(x)
        x = self.fc(x)

        return c,x

    
    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr= self.lr, betas= (0.9,0.999), 
                          weight_decay= self.weight_decay, amsgrad=False)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )
        return [optimizer], [scheduler]

    
    def training_step(self, batch):
        x, y, y_av = batch
        c, preds = self(x)
        loss = self.criterion(c,preds,y, y_av)
        
        result = pl.TrainResult(minimize=loss)
        result.loss = loss
        mse = MSE(preds, y, y_av)
        result.mse = mse
        result.mae = MAE(preds, y, y_av)
        result.rmse = torch.sqrt(mse)
        return result
    
    def validation_step(self, batch):
        x, y, y_av = batch
        c,preds = self(x)
        loss = self.criterion(c, preds, y, y_av)
        
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.loss = loss
        mse = MSE(preds, y, y_av)
        result.mse = mse
        result.mae = MAE(preds, y, y_av)
        result.rmse = torch.sqrt(mse)
        result.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        return result


    def test_step(self, batch):
        x, y, y_av = batch
        c,preds = self(x)
        loss = self.criterion(c, preds, y, y_av)
        
        result = pl.EvalResult(checkpoint_on=loss)
        result.loss = loss
        mse = MSE(preds, y, y_av)
        result.mse = mse
        result.mae = MAE(preds, y, y_av)
        result.rmse = torch.sqrt(mse)
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

    dm = LyftDataModule(batch_size=hparams.batch_size, num_workers=hparams.num_workers)

    lit_model = LyftNet(hparams)

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