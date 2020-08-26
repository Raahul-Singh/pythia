import pytorch_lightning as pl

__all__ = ['LearningTask']

class LearningTask(pl.LightningModule):

    def __init__(self, model, hparams):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The Deep Learning Algorithm.
        hparams : dict
            The hyperparameter dictionary.
        """
        super().__init__()

        self._test_hparams(hparams)

        self.hparams = hparams
        self.model = model
        self.loss = self.hparams['loss']
        self.metrics = self.hparams['metrics']
        self.optimizer = self.hparams['optimizer']
        self.learning_rate = self.hparams['learning_rate']

        if 'scheduler' in self.hparams:
            self.scheduler = self.hparams['scheduler']

    def _test_hparams(self, hparams):
        """
        Tests the hyperparameter dictionary for essential parameters.

        Parameters
        ----------
        hparams : dict
            The hyperparameter dictionary.

        Raises
        ------
        ValueError
            If essential parameters are absent.
        """
        essential_args = ['loss', 'metrics', 'optimizer', 'learning_rate']
        for arg in essential_args:
            if arg not in hparams:
                raise ValueError(f"The model configuration hparams must contain {arg}")

    def training_step(self, batch, batch_idx):
        """
        The training step, run per batch during training.

        Parameters
        ----------
        batch : torch.Tensor
            The Data batch.
        batch_idx : int
            The batch id.

        Returns
        -------
        Result
            The output of forward pass.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):
        """
        The validation step, run per batch during training.

        Parameters
        ----------
        batch : torch.Tensor
            The Data batch.
        batch_idx : int
            The batch id.

        Returns
        -------
        Result
            The output of forward pass.

        Notes
        -----
        Supports Early Stopping on loss by default.

        # TODO: make a validation dictionary / callback for more control. 
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({'val_metric': metrics, 'val_loss': loss})
        return result

    def test_step(self, batch, batch_idx):
        """
        The test step, run per batch during test.

        Parameters
        ----------
        batch : torch.Tensor
            The Data batch.
        batch_idx : int
            The batch id.

        Returns
        -------
        Result
            The output of forward pass.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'test_metric': metrics, 'test_loss': loss})
        return result

    def configure_optimizers(self):
        """
        Configure the Training Optimizers.

        Returns
        -------
        optimizer : torch.optim
            The Optimizer
        """
        if 'optimizer_kwargs' in self.hparams:
            return self.optimizer(self.model.parameters(), self.learning_rate, **self.hparams['optimizer_kwargs'])
        else:
            return self.optimizer(self.model.parameters(), self.learning_rate)

    def configure_schedulers(self):
        """
        Configure the Learning Rate Schedulers.

        Returns
        -------
        scheduler : torch.optim.lr_scheduler
            The Learning Rate Scheduler

        # TODO: Check if scheduler has to be appended with optimizer.
        """
        if 'scheduler_kwargs' in self.hparams:
            return self.scheduler(self.configure_optimizers(), **self.hparams['scheduler_kwargs'])
        else:
            return self.scheduler(self.configure_optimizers())
