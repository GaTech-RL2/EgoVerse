class Algo:
    """
    Base class for all algorithms.  These functions are expected by Pytorch Lightning trainer (pl_model.py)
    """

    def __init__(self):
        pass

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            batch (dict) : processed dict of batches of form
            <embodiment_id> : {<dataset_keys> torch.Tensor}
        """
        raise NotImplementedError(
            "Must implement process_batch_for_training in subclass"
        )

    def forward_training(self, batch):
        """
        One iteration of training.  Compute forward pass and compute losses.  Return predictions dictionary.  ACT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        raise NotImplementedError("Must implement forward_training in subclass")

    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        This is the only eval-time entry point the model is expected to expose;
        all metric computation and visualization is performed by Eval classes
        on top of these predictions.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
        """
        raise NotImplementedError("Must implement forward_eval in subclass")

    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)
        """
        raise NotImplementedError("Must implement compute_losses in subclass")

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic
        """
        raise NotImplementedError("Must implement log_info in subclass")
