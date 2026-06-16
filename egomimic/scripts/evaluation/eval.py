class Eval:
    """
    Base class for all evaluation. Using this you can easily adopt your BC rollout pipeline
    """

    def __init__(self, config, ckpt_path):
        """
        config (DictConfig) : model config that would be used to instantiate the model
        ckpt_path (str) : model checkpoint path to instantiate the model
        """
        self.config = config
        self.ckpt_path = ckpt_path

    def process_batch_for_eval(self, batch):
        """
        Processes input data from simulation environment or robot data
        and filters our relevant information to prepare the batch for eval.
        Args:
            batch (dict): dictionary with torch.Tensors

        Returns:
            batch (dict) : processed dict of batches of form
            <embodiment_id> : {<dataset_keys> torch.Tensor}
        """
        raise NotImplementedError(
            "Must implement process_batch_for_eval for this subclass"
        )

    def perform_eval(self):
        """
        Calls the relevant methods to perform the rollout
        """
        raise NotImplementedError("Must implement perform_eval for this subclass")

    def eval_real(self):
        """
        Perform real world rollout
        """
        pass

    def eval_sim(self):
        """
        Perform sim rollout
        """
        pass
