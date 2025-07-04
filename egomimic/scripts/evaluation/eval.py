import os
from datetime import datetime
class Eval:
    """
    Base class for all evaluation. Using this you can easily adopt your BC rollout pipeline
    """
    def __init__(self, eval_path):
        """
        config (DictConfig) : model config that would be used to instantiate the model
        ckpt_path (str) : model checkpoint path to instantiate the model
        """
        eval_dir = os.path.join(eval_path, 'eval')
        class_path = os.path.join(eval_dir, self.__class__.__name__)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.eval_path = os.path.join(class_path, timestamp)

        self.datamodule = None
        self.data_schematic = None
        
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
        raise NotImplementedError("Must implement process_batch_for_eval for this subclass")

    def run_eval(self):
        """
        Calls the relevant methods to perform the rollout
        """
        raise NotImplementedError("Must implement perform_eval for this subclass")