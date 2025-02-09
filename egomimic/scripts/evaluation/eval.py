class Eval:
    """
    Base class for all evaluation. Using this you can easily adopt your BC rollout pipeline
    """
    def __init__(self):
        pass
    
    def process_batch_for_eval(self):
        """
        """
        raise NotImplementedError("Must implement process_batch_for_eval for this subclass")
    
    def eval_real(self):
        """
        """
        raise NotImplementedError("Must implement eval_real for this subclass")

    def eval_sim(self):
        """
        """
        raise NotImplementedError("Must implement eval_sim for this subclass")

