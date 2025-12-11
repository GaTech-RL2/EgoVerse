from abc import abstractmethod


class BaseInterface:
    """
    This is the base interface to define the method names for inherited camera interfaces
    """
    def __init__(
            self
    ):
        pass

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def get_last_obs(self):
        """
        Get last observation from camera
        """
        raise NotImplementedError

    @abstractmethod
    def get_color_intrinsics(self, mode=None):
        raise NotImplementedError

    @abstractmethod
    def get_color_distortion(self):
        raise NotImplementedError        

    @abstractmethod
    def get_depth_intrinsics(self, mode=None):
        raise NotImplementedError

    @abstractmethod
    def get_depth_distortion(self):
        raise NotImplementedError        
    
