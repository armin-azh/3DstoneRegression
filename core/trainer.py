# utils
from core.utils import flush_and_gc


class BaseTrain:
    def __init__(self, *args, **kwargs):
        super(BaseTrain).__init__(*args, **kwargs)

    @flush_and_gc
    def train_step(self, **kwargs):
        raise NotImplementedError

    @flush_and_gc
    def validation_step(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError


class TrainerV1(BaseTrain):
    def __init__(self, *args, **kwargs):
        super(TrainerV1, self).__init__()

    @flush_and_gc
    def train_step(self, **kwargs):
        pass

    @flush_and_gc
    def validation_step(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass
