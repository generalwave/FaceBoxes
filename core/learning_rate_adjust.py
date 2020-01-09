class LearningRateAdjust:
    def __init__(self, epoches, lr, gamma, optimizer):
        self.epoches = epoches
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optimizer

    def __call__(self, epoch):
        if self.epoches is not None and len(self.epoches) > 0 and epoch >= self.epoches[0]:
            self.lr = self.lr * self.gamma
            self.epoches.pop(0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        return self.lr
