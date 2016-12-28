class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
    
    def update(self):
        for link in self.model.layers.values():
            if hasattr(link, "W") and hasattr(link, "dW"):
                link.W -= self.lr * link.dW

            if hasattr(link, "b") and hasattr(link, "db"):
                link.b -= self.lr * link.db
