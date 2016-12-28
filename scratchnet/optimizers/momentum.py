import numpy as np

class Momentum:
    def __init__(self, model, lr=0.01, momentum=0.9):
        self.model = model
        self.momentum = momentum
        self.lr = lr
        self.v = {}
        for name, link in model.layers.items():
            if hasattr(link, "W") and hasattr(link, "dW"):
                self.v[name+"_W"] = np.zeros_like(link.W)

            if hasattr(link, "b") and hasattr(link, "db"):
                self.v[name+"_b"] = np.zeros_like(link.b)

    
    def update(self):
        for name, link in self.model.layers.items():
            if hasattr(link, "W") and hasattr(link, "dW"):
                v_name = name+"_W"
                self.v[v_name] = self.momentum * self.v[v_name] - self.lr * link.dW
                link.W += self.v[v_name]

            if hasattr(link, "b") and hasattr(link, "db"):
                v_name = name+"_b"
                self.v[v_name] = self.momentum * self.v[v_name] - self.lr * link.db
                link.b += self.v[v_name]
