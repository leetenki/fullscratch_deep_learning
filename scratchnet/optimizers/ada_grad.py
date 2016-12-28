import numpy as np

class AdaGrad:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
        self.h = {}

        for name, link in model.layers.items():
            if hasattr(link, "W") and hasattr(link, "dW"):
                self.h[name+"_W"] = np.zeros_like(link.W)

            if hasattr(link, "b") and hasattr(link, "db"):
                self.h[name+"_b"] = np.zeros_like(link.b)

    
    def update(self):
        for name, link in self.model.layers.items():
            if hasattr(link, "W") and hasattr(link, "dW"):
                h_name = name+"_W"
                self.h[h_name] += link.dW * link.dW
                link.W -= self.lr * link.dW / (np.sqrt(self.h[h_name]) + 1e-7) # ゼロ除算防止

            if hasattr(link, "b") and hasattr(link, "db"):
                h_name = name+"_b"
                self.h[h_name] += link.db * link.db
                link.b -= self.lr * link.db / (np.sqrt(self.h[h_name]) + 1e-7) # ゼロ除算防止
