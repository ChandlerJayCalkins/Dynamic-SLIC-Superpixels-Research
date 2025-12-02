import numpy as np

class ImageIndex:
    def __init__(self):
        self.ids = []
        self.features = None

    def add(self, image_id, vec):
        vec = vec.astype(np.float32)[None, :]
        if self.features is None:
            self.features = vec
        else:
            self.features = np.vstack([self.features, vec])
        self.ids.append(image_id)

    def search(self, query_vec, top_k=5):
        q = query_vec.astype(np.float32)[None, :]
        dists = np.linalg.norm(self.features - q, axis=1)
        order = np.argsort(dists)[:top_k]
        return [(self.ids[i], float(dists[i])) for i in order]
