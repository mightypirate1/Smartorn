import numpy as np
import open3d as o3d
import time

class visualizer:
    def __init__(self, **kwargs):
        self.delta = kwargs.pop('dir_delta', 0.01)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(**kwargs)
        self.first = True
    def close(self):
        self.vis.destroy_window()
    def draw_np(self,x,col=None, dir=None):
        n = x.shape[0]
        delta = 0.01
        if dir is not None:
            assert dir.shape[0] == n, "must give same number of points and directions ({} != {})".format(n,dir.shape[0])
            p = np.concatenate([x, x+self.delta*dir], axis=0)
            if col is not None:
                col = np.concatenate([col, np.zeros_like(col)])
        else:
            p = x
        self.pcd.points = o3d.utility.Vector3dVector(p)
        if col is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(col)
        if self.first:
            self.vis.add_geometry(self.pcd)
            # self.vis.run()
            self.first = False
        # else:
        self.vis.poll_events()
        self.vis.update_geometry()
        self.vis.update_renderer()

if __name__ == "__main__":
    d = visualizer()
    T = time.time()
    for t in range(10):
        print(t)
        while time.time() < T + t:
            pass
        X = np.random.uniform(size=(10,3))
        D = np.random.uniform(size=(10,3))
        D = D / np.linalg.norm(D, axis=-1, keepdims=True)
        d.draw_np(X, col=X, dir=D)
