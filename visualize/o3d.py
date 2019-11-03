import numpy as np
import open3d as o3d
import time

class visualizer:
    def __init__(self, **kwargs):
        print("init  start")
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(**kwargs)
        self.first = True
        print("init  done")
    def close(self):
        self.vis.destroy_window()
    def draw_np(self,x,col=None):
        self.pcd.points = o3d.utility.Vector3dVector(x)
        if col is not None:
            # col = np.zeros(x.shape)
            # col[:,2] = 1
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
    d = draw()
    T = time.time()
    for t in range(100):
        print(t)
        while time.time() < T + t:
            pass
        X = np.random.uniform(size=(10,3))
        d.draw_np(X)
