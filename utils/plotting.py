from matplotlib import pyplot as plt
import numpy as np


def plot(ax, data, estimated_labels):
        for label in set(estimated_labels):
            indices = np.nonzero(estimated_labels==label)
            samples = data[indices, :].squeeze()
            if samples.ndim ==1:
                x_points = samples[0]
                y_points = samples[1]
                z_points = samples[2]
            else:
                x_points = samples[:, 0]
                y_points = samples[:, 1]
                z_points = samples[:, 2]
            ax.scatter3D(x_points,z_points, y_points, label= label)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_ylim([-0.8,0.8])
            ax.set_zlabel('Z')
            plt.legend(loc = 'best',bbox_to_anchor=(1.05, 1))
