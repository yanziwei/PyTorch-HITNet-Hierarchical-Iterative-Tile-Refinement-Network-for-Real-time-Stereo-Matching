from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneflowDataset

__datasets__ = {
    "kitti": KITTIDataset,
    "sceneflow": SceneflowDataset
}
