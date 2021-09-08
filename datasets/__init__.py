from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneflowDataset
from .middlebury_dataset import MiddleburyDataset

__datasets__ = {
    "kitti": KITTIDataset,
    "sceneflow": SceneflowDataset,
    "middlebury": MiddleburyDataset,
}
