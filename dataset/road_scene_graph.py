import os
import sys
from typing import Callable, List, Optional, Iterable

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.utils import get_polygon, Minkowski_sum, signed_distance, rotation, normalize_angle
sys.path.pop(-1)

lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}



class RoadSceneDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        episode: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if episode is None:
            episode = np.random.randint(0, len(self.processed_paths))
        self.load(self.processed_paths[episode])


    @property
    def raw_file_names(self) -> List[str]:
        files = os.listdir(self.raw_dir)
        for file in files[::-1]:
            if file.split('.')[-1]!='npy':
                files.pop(-1)

        return files


    @property
    def processed_file_names(self) -> List[str]:
        return [f'episode_{episode}.pt' for episode in range(len(self.raw_file_names))]


    def process(self) -> None:

        for file, processed_path in zip(self.raw_file_names, self.processed_paths):
            episode = np.load(os.path.join(self.raw_dir, file), allow_pickle=True)

            data_list = []
            for scene in episode:
                data = self.data_from_scene(scene)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            self.save(data_list, processed_path)


    def data_from_scene(self, scene):
        data = Data()
        data.x = torch.from_numpy(np.c_[scene['observation'][:, 1:9]]).to(torch.float)
        row = torch.from_numpy(np.r_[np.zeros(4), np.arange(1, 5)]).to(torch.long)
        col = torch.from_numpy(np.r_[np.arange(1, 5), np.zeros(4)]).to(torch.long)
        data.edge_index = torch.stack([row, col], dim=0)
        data.time = torch.from_numpy(np.array([scene['timestamp']])).to(torch.float)
        data.lc = torch.from_numpy(np.array([scene['action'][0]])).to(torch.long)
        data.u = torch.from_numpy(np.array([5.0*scene['action'][1]])).to(torch.float)
        for param in scene['parameters']:
            if lane_change_dict[param[0]]==scene['action'][0]:
                slack_penalties = param[2]
        data.slack_penalties = torch.from_numpy(np.array([slack_penalties])).to(torch.float)
        data.P = torch.from_numpy(scene['QP']['P']).to(torch.float)
        data.q = torch.from_numpy(scene['QP']['q']).to(torch.float)
        data.G = torch.from_numpy(scene['QP']['G']).to(torch.float)
        data.h = torch.from_numpy(scene['QP']['h']).to(torch.float)
        data.lb = torch.from_numpy(scene['QP']['lb']).to(torch.float)
        data.ub = torch.from_numpy(scene['QP']['ub']).to(torch.float)
        return data



class RelativeRoadSceneDataset(RoadSceneDataset):

    #def __init__(
    #    self,
    #    root: str,
    #    episode: Optional[int] = None,
    #    transform: Optional[Callable] = None,
    #    pre_transform: Optional[Callable] = None,
    #    pre_filter: Optional[Callable] = None,
    #    force_reload: bool = False,
    #) -> None:

    #    super().__init__(root, episode, transform, pre_transform, pre_filter,
    #                     force_reload=force_reload)

    def data_from_scene(self, scene):
        data = Data()
        R = rotation(scene['observation'][0, 5]).T
        data.x = torch.from_numpy(
            np.c_[
                (scene['observation'][:, 1:3] - scene['observation'][:1, 1:3]) @ R,
                (scene['observation'][:, 3:5] - scene['observation'][:1, 3:5]) @ R,
                np.cos(scene['observation'][:, 5:6] - scene['observation'][:1, 5:6]),
                np.sin(scene['observation'][:, 5:6] - scene['observation'][:1, 5:6]),
                scene['observation'][:, 6:9]
            ]
        ).to(torch.float)
        row = torch.from_numpy(np.r_[np.zeros(4), np.arange(1, 5)]).to(torch.long)
        col = torch.from_numpy(np.r_[np.arange(1, 5), np.zeros(4)]).to(torch.long)
        data.edge_index = torch.stack([row, col], dim=0)
        data.time = torch.from_numpy(np.array([scene['timestamp']])).to(torch.float)
        data.pos = torch.from_numpy(np.c_[scene['observation'][:, 1:6]]).to(torch.float)
        data.lc = torch.from_numpy(np.array([scene['action'][0]])).to(torch.long)
        data.u = torch.from_numpy(np.array([5.0*scene['action'][1]])).to(torch.float)
        for param in scene['parameters']:
            if lane_change_dict[param[0]]==scene['action'][0]:
                slack_penalties = param[2]
        data.slack_penalties = torch.from_numpy(np.array([slack_penalties])).to(torch.float)
        data.P = torch.from_numpy(scene['QP']['P']).to(torch.float)
        data.q = torch.from_numpy(scene['QP']['q']).to(torch.float)
        data.G = torch.from_numpy(scene['QP']['G']).to(torch.float)
        data.h = torch.from_numpy(scene['QP']['h']).to(torch.float)
        data.lb = torch.from_numpy(scene['QP']['lb']).to(torch.float)
        data.ub = torch.from_numpy(scene['QP']['ub']).to(torch.float)
        return data

    

class HeteroRoadSceneDataset(RoadSceneDataset):

    #def __init__(
    #    self,
    #    root: str,
    #    episode: Optional[int] = None,
    #    transform: Optional[Callable] = None,
    #    pre_transform: Optional[Callable] = None,
    #    pre_filter: Optional[Callable] = None,
    #    force_reload: bool = False,
    #) -> None:

    #    super().__init__(root, episode, transform, pre_transform, pre_filter,
    #                     force_reload=force_reload)

    def data_from_scene(self, scene):
        data = HeteroData()
        data['ego_vehicle'].x = torch.from_numpy(scene['observation'][:1, 6:9]).to(torch.float)
        data['ego_vehicle'].pos = torch.from_numpy(np.c_[scene['observation'][:1, 1:6]]).to(torch.float)
        data['other_vehicle'].x = torch.from_numpy(scene['observation'][1:, 6:9]).to(torch.float)
        data['other_vehicle'].pos = torch.from_numpy(np.c_[scene['observation'][1:, 1:6]]).to(torch.float)
        row = torch.from_numpy(np.zeros(4)).to(torch.long)
        col = torch.from_numpy(np.arange(4)).to(torch.long)
        R = rotation(scene['observation'][0, 5]).T
        data['ego_vehicle', 'to', 'other_vehicle'].edge_index = torch.stack([row, col], dim=0)
        data['ego_vehicle', 'to', 'other_vehicle'].edge_attr = torch.from_numpy(
            np.c_[
                (scene['observation'][1:, 1:3] - scene['observation'][:1, 1:3]) @ R,
                (scene['observation'][1:, 3:5] - scene['observation'][:1, 3:5]) @ R,
                np.cos(scene['observation'][1:, 5:6] - scene['observation'][:1, 5:6]),
                np.sin(scene['observation'][1:, 5:6] - scene['observation'][:1, 5:6]),
            ]
        ).to(torch.float)
        data['other_vehicle', 'to', 'ego_vehicle'].edge_index = torch.stack([col, row], dim=0)
        data['other_vehicle', 'to', 'ego_vehicle'].edge_attr = torch.from_numpy(
            np.vstack(
                [
                    np.c_[
                        (scene['observation'][:1, 1:3] - scene['observation'][k+1:k+2, 1:3]) @ R,
                        (scene['observation'][:1, 3:5] - scene['observation'][k+1:k+2, 3:5]) @ R,
                        np.cos(scene['observation'][:1, 5:6] - scene['observation'][k+1:k+2, 5:6]),
                        np.sin(scene['observation'][:1, 5:6] - scene['observation'][k+1:k+2, 5:6]),
                    ] for k, R in enumerate([rotation(a).T for a in scene['observation'][1:, 5]])
                ]
            )
        ).to(torch.float)
        row = []
        col = []
        edge_attr = []
        for j, oj in enumerate(scene['observation'][1:, :]):
            R = rotation(scene['observation'][j, 5]).T
            for k, ok in enumerate(scene['observation'][1:, :]):
                if j!=k:
                    row.append(j)
                    col.append(k)
                    edge_attr.append(
                        np.c_[
                            (ok[np.newaxis, 1:3] - oj[np.newaxis, 1:3]) @ R,
                            (ok[np.newaxis, 3:5] - oj[np.newaxis, 3:5]) @ R,
                            np.cos(ok[np.newaxis, 5:6] - oj[np.newaxis, 5:6]),
                            np.sin(ok[np.newaxis, 5:6] - oj[np.newaxis, 5:6]),
                        ]
                    )
        row = torch.from_numpy(np.array(row, dtype=int)).to(torch.long)
        col = torch.from_numpy(np.array(col, dtype=int)).to(torch.long)
        data['other_vehicle', 'to', 'other_vehicle'].edge_index = torch.stack([row, col], dim=0)
        data['other_vehicle', 'to', 'other_vehicle'].edge_attr = torch.from_numpy(
            np.vstack(edge_attr)
        ).to(torch.float)
        data.time = torch.from_numpy(np.array([scene['timestamp']])).to(torch.float)
        data.lc = torch.from_numpy(np.array([scene['action'][0]])).to(torch.long)
        data.u = torch.from_numpy(np.array([5.0*scene['action'][1]])).to(torch.float)
        for param in scene['parameters']:
            if lane_change_dict[param[0]]==scene['action'][0]:
                slack_penalties = param[2]
        data.slack_penalties = torch.from_numpy(np.array([slack_penalties])).to(torch.float)
        data.P = torch.from_numpy(scene['QP']['P']).to(torch.float)
        data.q = torch.from_numpy(scene['QP']['q']).to(torch.float)
        data.G = torch.from_numpy(scene['QP']['G']).to(torch.float)
        data.h = torch.from_numpy(scene['QP']['h']).to(torch.float)
        data.lb = torch.from_numpy(scene['QP']['lb']).to(torch.float)
        data.ub = torch.from_numpy(scene['QP']['ub']).to(torch.float)
        return data
    
if __name__=="__main__":

    root = os.path.join(os.path.dirname(__file__), 'rollout_hocbf')
    dataset = HeteroRoadSceneDataset(root, episode=0)
    print(dataset)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=5)
    batch_list = []
    for batch in loader:
        batch_list.append(batch)
    print(batch_list[0])
    print(batch_list[0][0])
