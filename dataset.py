from dataclasses import dataclass, field
import os
from glob import glob 
from typing import Union
import itertools


def extract_date_and_num(traj_path): 
    traj = traj_path.split('/')
    traj_num = int(traj[-1][len('traj'):])
    dated_dir = traj[-4]
    return dated_dir, traj_num

def sort_trajectories(trajs): 
    return sorted(trajs, key=extract_date_and_num)


@dataclass
class DatasetDescription:
    dataset_directory: str 
    depth: int = 2 # depth=0 corresponds to being a dated directory
    exclude_directories: dict[str: int] = None
    start_date: Union[str, tuple[str, int]] = ('', -1)
    include_every: int = 1
    def __post_init__(self): 
        if self.exclude_directories is None: 
            self.exclude_directories = {
                'verified_data_dnw': 2, 
                'metal_trajs': 2
            }
    

class DirectoryIterator:
    def __init__(self, dataset: DatasetDescription): 
        self._dataset = dataset
    def get_trajectory_list(self):

        trajgroup_path = os.path.join(
            self._dataset.dataset_directory,
            *['*' for _ in range(self._dataset.depth)], 
            'raw', 
            'traj_group0', 
            'traj*'
        )
        all_trajgroups = glob(trajgroup_path)
        exclude_full_paths =  [os.path.join(self._dataset.dataset_directory, *['*' for _ in range(dist)], k) for k, dist in self._dataset.exclude_directories.items()]
        exclude_full_paths = [glob(pth) for pth in exclude_full_paths]
        exclude_full_paths = set(itertools.chain.from_iterable(exclude_full_paths))

        def exclude(traj_path): 
            for exclude_path in exclude_full_paths: 
                if traj_path.startswith(exclude_path): 
                    return True 
            return False 
        all_trajgroups = [traj for traj in all_trajgroups if not exclude(traj)]
        sorted_trajs = sorted(all_trajgroups, key=extract_date_and_num) 
        filtered_trajs = [
            traj for traj in sorted_trajs if extract_date_and_num(traj) >= self._dataset.start_date
        ]
        filtered_trajs = [
            traj for i, traj in enumerate(filtered_trajs) if i % self._dataset.include_every == 0
        ]
        return filtered_trajs