from typing import List, Dict
import numpy as np
from scipy.spatial.transform import Rotation
from pydantic import BaseModel
from datetime import datetime
from glob import glob
from parse import parse

class Position(BaseModel):
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

class Frame(BaseModel):
    id: int
    timestamp: datetime # UTC timestamp
    position: Position # in mm, object space
    rotation: Rotation # in radians, object frame
    color_file: str
    depth_file: str

    class Config:
        arbitrary_types_allowed = True

    def get_transform(self) -> np.ndarray:
        transform = np.identity(4)
        transform[:3, :3] = self.rotation.as_matrix()
        transform[:3, 3] = self.position.to_numpy()

        return transform

class Dataset(BaseModel):
    frames: List[Frame]

    @staticmethod
    def read(base_dir: str):
        trajectory = _read_log(f"{base_dir}/scanner.log")
        frames = []
        for t in trajectory:
            t["color_file"] = f"{base_dir}/rgb-{t['id']:05}.png"
            t["depth_file"] = f"{base_dir}/depth-{t['id']:05}.png"

            frames.append(Frame.parse_obj(t))

        return Dataset(frames=frames)




def _read_log(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        trajectory = [_parse_log_line(line.rstrip()) for line in f]
        return trajectory


def _parse_log_line(line: str) -> Dict:
    result = parse("{id:d} {timestamp:g} "
        "{position[x]:g} {position[y]:g} {position[z]:g} "
        "{rotation[x]:g} {rotation[y]:g} {rotation[z]:g}", line).named
    result["timestamp"] = datetime.fromtimestamp(result["timestamp"])
    result["rotation"] = Rotation.from_rotvec((result["rotation"]["x"], result["rotation"]["y"], result["rotation"]["z"]))
    return result