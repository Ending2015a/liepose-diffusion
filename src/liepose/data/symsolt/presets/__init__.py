from .base import BasePreset
from .cone import ConePreset
from .cube import CubePreset
from .cylinder import CylinderPreset
from .icosahedron import IcosahedronPreset
from .tetrahedron import TetrahedronPreset

preset_dict = {
  "tet": TetrahedronPreset,
  "cube": CubePreset,
  "icosa": IcosahedronPreset,
  "cone": ConePreset,
  "cyl": CylinderPreset,
}

preset_keys = list(preset_dict.keys())
