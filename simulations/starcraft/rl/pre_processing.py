from collections import namedtuple
import collections
import six
import math

import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

from collections import defaultdict



FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])

NUM_FUNCTIONS = len(actions.FUNCTIONS)
MINIMAP_FEATURE_SIZE = 3
SCREEN_FEATURE_SIZE = 2
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale
SUPP_DEP_COUNT = 0


_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_MARINE = 48
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_FACTORY = 27
_TERRAN_FACTORYTECHLAB = 39
_TERRAN_STARPORT = 28
_TERRAN_STARPORTTECHLAB = 41
_TERRAN_BANSHEE = 55
_TERRAN_SIEGETANK = 33 
_NEUTRAL_MINERAL_FIELD = 341


_NO_OP = actions.FUNCTIONS.no_op.id #0
_SELECT_POINT = actions.FUNCTIONS.select_point.id #2
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id #91
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id #42
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id #477
_SELECT_ARMY = actions.FUNCTIONS.select_army.id #7
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id #13
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id #264
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id #6
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id #53
_BUILD_REFINARY = actions.FUNCTIONS.Build_Refinery_screen.id #Build_Refinery_screen
_BUILD_TECHLAB_FACTORY = actions.FUNCTIONS.Build_TechLab_Factory_quick.id #96
_TRAIN_TANK = actions.FUNCTIONS.Train_SiegeTank_quick.id #492
_BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id #89
_BUILD_TECHLAB_STARPORT = actions.FUNCTIONS.Build_TechLab_Starport_quick.id #98
_TRAIN_BANSHEE = actions.FUNCTIONS.Train_Banshee_quick.id #459


FLAT_FEATURES = [
    FlatFeature(0,  features.FeatureType.SCALAR, 1, 'worker_count'),
    FlatFeature(1,  features.FeatureType.SCALAR, 1, 'supply_depots'),
    FlatFeature(2,  features.FeatureType.SCALAR, 1, 'barracks'),
    FlatFeature(3,  features.FeatureType.SCALAR, 1, 'amry_count'),
    FlatFeature(4,  features.FeatureType.SCALAR, 1, 'army_health'),
    FlatFeature(5,  features.FeatureType.SCALAR, 1, 'unit_location'),
    FlatFeature(6,  features.FeatureType.SCALAR, 1, 'enemey_unit_location'),
    FlatFeature(7,  features.FeatureType.SCALAR, 1, 'destroyed_enemies'),
    FlatFeature(8,  features.FeatureType.SCALAR, 1, 'destroyed_buildings'),
]

class ScFeatures(collections.namedtuple("ScreenFeatures", [
 "unit_type", "selected"])):
  """The set of screen feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette, clip) in six.iteritems(kwargs):
      feats[name] = features.Feature(
          index=ScFeatures._fields.index(name),
          name=name,
          layer_set="renders",
          full_name="screen " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=clip)
    return super(ScFeatures, cls).__new__(cls, **feats)




class MiFeatures(collections.namedtuple("MinimapFeatures", ["player_relative", "selected", "height_map"])):
  """The set of minimap feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette) in six.iteritems(kwargs):
      feats[name] = features.Feature(
          index=MiFeatures._fields.index(name),
          name=name,
          layer_set="minimap_renders",
          full_name="minimap " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=False)
    return super(MiFeatures, cls).__new__(cls, **feats)




SFEATURES = ScFeatures(
    unit_type=(None, features.FeatureType.CATEGORICAL, features.colors.unit_type, False),
    selected=(2, features.FeatureType.CATEGORICAL, features.colors.SELECTED_PALETTE, False),
)

MFEATURES = MiFeatures(
    height_map=(256, features.FeatureType.SCALAR, features.colors.winter),
    player_relative=(5, features.FeatureType.CATEGORICAL,
                     features.colors.PLAYER_RELATIVE_PALETTE),
    selected=(2, features.FeatureType.CATEGORICAL, features.colors.winter),

)


is_spatial_action = {}
is_queued_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']
  is_queued_action[arg_type] = name in ['queued']


def stack_ndarray_dicts(lst, axis=0):
  """Concatenate ndarray values from list of dicts
  along new axis."""
  res = {}
  for k in lst[0].keys():
    res[k] = np.stack([d[k] for d in lst], axis=axis)
  return res


class Preprocessor():
  """Compute network inputs from pysc2 observations.

  See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
  for the semantics of the available observations.
  """

  def __init__(self, obs_spec, minimap_size=(64,64)):
    self.screen_channels = SCREEN_FEATURE_SIZE
    self.minimap_channels = MINIMAP_FEATURE_SIZE
    self.flat_channels = len(FLAT_FEATURES)
    self.available_actions_channels = NUM_FUNCTIONS
    self.minimap_size = minimap_size

  def get_input_channels(self):
    """Get static channel dimensions of network inputs."""
    return {
        'screen': self.screen_channels,
        'minimap': self.minimap_channels,
        'flat': self.flat_channels,
        'available_actions': self.available_actions_channels}

  def preprocess_obs(self, obs_list):
    return stack_ndarray_dicts(
        [self._preprocess_obs(o.observation) for o in obs_list])

  def _preprocess_obs(self, obs):
    """Compute screen, minimap and flat network inputs from raw observations.
    """
    unit_type = obs['feature_screen'][_UNIT_TYPE]
    unit_counts = obs['unit_counts']
    
    unit_counts_dict = defaultdict(list)

    for k, v in unit_counts:
        unit_counts_dict[k] = v

    available_actions = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
    av_ob_act = []

    do_it = False
    if len(obs['single_select']) > 0 and obs['single_select'][0][0] == _TERRAN_MARINE:
        do_it = True
    if len(obs['multi_select']) > 0 and obs['multi_select'][0][0] == _TERRAN_MARINE:
        do_it = True

    if _NO_OP in obs['available_actions']:
        av_ob_act.append(_NO_OP)
    if _SELECT_POINT in obs['available_actions']:
        av_ob_act.append(_SELECT_POINT)
    if _BUILD_SUPPLY_DEPOT in obs['available_actions']:
        av_ob_act.append(_BUILD_SUPPLY_DEPOT) 
    if _BUILD_BARRACKS in obs['available_actions']:
        av_ob_act.append(_BUILD_BARRACKS)
    if _TRAIN_MARINE in obs['available_actions']:
        av_ob_act.append(_TRAIN_MARINE)
    if _SELECT_ARMY in obs['available_actions']:
        av_ob_act.append(_SELECT_ARMY)
    if do_it and _ATTACK_MINIMAP in obs['available_actions']:
        av_ob_act.append(_ATTACK_MINIMAP)

    """More actions for complex tasks and maps"""    
    # if _BUILD_FACTORY in obs['available_actions'] and unit_counts_dict.get(_TERRAN_FACTORY, 0) < 1:
    #     av_ob_act.append(_BUILD_FACTORY)
    # if _BUILD_TECHLAB_FACTORY in obs['available_actions']:
    #     av_ob_act.append(_BUILD_TECHLAB_FACTORY)
    # if _TRAIN_TANK in obs['available_actions']:
    #     av_ob_act.append(_TRAIN_TANK)
    # if _BUILD_STARPORT in obs['available_actions'] and unit_counts_dict.get(_TERRAN_STARPORT, 0) < 2:
    #     av_ob_act.append(_BUILD_STARPORT)
    # if _BUILD_TECHLAB_STARPORT in obs['available_actions']:
    #     av_ob_act.append(_BUILD_TECHLAB_STARPORT)    
    # if _TRAIN_BANSHEE in obs['available_actions']:
    #     av_ob_act.append(_TRAIN_BANSHEE)                       
    # if _HARVEST_GATHER in obs['available_actions']:
    #     av_ob_act.append(_HARVEST_GATHER)
    # if _BUILD_REFINARY in obs['available_actions']:
    #     av_ob_act.append(_BUILD_REFINARY)

    available_actions[av_ob_act] = 1

    minimap = self._preprocess_spatial([
                                        np.where(np.asarray(obs['feature_minimap'][features.MINIMAP_FEATURES.player_relative.index]) == _PLAYER_SELF , 1, 0),
                                        np.where(np.asarray(obs['feature_minimap'][features.MINIMAP_FEATURES.player_relative.index]) == _PLAYER_HOSTILE , 1, 0),
                                        obs['feature_minimap'][features.MINIMAP_FEATURES.selected.index],
                                        ])
    screen = self._preprocess_spatial([
                                        np.where(np.asarray(obs['feature_screen'][features.SCREEN_FEATURES.player_relative.index]) == _PLAYER_SELF , 1, 0),
                                        np.where(np.asarray(obs['feature_screen'][features.SCREEN_FEATURES.player_relative.index]) == _PLAYER_HOSTILE , 1, 0),
                                      ])
    
    
    enemy_loc_matrix = np.where(np.asarray(obs['feature_minimap'][features.MINIMAP_FEATURES.player_relative.index]) == _PLAYER_HOSTILE , 1, 0)
    player_loc_matrix = np.where(np.asarray(obs['feature_minimap'][features.MINIMAP_FEATURES.player_relative.index]) == _PLAYER_SELF , 1, 0)
    
    player_ob = [
        unit_counts_dict.get(_TERRAN_SCV, 0), #1
        unit_counts_dict.get(_TERRAN_SUPPLY_DEPOT, 0), #2
        unit_counts_dict.get(_TERRAN_BARRACKS, 0), #3
        unit_counts_dict.get(_TERRAN_MARINE, 0), #4
        obs['score_by_vital'][1][0], #5
        self._decode_relevant_location(player_loc_matrix, 2, 4, self.minimap_size[0], self.minimap_size[1]), #6
        self._decode_relevant_location(enemy_loc_matrix, 2, 4, self.minimap_size[0], self.minimap_size[1]),  #7     
        obs['score_cumulative'][5], #8
        obs['score_cumulative'][6], #9
        # add more state featurs for compelx tasks
        # unit_counts_dict.get(_TERRAN_FACTORY, 0),
        # unit_counts_dict.get(_TERRAN_STARPORT, 0),
        # unit_counts_dict.get(_TERRAN_STARPORTTECHLAB, 0),
        # unit_counts_dict.get(_TERRAN_BANSHEE, 0),
        
    ]
    flat = np.concatenate([
        player_ob
    ])

    return {
        'screen': screen,
        'minimap': minimap,
        'flat': flat,
        'available_actions': available_actions}

  def _preprocess_spatial(self, spatial):
    return np.transpose(spatial, [1, 2, 0])

  def _decode_relevant_location(self, loc_matrix, loc_divide_x, loc_divide_y, map_size_x, map_size_y):
    loc_squares = np.zeros(loc_divide_x * loc_divide_y)
    x_divide = int(map_size_x/loc_divide_x)
    y_divide = int(map_size_y/loc_divide_y) 
    unit_y, unit_x = loc_matrix.nonzero()
    for i in range(0, len(unit_y)):
        y = int(math.ceil((unit_y[i] + 1) / y_divide))
        x = int(math.ceil((unit_x[i] + 1) / x_divide))
        loc_squares[((y - 1) * 2) + (x - 1)] += 1
    return np.argmax(loc_squares)




