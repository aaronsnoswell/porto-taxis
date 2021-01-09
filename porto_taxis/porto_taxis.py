import os
import tqdm
import pickle
import warnings

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from mdp_extras.utils import PaddedMDPWarning
from mdp_extras import DiscreteImplicitExtras, FeatureFunction, padding_trick

from porto_taxis.utils import nonoverlapping_shared_subsequences, geoid_dist

from unimodal_irl import maxent_path_logprobs

PORTO_CBD_LATLON = np.array([41.1480637, -8.6329584])


class PortoExtras(DiscreteImplicitExtras):
    """Extras object for the Porto MDP"""

    def __init__(self, bin_prefix=os.path.join(os.path.dirname(__file__), "bin")):
        """C-tor - takes about 11 seconds"""

        # About 3 seconds to load
        with open(
            os.path.join(bin_prefix, "porto-road-graph-updated.pkl"), "rb"
        ) as file:
            self.rg = pickle.load(file)

        # Nodes are tuples of OpenStreetMaps object IDs indicating a start and end node
        # for a road segment e.g. (1295427230, 1295426882)
        self._nodes = list(self.rg.nodes)
        self._states = list(range(len(self._nodes)))
        self._node2state = {n: s for n, s in zip(self._nodes, self._states)}

        # Edges are tuples of states indicating a valid turn at an intersection from one
        # state to another e.g. ((1295427230, 1295426882), (1295426882, 1295426880))
        self._edges = list(self.rg.edges)
        self._actions = list(range(len(self._edges)))
        self._edge2action = {e: a for e, a in zip(self._edges, self._actions)}

        # Set initial state distribution to uniform
        self._p0s = np.ones(len(self._states)) / len(self._states)

        # No terminal states
        self._terminal_state_mask = np.zeros(len(self._states))

        # Discount factor
        self._gamma = 0.999

        # Build children dictionary - about 3 seconds
        self._children = {}
        max_children = 0
        self._children_fixedsize = np.zeros((len(self._states), 1), dtype=int) - 1
        for s, n in zip(self._states, self._nodes):
            vec = [self._node2state[succ] for succ in self.rg.succ[n]]
            self._children[s] = vec
            max_children = max(max_children, len(vec))

            # Incrementally build fixed-size children array, expanding if needed
            prev_max_children = self._children_fixedsize.shape[1]
            if max_children > prev_max_children:
                # Add extra column(s) to expand array
                tmp = np.zeros((len(self._states), max_children), dtype=int) - 1
                tmp[:, :prev_max_children] = self._children_fixedsize
                self._children_fixedsize = tmp
            self._children_fixedsize[s, : len(vec)] = vec

        # Build parents dictionary - about 3 seconds
        self._parents = {}
        max_parents = 0
        self._parents_fixedsize = np.zeros((len(self._states), 1), dtype=int) - 1
        for s, n in zip(self._states, self._nodes):
            vec = [self._node2state[pred] for pred in self.rg.pred[n]]
            self._parents[s] = vec
            max_parents = max(max_parents, len(vec))

            # Incrementally build fixed-size parents array, expanding if needed
            prev_max_parents = self._parents_fixedsize.shape[1]
            if max_parents > prev_max_parents:
                # Add extra column(s) to expand array
                tmp = np.zeros((len(self._states), max_parents), dtype=int) - 1
                tmp[:, :prev_max_parents] = self._parents_fixedsize
                self._parents_fixedsize = tmp
            self._parents_fixedsize[s, : len(vec)] = vec

        self._is_deterministic = True

        self._is_padded = False

    def t_prob(self, s1, a, s2):
        """Get transition probability"""
        if s2 in self._children[s1]:
            return 1.0
        else:
            return 0.0

    def plot_path(self, path, ax=None, **kwargs):
        """Plot a state path in GPS coordinates onto an image
        
        Args:
            path (list): List of (s, a)
            
            ax (matplotlib.Axes): Axis to plot to, or get current axis if None
            kwargs (dict): Keyword args to pass to matplotlib.pyplot.plot() function
        
        Returns:
            (list) List of matplotlib.lines.Line2D returned from matplotlib.pyplot.plot()
        """

        # Convert state path to node path
        node_path = [self._nodes[s] for (s, _) in path]

        # Convert nodes to GPS coords
        gps_coords = [self.rg.nodes[n]["start_latlon"] for n in node_path]

        # Add final GPS coord for end of last road segment
        gps_coords.append(self.rg.nodes[node_path[-1]]["end_latlon"])

        # Convert to matrix with Latitude, Longitude columns
        gps_coords = np.array(gps_coords)

        if ax is None:
            ax = plt.gca()
        return ax.plot(gps_coords[:, 1], gps_coords[:, 0], **kwargs)

    def path_dist(self, path):
        """Find distance (in km) of a path"""
        return (
            np.sum([self.rg.nodes[self._nodes[s]]["distance"] for s, _ in path])
            / 1000.0
        )

    def n2s_path(self, node_path):
        """Convert a node path to a state action path"""

        # Generate list of states from node path
        state_path = [self._node2state[n] for n in node_path]

        # Generate list of actions from node path
        edge_path = [e for e in zip(node_path[:-1], node_path[1:])]
        action_path = [self._edge2action[e] for e in edge_path]
        # Add final 'none' action
        action_path.append(None)

        # Zip states and actions
        return list(zip(state_path, action_path))

    def shortest_path(self, s1, sg):
        """Find the shortest (distance) path between two states
        
        Args:
            s1 (int): Starting state
            sg (int): Goal state
        
        Returns:
            (list): Shortest distance state-action path from s1 to sg
        """
        return self.n2s_path(
            nx.shortest_path(
                self.rg, self._nodes[s1], self._nodes[sg], weight="distance"
            )
        )

    def percent_distance_missed_metric(self, path_l, path_gt):
        """Compute % distance missed metric from learned to GT path
        
        Assumes paths are non-cyclic and share start and end states
        """
        gt_path_km = self.path_dist(path_gt)

        # Find overlapping sub-paths
        shared_distance = 0.0
        for shared_subpath in nonoverlapping_shared_subsequences(path_l, path_gt):
            shared_distance += self.path_dist(shared_subpath)

        # Compute % shared distance
        pc_shared_distance = shared_distance / gt_path_km

        # Distance missed is the complement of this
        return 1.0 - pc_shared_distance


class PortoFeatures(FeatureFunction):
    """Feature function for the Porto MDP"""

    # Geographic regions, and compass angles from E=0, (CCW +ve) where they point
    # The valid angle ranges are from -180 to 180, East is 0
    GEO_ZONES = {
        # For 'central', the number is the radius in KM from CBD
        "Central": 2.0,
        "North": np.deg2rad(90.0),
        "NorthNorthEast": np.deg2rad(67.5),
        "NorthEast": np.deg2rad(45.0),
        "EastNorthEast": np.deg2rad(22.5),
        "East": 0.0,
        "EastSouthEast": np.deg2rad(-22.5),
        "SouthEast": np.deg2rad(-45.0),
        "SouthSouthEast": np.deg2rad(-67.5),
        "South": np.deg2rad(-90.0),
        "SouthSouthWest": np.deg2rad(-112.5),
        "SouthWest": np.deg2rad(-135.0),
        "WestSouthWest": np.deg2rad(-157.5),
        "West": np.deg2rad(180.0),
        "WestNorthWest": np.deg2rad(157.5),
        "NorthWest": np.deg2rad(135.0),
        "NorthNorthWest": np.deg2rad(112.5),
    }

    EARTH_RADIUS_KM = 6371.0
    EARTH_CIRCUMFERNCE_KM = 2 * np.pi * (6371.0)

    def __init__(
        self, xtr, feature_names=["type", "speed",], use_geo=True,
    ):
        """C-tor
        
        Args:
            xtr (PortoExtras): Extras object
            feature_names (list): List of features in the road graph nodes to use.
                Options are 'type', 'speed', 'lanes', 'toll', however we recommend only
                using 'type' and 'speed' - lanes and toll are highly correlated with
                the first two options.
            use_geo (bool): If true, generate geographical features as well
        """

        # Porto uses state-based features
        super().__init__(self.Type.OBSERVATION)

        self._use_geo = use_geo
        self.rg = xtr.rg

        # Enumerate possible feature values from graph
        self.feature_names = feature_names
        self.feature_values = [set() for _ in self.feature_names]
        for n in self.rg.nodes:
            node_props = self.rg.nodes[n]
            for feat_idx, feat in enumerate(self.feature_names):
                self.feature_values[feat_idx].add(node_props[feat])
        self.feature_values = [list(f) for f in self.feature_values]

        self.dimension_names = []
        for feat_idx, (feat, feat_vals) in enumerate(
            zip(self.feature_names, self.feature_values)
        ):
            for feat_val in feat_vals:
                self.dimension_names.append("{}-{}".format(feat, str(feat_val)))

        if self._use_geo:
            for geo_zone in self.GEO_ZONES.keys():
                self.dimension_names.append(f"Region-{geo_zone}")

        # Pre-compute and cache feature vectors for fast retrieval
        self.angle_dist = lambda a, b: np.min(
            [2 * np.pi - np.abs(a - b), np.abs(a - b)], axis=0
        )
        self.geo_angles = np.array(list(self.GEO_ZONES.values())[1:])
        self._feat_vals = np.array([self._compute(n) for n in self.rg.nodes])

    def _node_geovec(self, node):
        """Compute geographical descriptor vector from a node"""
        node_props = self.rg.nodes[node]

        # Nodes have tuple properties 'start_latlon' and 'end_latlon' - average to get
        # an approximate lat, lon for this road segment
        node_latlon = (
            (node_props["start_latlon"][0] + node_props["end_latlon"][0]) / 2.0,
            (node_props["start_latlon"][1] + node_props["end_latlon"][1]) / 2.0,
        )

        # GPS delta vector
        cbd_delta = node_latlon - PORTO_CBD_LATLON

        # Distance from CBD to this road in km
        # cbd_dist = np.linalg.norm(cbd_delta / 360.0 * self.EARTH_CIRCUMFERNCE_KM)
        cbd_dist = geoid_dist(*node_latlon, *PORTO_CBD_LATLON)

        # Geo Zones are: Central, N, NE, ...
        geo_vec = np.zeros(len(self.GEO_ZONES))
        if cbd_dist <= self.GEO_ZONES["Central"]:
            geo_vec[0] = 1.0
        else:
            # Angle in radians of the vector from CBD to this road, where due east is 0, CCW is +ve
            cbd_angle = np.arctan2(cbd_delta[0], cbd_delta[1])
            geo_zone_idx = np.argmin(self.angle_dist(cbd_angle, self.geo_angles))
            geo_vec[geo_zone_idx + 1] = 1.0
        return geo_vec

    def _compute(self, node):
        """Compute a feature vector by reference to the road graph
        
        This method is quite slow
        """
        node_props = self.rg.nodes[node]
        sub_vecs = []
        for feat_idx, feat in enumerate(self.feature_names):
            vec = np.zeros(len(self.feature_values[feat_idx]))
            vec[self.feature_values[feat_idx].index(node_props[feat])] = 1.0
            sub_vecs.append(vec)

        if self._use_geo:
            # Add feature for geographic region
            sub_vecs.append(self._node_geovec(node))

        # The distances stored in the road graph are km
        phi = np.concatenate(sub_vecs) * node_props["distance"]
        return phi

    def __len__(self):
        """Get length of the feature vector
        
        Returns:
            (int): Length of this feature vector
        """
        _len = np.sum([len(f) for f in self.feature_values])
        if self._use_geo:
            _len += len(self.GEO_ZONES)
        return _len

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action
        
        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation
        
        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        try:
            return self._feat_vals[o1]
        except IndexError:
            warnings.warn(
                f"Requested φ({o1}, {a}, {o2}), however slice is out-of-bounds. This could be due to using padded rollouts, in which case you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.zeros(len(self))

    def feature_distance_metric(self, path_1, path_2, gamma=1.0):
        """Compute feature distance in km between two paths"""
        phi_bar_1 = self.expectation([path_1], gamma=gamma)
        phi_bar_2 = self.expectation([path_2], gamma=gamma)
        return np.linalg.norm(phi_bar_1 - phi_bar_2)


class PortoInference:
    """Container object to do inference with a single model"""

    def __init__(self, xtr, phi, reward_parameters):
        self.xtr = xtr
        self.phi = phi
        self.rg = xtr.rg.copy()
        self.reward_parameters = reward_parameters

        assert (
            np.max(reward_parameters) <= 0
        ), "Fast ML path inference only supported for negative reward parameters"

        # Store negative log likelihood (-ve state reward) for every state in the road graph
        self.state_nlls = -1.0 * (phi._feat_vals @ reward_parameters)
        for s, n in zip(xtr._states, xtr._nodes):
            self.rg.nodes[n]["nll"] = self.state_nlls[s]

        # Now, store the worst NLL (maximum) for each edge
        for e in self.rg.edges:
            n1, n2 = e
            self.rg.edges[e]["nll"] = max(
                self.rg.nodes[n1]["nll"], self.rg.nodes[n2]["nll"]
            )

    def ml_path(self, s1, sg):
        """Find Maximum Likelihood path between two states"""
        n1 = self.xtr._nodes[s1]
        ng = self.xtr._nodes[sg]
        node_path = nx.shortest_path(self.rg, source=n1, target=ng, weight="nll")
        return self.xtr.n2s_path(node_path)

    def plot_reward(self, ax=None, logscale=False, **kwargs):
        """Plot the reward function"""

        if ax is None:
            ax = plt.gca()

        # Reward is -ve of NLL
        state_rewards = -1.0 * self.state_nlls
        min_reward = np.min(state_rewards)
        max_reward = np.max(state_rewards)

        # Get colormap object
        cmap = plt.get_cmap("Reds_r")
        if logscale:
            # Regular log scale doesn't work as our values reach 0.0
            # norm = mpl.colors.LogNorm(vmin=min_reward, vmax=max_reward)

            # Instead, we use a Symetric Log Norm that provides a small linear norm
            # area close to zero - we chose some small threshold where this has effect
            lin_thresh = (np.abs(min_reward) - np.abs(max_reward)) / 250.0
            norm = mpl.colors.SymLogNorm(
                linthresh=lin_thresh, vmin=min_reward, vmax=max_reward
            )
        else:
            norm = mpl.colors.Normalize(vmin=min_reward, vmax=max_reward)

        # We use LineCollection for efficient rendering of many lines
        segments = []
        for state, node in zip(self.xtr._states, self.xtr._nodes):
            # Get the GPS coords of this state's road segment
            start_latlon = self.rg.nodes[node]["start_latlon"]
            end_latlon = self.rg.nodes[node]["end_latlon"]
            gps_coords = np.array(
                [[start_latlon[1], start_latlon[0]], [end_latlon[1], end_latlon[0]]]
            )
            segments.append(gps_coords)

        lc = LineCollection(segments, norm=norm, cmap=cmap)
        lc.set_array(state_rewards)
        ax.add_collection(lc)
        plt.sci(lc)

        return lc


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
