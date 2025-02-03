"""
Line cleaning utilities for fracture/lineament analysis.
"""

import math
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, split, nearest_points, unary_union
from shapely.validation import make_valid
from sklearn.cluster import DBSCAN

def angle_of_line(p1, p2):
    """Compute angle in radians of the line from p1->p2 using atan2(dy, dx)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def angle_difference(th1, th2):
    """Return absolute difference between two angles, normalized to [0, pi]."""
    diff = abs(th1 - th2)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff

def merge_lines_by_angle(G, angle_threshold_degs=10.0):
    """
    Merge lines at a node if they are nearly collinear.
    Returns the count of merges done in this pass.
    """
    angle_threshold = math.radians(angle_threshold_degs)
    merges_done = 0

    to_merge = []
    for node in list(G.nodes()):
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue

        angle_info = []
        for nb in neighbors:
            th = angle_of_line(node, nb)
            angle_info.append((nb, th))

        for i in range(len(angle_info)):
            for j in range(i+1, len(angle_info)):
                nb1, th1 = angle_info[i]
                nb2, th2 = angle_info[j]
                diff = angle_difference(th1, th2)
                if diff < angle_threshold:
                    to_merge.append((node, nb1, nb2))

    for (node, nb1, nb2) in to_merge:
        if G.has_edge(node, nb1) and G.has_edge(node, nb2):
            G.remove_edge(node, nb1)
            G.remove_edge(node, nb2)
            dist = np.hypot(nb2[0] - nb1[0], nb2[1] - nb1[1])
            if not G.has_edge(nb1, nb2):
                G.add_edge(nb1, nb2, length=dist)
            merges_done += 1

    return merges_done

def snap_line_endpoints(gdf, snap_tolerance=5.0):
    """Snap nearby line endpoints together."""
    endpoints = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            endpoints.extend([coords[0], coords[-1]])
        elif geom.geom_type == 'MultiLineString':
            for part in geom.geoms:
                coords = list(part.coords)
                endpoints.extend([coords[0], coords[-1]])

    if not endpoints:
        return gdf

    endpoints_np = np.array(endpoints)
    clustering = DBSCAN(eps=snap_tolerance, min_samples=1).fit(endpoints_np)
    clusters = clustering.labels_

    cluster_centroids = {}
    for idx, label in enumerate(clusters):
        if label not in cluster_centroids:
            cluster_centroids[label] = []
        cluster_centroids[label].append(endpoints_np[idx])
    for label in cluster_centroids:
        cluster_centroids[label] = np.mean(cluster_centroids[label], axis=0)

    snapped_points = {tuple(endpoints[idx]): tuple(cluster_centroids[label]) 
                     for idx, label in enumerate(clusters)}

    def snap_coords(coords):
        start = tuple(coords[0])
        end = tuple(coords[-1])
        new_start = snapped_points.get(start, start)
        new_end = snapped_points.get(end, end)
        return [new_start] + list(coords[1:-1]) + [new_end]

    snapped_geoms = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == 'LineString':
            new_coords = snap_coords(list(geom.coords))
            snapped_geoms.append(LineString(new_coords))
        elif geom.geom_type == 'MultiLineString':
            new_parts = []
            for part in geom.geoms:
                new_coords = snap_coords(list(part.coords))
                new_parts.append(LineString(new_coords))
            snapped_geoms.append(MultiLineString(new_parts))
    
    return gpd.GeoDataFrame(geometry=snapped_geoms, crs=gdf.crs)

def buffer_and_merge_lines(gdf, buffer_dist=2.0):
    """Buffer each line and merge overlapping ones."""
    buffers = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        geom = make_valid(geom)
        buf = geom.buffer(buffer_dist, join_style=2)
        if not buf.is_empty:
            buffers.append(buf)

    if not buffers:
        return gdf

    unioned_poly = unary_union(buffers)
    if unioned_poly.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    boundary = unioned_poly.boundary
    boundary_list = []
    if boundary.geom_type == "LineString":
        boundary_list = [boundary]
    elif boundary.geom_type == "MultiLineString":
        for part in boundary.geoms:
            boundary_list.append(part)
    elif boundary.geom_type == "GeometryCollection":
        for geom2 in boundary.geoms:
            if "LineString" in geom2.geom_type:
                boundary_list.append(geom2)

    return gpd.GeoDataFrame(geometry=boundary_list, crs=gdf.crs)

def lines_to_graph(gdf):
    """Convert a GeoDataFrame of lines to a NetworkX graph."""
    G = nx.Graph()
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            add_line_to_graph(geom, G)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                add_line_to_graph(part, G)
    return G

def add_line_to_graph(line, G):
    """Add a single line to the graph."""
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        G.add_node(p1, x=p1[0], y=p1[1])
        G.add_node(p2, x=p2[0], y=p2[1])
        G.add_edge(p1, p2, length=dist)

def remove_short_edges(G, min_length=10.0):
    """Remove edges shorter than min_length."""
    to_remove = []
    for (u, v, data) in list(G.edges(data=True)):
        if data['length'] < min_length:
            to_remove.append((u, v))
    G.remove_edges_from(to_remove)

def resolve_artificial_fragmentation(G):
    """Identify and merge chains of degree-2 nodes."""
    deg_dict = {n: G.degree(n) for n in G.nodes}
    node_neighbors_to_check = [n for n, deg in deg_dict.items() if deg != 2]

    chains_type1 = []
    chains_type2 = []

    def neighbors_degree_2(current_node, prev_node):
        nbrs = [nbr for nbr in G.neighbors(current_node) if nbr != prev_node]
        if len(nbrs) == 1 and deg_dict[nbrs[0]] == 2:
            return nbrs[0]
        return None

    def neighbors_degree_not_2(current_node, prev_node):
        nbrs = [nbr for nbr in G.neighbors(current_node) if nbr != prev_node]
        for nb in nbrs:
            if deg_dict[nb] != 2:
                return nb
        return None

    for origin_node in node_neighbors_to_check:
        neighs = list(G.neighbors(origin_node))
        neigh_deg2 = [n for n in neighs if deg_dict[n] == 2]
        for current_node in neigh_deg2:
            chain_path = [origin_node, current_node]
            prev_node = origin_node

            while True:
                next_node = neighbors_degree_2(current_node, prev_node)
                if not next_node:
                    break
                prev_node = current_node
                current_node = next_node
                chain_path.append(current_node)

            last_node = neighbors_degree_not_2(current_node, prev_node)
            if last_node:
                chain_path.append(last_node)

            chains_type1.append(chain_path)

    for origin_node in node_neighbors_to_check:
        neighs = list(G.neighbors(origin_node))
        neigh_deg_not_2 = [n for n in neighs if deg_dict[n] != 2]
        for current_node in neigh_deg_not_2:
            chain_path = [origin_node, current_node]
            chains_type2.append(chain_path)

    all_chains = chains_type1 + chains_type2

    unique_set = set()
    final_chains = []
    for chain in all_chains:
        t = tuple(chain)
        t_rev = tuple(reversed(t))
        if t_rev < t:
            t = t_rev
        if t not in unique_set:
            unique_set.add(t)
            final_chains.append(list(t))

    return final_chains

def graph_to_lines(G):
    """Convert graph edges to LineStrings."""
    lines = []
    for (u, v, data) in G.edges(data=True):
        lines.append(LineString([u, v]))
    return lines

def merge_connected_edges(G):
    """Merge connected edges into continuous lines."""
    merged_lines = []
    for component in nx.connected_components(G):
        segs = []
        subg = G.subgraph(component)
        for (u, v) in subg.edges():
            segs.append(LineString([u, v]))

        if not segs:
            continue

        unioned = unary_union(segs)
        if unioned.is_empty:
            continue
        elif unioned.geom_type == "LineString":
            merged_lines.append(unioned)
        elif unioned.geom_type == "MultiLineString":
            merged = linemerge(unioned)
            merged_lines.append(merged)
        else:
            merged_lines.append(unioned)
    return merged_lines

def simplify_lines(geom_list, tolerance=1.0):
    """Simplify lines using Douglas-Peucker algorithm."""
    simplified = []
    for geom in geom_list:
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            simp = geom.simplify(tolerance, preserve_topology=True)
            simplified.append(simp)
        elif geom.geom_type == "MultiLineString":
            parts = []
            for part in geom.geoms:
                part_simp = part.simplify(tolerance, preserve_topology=True)
                parts.append(part_simp)
            multi_simpl = unary_union(parts)
            simplified.append(multi_simpl)
        else:
            simplified.append(geom)
    return simplified