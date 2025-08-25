import numpy as np
import pandas as pd
import pyvista as pv

from meshmash import edges_to_lines


def get_hue_info(layer, hue, clim):
    if hue is None:
        return None, None
    if isinstance(hue, str):
        scalars = layer.nodes[hue].values
    elif isinstance(hue, (np.ndarray, pd.Series)):
        scalars = hue

    if clim is not None and isinstance(clim, tuple):
        return scalars, clim

    if clim is None:
        return scalars, None
    elif clim == "robust":
        low, high = np.nanpercentile(scalars, [2, 98])
    elif clim == "symmetric":
        low, high = np.nanmin(scalars), np.nanmax(scalars)
        val = max(abs(low), abs(high))
        low, high = -val, val
    clim = (low, high)
    return scalars, clim


class MorphPlotter(pv.Plotter):
    def add_mesh_layer(self, morph, layer_name="mesh", hue=None, clim=None, **kwargs):
        if (
            morph.has_layer(layer_name)
            and morph.get_layer(layer_name).is_spatially_valid
        ):
            mesh_layer = morph.get_layer(layer_name)
            mesh = pv.make_tri_mesh(mesh_layer.vertices, mesh_layer.faces)
            scalars, clim = get_hue_info(mesh_layer, hue, clim=clim)
            super().add_mesh(
                mesh,
                name=layer_name,
                scalars=scalars,
                clim=clim,
                scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
                **kwargs,
            )

    def add_graph_layer(self, morph, layer_name="graph", hue=None, clim=None, **kwargs):
        if (
            morph.has_layer(layer_name)
            and morph.get_layer(layer_name).is_spatially_valid
        ):
            graph_layer = morph.get_layer(layer_name)
            lines = edges_to_lines(graph_layer.facets_positional)
            line_poly = pv.PolyData(graph_layer.vertices, lines=lines)
            scalars, clim = get_hue_info(graph_layer, hue, clim)
            super().add_mesh(
                line_poly,
                # name=layer_name,
                scalars=scalars,
                clim=clim,
                scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
                **kwargs,
            )

    def add_point_layer(
        self, morph, layer_name="points", hue=None, clim=None, **kwargs
    ):
        if (
            morph.has_layer(layer_name)
            and morph.get_layer(layer_name).is_spatially_valid
        ):
            points_layer = morph.get_layer(layer_name)
            scalars, clim = get_hue_info(points_layer, hue, clim)

            super().add_points(
                points_layer.vertices,
                name=layer_name,
                scalars=scalars,
                clim=clim,
                scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
                **kwargs,
            )
