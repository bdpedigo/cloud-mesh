import numpy as np
import pandas as pd
import pyvista as pv

from meshmash import edges_to_lines


def get_hue_info(layer, hue, clim, cmap=None):
    if hue is None:
        return None, None
    if isinstance(hue, str):
        scalars = layer.nodes[hue].values

    elif isinstance(hue, (np.ndarray, pd.Series)):
        scalars = hue

    if cmap is not None and isinstance(cmap, dict):
        colors = np.array([cmap[val] for val in scalars])
        if colors.max() > 1:  # assume 0-255
            colors = colors / 255.0
        return colors, None

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

    if scalars.dtype.isin([np.float16]):
        scalars = scalars.astype(np.float32)
    return scalars, clim


def project_points(points, projection="identity"):
    if projection == "identity":
        return points
    elif projection == "pca-tall":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        projected_points = pca.fit_transform(points)
        projected_points = projected_points[:, [1, 0, 2]]
        return projected_points


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
                # name=layer_name,
                scalars=scalars,
                clim=clim,
                scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
                **kwargs,
            )

    def add_graph_layer(
        self,
        morph,
        layer_name="graph",
        hue=None,
        clim=None,
        projection="identity",
        **kwargs,
    ):
        if (
            morph.has_layer(layer_name)
            and morph.get_layer(layer_name).is_spatially_valid
        ):
            graph_layer = morph.get_layer(layer_name)
            lines = edges_to_lines(graph_layer.facets_positional)
            vertices = project_points(graph_layer.vertices, projection=projection)
            line_poly = pv.PolyData(vertices, lines=lines)
            if clim is None:
                scalars, clim = get_hue_info(graph_layer, hue, clim)
            else:
                scalars, _ = get_hue_info(graph_layer, hue, clim)
            super().add_mesh(
                line_poly,
                # name=layer_name,
                scalars=scalars,
                clim=clim,
                # scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
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
                # name=layer_name,
                scalars=scalars,
                clim=clim,
                scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
                **kwargs,
            )

    def add_sphere_point_layer(
        self,
        morph,
        layer_name="points",
        hue=None,
        clim=None,
        point_size=50,
        cmap=None,
        **kwargs,
    ):
        if (
            morph.has_layer(layer_name)
            and morph.get_layer(layer_name).is_spatially_valid
        ):
            points_layer = morph.get_layer(layer_name)
            scalars, clim = get_hue_info(points_layer, hue, clim, cmap=cmap)
            points = points_layer.vertices
            cloud = pv.PolyData(points)
            cloud["scalars"] = scalars
            # cloud["rgb"] = np.array(colors) / 255.0
            sphere = pv.Sphere(radius=point_size)
            glyphs = cloud.glyph(scale=False, geom=sphere)

            # print(glyphs["scalars"])
            # can also end up as "GlyphVector"
            if isinstance(cmap, dict):
                cmap = None
                kwargs["rgb"] = True
            self.add_mesh(
                glyphs,
                # scalars="scalars",
                scalars="GlyphVector",
                # rgb=True,
                clim=clim,
                scalar_bar_args={"title": hue} if isinstance(hue, str) else None,
                cmap=cmap,
                **kwargs,
            )
