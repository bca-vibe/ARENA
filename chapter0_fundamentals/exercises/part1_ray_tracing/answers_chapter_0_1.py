# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1]) # Set y directions spaced accordingly
    rays[:, 1, 0] = 1 # Set x direction to 1
    return rays


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)

# %% 
ray = t.tensor([[0.0, 0, 0], [1, 1, 0]])
print(ray)

segment = t.tensor([[0.0, 0, 0], [0, 2, 0]])
print(segment)

lin_trans = t.tensor([[1.0, 0], [1, 2]])
sol = t.tensor([0.0, 2])

print(t.linalg.solve(lin_trans, sol))

# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # Remove z coordinates
    ray_2d = ray[:,:2]
    segment_2d = segment[:,:2]
    # Get direction vectors
    d_ray = ray_2d[1] # Could also parse O and D separately like O, D = ray
    d_seg = segment_2d[0] - segment_2d[1] # Could also parse into pieces first like L_1, L_2 = segment
    lin_trans = t.stack([d_ray, d_seg], dim = 1)
    res = segment_2d[0]
    # Solve equation (return False if no solution)
    try:
        sol = t.linalg.solve(lin_trans, res)
    except RuntimeError:
        return False
    
    # If there is a solution, check the soln is in the correct range for there to be an intersection
    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)
    
tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    # Get rid of the unnecessary z dimension
    rays = rays[..., :2]
    segments = segments[..., :2]

    # Copy to match dimensionality
    nrays = rays.shape[0]
    nsegments = segments.shape[0]
    rays = einops.repeat(rays, "nrays r c -> nrays nsegments r c", nsegments = nsegments)
    segments = einops.repeat(segments, "nsegments r c -> nrays nsegments r c", nrays = nrays)

    # Construct matrix and solution of linear equations
    O, D = rays.unbind(dim = 2)
    L1, L2 = segments.unbind(dim = 2)
    M = t.stack([D, L1 - L2], dim = -1)
    res = L1 - O

    # Check for singular matrices and replace with identity matrix
    is_singular = t.linalg.det(M).abs() < 1e-8
    M[is_singular] = t.eye(2)

    # Solve the systems of equations
    sol = t.linalg.solve(M, res)
    u, v = sol.unbind(dim = 2)
    return t.any((u >= 0) & (v >= 0) & (v <= 1) & (~is_singular), dim = 1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%
D_toy = t.tensor([1., 2.])
L_toy = t.tensor([3., 4.])
print(t.stack([D_toy, L_toy], dim=0))   # rows:    [[1,2],[3,4]]
print(t.stack([D_toy, L_toy], dim=-1))  # columns: [[1,3],[2,4]]

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    rays = t.zeros((num_pixels_y * num_pixels_z, 2, 3), dtype = t.float32)
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays[:, 1, 0] = 1 # Set x direction to 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %%
A = t.tensor([1.0, 0.0, 0.0])
B = t.tensor([1.0, 1.0, 0.0])
C = t.tensor([1.0, 1.0, 1.0])
O = t.tensor([0.0, 0.0, 0.0])
D = t.tensor([1.0, 0.0, 0.0])

Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    M = t.stack([-D, B - A, C - A], dim = -1)
    vec = O - A

    # Check for singular matrix implying parallel plane + ray
    if t.linalg.det(M).abs() < 1e-8:
        return False
    s, u, v = t.linalg.solve(M, vec)

    return s.item() >= 0 and u.item() >= 0 and v.item() >= 0 and (u.item() + v.item() <= 1)

print(triangle_ray_intersects(A, B, C, O, D))
tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    # Number of rays
    nrays = rays.shape[0]

    # Capture the O, D from rays
    O, D = rays.unbind(dim = 1)

    # Broadcast the triangle to match rays dimensions, then grab the vertices
    triangle = einops.repeat(triangle, "p c -> nrays p c", nrays = nrays)
    A, B, C = triangle.unbind(dim = 1)

    # Get a tensor full of the systems of equations
    Ms = t.stack([-D, B - A, C - A], dim = -1)
    vecs = O - A

    # Find parallel rays and deal with them
    is_singular = t.linalg.det(Ms).abs() < 1e-8
    Ms[is_singular] = t.eye(3)

    # Compute solutions
    sol = t.linalg.solve(Ms, vecs)
    s, u, v = sol.unbind(dim = 1)
    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 100
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, " nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(-1)

    mat = t.stack([- D, B - A, C - A])

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %% 
# Load Pikachu! 
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    # Number of rays
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    # Match the dimensions
    rays = einops.repeat(rays, "nrays rayPoints coords -> nrays ntriangles rayPoints coords", ntriangles = ntriangles)
    triangles = einops.repeat(triangles, "ntriangles rayPoints coords -> nrays ntriangles rayPoints coords", nrays = nrays)

    # Capture the O, D from rays
    O, D = rays.unbind(dim = 2)

    # Grab the triangle vertices
    A, B, C = triangles.unbind(dim = 2)

    # Get a tensor full of the systems of equations
    Ms = t.stack([-D, B - A, C - A], dim = -1)
    vecs = O - A

    # Find parallel rays and deal with them
    is_singular = t.linalg.det(Ms).abs() < 1e-8
    Ms[is_singular] = t.eye(3)

    # Compute solutions
    sol = t.linalg.solve(Ms, vecs)
    s, u, v = sol.unbind(dim = 2)

    # Replace invalid solutions (non-intersections) with infinity
    s[(s < 0) | (u < 0) | (v < 0) | (u + v > 1) | is_singular] = float('inf')

    # Get the closest triangle distance for each ray in nrays
    # s is in units of norm(D)
    dists = t.min(s, dim = 1).values 

    return dists


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()



# %%
