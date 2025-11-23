import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calc_centroid_triangle(mesh_cart_nodes, n1, n2, n3):
  x1 = mesh_cart_nodes[n1-1][1]
  x2 = mesh_cart_nodes[n2-1][1]
  x3 = mesh_cart_nodes[n3-1][1]
  y1 = mesh_cart_nodes[n1-1][2]
  y2 = mesh_cart_nodes[n2-1][2]
  y3 = mesh_cart_nodes[n3-1][2]

  return [(x1 + x2 + x3)/3, (y1 + y2 + y3)/3]


def full_circular_grid(radius,
                       number_of_radial_cells,
                       number_of_angular_cells,
                       radial_expansion_ratio,
                       angular_expansion_ratio,
                       elliptic_grid_nos,
                       elliptic_radial_decrement_fraction
                       ):
    d_alpha = angular_expansion_ratio
    d_r = radial_expansion_ratio
    R = radius
    N_alpha = number_of_angular_cells
    N_r = number_of_radial_cells
    alpha = 2*np.pi

    # calculation of a in ar^n for radial and angular progression
    if abs(d_alpha - 1.0) > 1e-3:
        if d_alpha < 1:
            a_alpha = alpha * (1 - d_alpha) / (1 - d_alpha ** (N_alpha))
        elif d_alpha > 1:
            a_alpha = alpha * (d_alpha - 1) / (d_alpha ** (N_alpha) - 1)
    else:
        a_alpha = alpha / N_alpha

    if abs(d_r - 1.0) > 1e-3:
        a_r = R * (1 - d_r) / (1 - d_r ** N_r)
    else:
        a_r = R / N_r

    # calculating a, ar, ar^2 ... for angular and radial progression series
    alpha_inc = np.array([])
    r_inc = np.array([])

    for i in np.arange(N_alpha):
        alpha_inc = np.append(alpha_inc, a_alpha * d_alpha ** (i))
    for i in np.arange(N_r):
        r_inc = np.append(r_inc, a_r * d_r ** (i))

    # cumsum to find the r and angle of each grid curve
    alpha_range = np.cumsum(alpha_inc)
    alpha_range = np.insert(alpha_range, 0, 0)
    alpha_range = alpha_range[:-1]
    r_range = np.cumsum(r_inc)

    # mesh generation
    mesh = {
        'Radius': R,
        'Angle': alpha,
        'Radial_Cells': N_r,
        'Angular_Cells': N_alpha,
        'Radial_Expansion_Ratio': d_r,
        'Angular_Expansion_Ratio': d_alpha,
        'Initial_Angle': 0,
        'Final_Angle': 360,
        'nodes_sph': np.empty((0, 3)),
        'nodes_cart': np.empty((0, 3)),
    }
    radius_0 = 0
    mesh['nodes_sph'] = np.vstack([mesh['nodes_sph'], [1, 0, 0]])
    mesh['nodes_cart'] = np.vstack([mesh['nodes_cart'], [1, 0, 0]])

    node_number = 2
    elements = np.empty((0, 6))
    element_number = 1
    for r_id, rad in enumerate(r_range):
        for a_id, angle in enumerate(alpha_range):
            if r_id not in elliptic_grid_nos:
                mesh['nodes_sph'] = np.vstack([mesh['nodes_sph'], [node_number, rad, angle]])
                mesh['nodes_cart'] = np.vstack(
                    [mesh['nodes_cart'], [node_number, rad * np.cos(angle), rad * np.sin(angle)]])
            else:
                a = rad
                if r_id > 0:
                    b = rad - elliptic_radial_decrement_fraction * (rad - r_range[r_id-1])
                else:
                    b = rad - elliptic_radial_decrement_fraction * rad
                mesh['nodes_sph'] = np.vstack([mesh['nodes_sph'], [node_number, rad, angle]])
                mesh['nodes_cart'] = np.vstack(
                    [mesh['nodes_cart'], [node_number, a * np.cos(angle), b * np.sin(angle)]])
            if r_id == 0 and a_id < N_alpha - 1:
                n1, n2, n3 = 1, a_id+2, a_id+3
                x_elem_cent, y_elem_cent = 0, 0
                elements = np.vstack([elements, [element_number, n1, n2, n3, x_elem_cent, y_elem_cent]])
                element_number += 1

            elif r_id == 0 and a_id == N_alpha - 1:
                n1, n2, n3 = 1, a_id+2, 2
                x_elem_cent, y_elem_cent = 0, 0
                elements = np.vstack([elements, [element_number, n1, n2, n3, x_elem_cent, y_elem_cent]])
                element_number += 1

            elif a_id < (N_alpha - 1):
                quads = [1 + (r_id - 1) * N_alpha + a_id + 1,
                         1 + (r_id - 1) * N_alpha + a_id + 2,
                         1 + (r_id) * N_alpha + a_id + 1,
                         1 + (r_id) * N_alpha + a_id + 2]
                n1, n2, n3 = quads[0], quads[2], quads[3]
                x_elem_cent, y_elem_cent = 0, 0
                elements = np.vstack([elements, [element_number, n1, n2, n3, x_elem_cent, y_elem_cent]])
                element_number += 1

                n1, n2, n3 = quads[0], quads[3], quads[1]
                x_elem_cent, y_elem_cent = 0, 0
                elements = np.vstack([elements, [element_number, n1, n2, n3, x_elem_cent, y_elem_cent]])
                element_number += 1

            elif a_id == (N_alpha - 1):
                quads = [1 + (r_id - 1) * N_alpha + a_id + 1,
                         1 + (r_id - 1) * N_alpha + 1,
                         1 + (r_id) * N_alpha + a_id + 1,
                         1 + (r_id) * N_alpha + 1       ]

                n1, n2, n3 = quads[0], quads[2], quads[3]
                x_elem_cent, y_elem_cent = 0, 0
                elements = np.vstack([elements, [element_number, n1, n2, n3, x_elem_cent, y_elem_cent]])
                element_number += 1
                n1, n2, n3 = quads[0], quads[3], quads[1]
                x_elem_cent, y_elem_cent = 0, 0
                elements = np.vstack([elements, [element_number, n1, n2, n3, x_elem_cent, y_elem_cent]])
                element_number += 1
            else:
                print([r_id, a_id])
            node_number += 1

    for n, elem in enumerate(elements):
        n1, n2, n3 = int(elem[1]), int(elem[2]), int(elem[3])
        x_elem_cent, y_elem_cent = calc_centroid_triangle(mesh['nodes_cart'], n1, n2, n3)
        elements[n, 1] = n1
        elements[n, 2] = n2
        elements[n, 3] = n3
        elements[n, 4] = x_elem_cent
        elements[n, 5] = y_elem_cent
    mesh['elements'] = elements
    return mesh


def plot_grid(mesh):
    x_nodes = mesh['nodes_cart'][:, 1]
    y_nodes = mesh['nodes_cart'][:, 2]
    x_cent = mesh['elements'][:, 4]
    y_cent = mesh['elements'][:, 5]
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.scatter(x_nodes, y_nodes, facecolors='none', edgecolors='r', label='Nodes', s=25)
    # Add node labels (optional, can make plot cluttered for large meshes)
    show_node_numbers = False
    if show_node_numbers:
        labels = mesh['nodes_cart'][:, 0].astype(int)
        for xi, yi, label in zip(x_nodes, y_nodes, labels):
            if label > 0:
                plt.text(xi, yi, str(label), fontsize=12, ha='center', va='bottom')
    show_element_numbers = False
    if show_element_numbers:
        # Add element labels (optional, can make plot cluttered for large meshes)
        elem_labels = mesh['elements'][:, 0].astype(int)
        for xi, yi, elem_label in zip(x_cent, y_cent, elem_labels):
            plt.text(xi, yi, str(elem_label), fontsize=12, color='r', ha='center', va='center')

    for elem in mesh['elements']:
        plt.plot([x_nodes[int(elem[1])-1], x_nodes[int(elem[2])-1]],
                 [y_nodes[int(elem[1])-1], y_nodes[int(elem[2])-1]],
                 'k',
                 linewidth=0.7)
        plt.plot([x_nodes[int(elem[2])-1], x_nodes[int(elem[3])-1]],
                 [y_nodes[int(elem[2])-1], y_nodes[int(elem[3])-1]],
                 'k',
                 linewidth=0.7)
    plt.savefig('grid.svg')
    plt.show()


def write_mesh(mesh, Filename_Suffix):
  """
  the output generator function.
  """

  # cartesian nodes
  data_titles = ["Node Number", "x coordinate", "y coordinate"]
  df = pd.DataFrame(mesh['nodes_cart'], columns=data_titles)
  df_filtered = df.dropna(subset=["Node Number"])
  filename = "cartesian_nodes" + Filename_Suffix + ".xlsx"
  df_filtered.to_excel(filename, index=False)

  # elements
  data_titles = ["Element Number", "Node 1", "Node 2", "Node 3", "Centroid x coord.", "Centroid y coord"]
  df = pd.DataFrame(mesh['elements'], columns=data_titles)
  df_filtered = df.dropna(subset=["Element Number"])
  filename = "elements" + Filename_Suffix + ".xlsx"
  df_filtered.to_excel(filename, index=False)


if __name__ == "__main__":
    mesh1 = full_circular_grid(1,
                               10,
                               20,
                               1.0,
                               1.0,
                               [3, 6],
                               0.5)
    plot_grid(mesh1)
    write_mesh(mesh1, "grid_20_25")
    print('Done')


