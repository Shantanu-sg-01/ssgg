import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.debugger import set_trace
import pandas as pd

def calc_centroid_triangle(mesh_cart_nodes, n1, n2, n3):
  x1 = mesh_cart_nodes[n1][1]
  x2 = mesh_cart_nodes[n2][1]
  x3 = mesh_cart_nodes[n3][1]
  y1 = mesh_cart_nodes[n1][2]
  y2 = mesh_cart_nodes[n2][2]
  y3 = mesh_cart_nodes[n3][2]

  return [(x1 + x2 + x3)/3, (y1 + y2 + y3)/3]

def grid_generator(radius,
                   sector_angle,
                   number_of_radial_cells,
                   number_of_angular_cells,
                   radial_expansion_ratio,
                   angular_expansion_ratio,
                   radius_0,
                   sector_angle_0):
  """
  The main grid generator function
  """
  d_alpha = angular_expansion_ratio
  d_r = radial_expansion_ratio
  alpha = np.deg2rad(sector_angle)
  R = radius
  N_alpha = number_of_angular_cells
  N_r = number_of_radial_cells

  # calculation of a in ar^n for radial and angular progression
  if abs(d_alpha - 1.0) > 1e-3:
    if d_alpha < 1:
      a_alpha = alpha * (1 - d_alpha) / (1 - d_alpha**(N_alpha))
    elif d_alpha > 1:
      a_alpha = alpha * (d_alpha - 1) / (d_alpha**(N_alpha) - 1)
  else:
    a_alpha = alpha / N_alpha

  if abs(d_r - 1.0) > 1e-3:
    a_r = R * (1 - d_r) / (1 - d_r**N_r)
  else:
    a_r = R / N_r

  # calculating a, ar, ar^2 ... for angular and radial progression series
  alpha_inc = np.array([np.deg2rad(sector_angle_0)])
  r_inc = np.array([])

  for i in np.arange(N_alpha):
    alpha_inc = np.append(alpha_inc, a_alpha*d_alpha**(i))
  for i in np.arange(N_r):
    r_inc = np.append(r_inc, a_r*d_r**(i))

  # cumsum to find the r and angle of each grid curve
  alpha_range = np.cumsum(alpha_inc)
  r_range = np.cumsum(r_inc)

  # mesh generation
  mesh = {
      'Radius':R,
      'Angle':alpha,
      'Radial_Cells':N_r,
      'Angular_Cells':N_alpha,
      'Radial_Expansion_Ratio':d_r,
      'Angular_Expansion_Ratio':d_alpha,
      'Initial_Angle':sector_angle_0,
      'Final_Angle':sector_angle_0 + sector_angle,
      'nodes_sph':np.empty((0, 3)),
      'nodes_cart':np.empty((0, 3)),
  }
  mesh['nodes_sph'] = np.vstack([mesh['nodes_sph'], [1, radius_0, sector_angle_0]])
  mesh['nodes_cart'] = np.vstack([mesh['nodes_cart'], [1, radius_0*np.cos(sector_angle_0*np.pi/180.), radius_0*np.sin(sector_angle_0*np.pi/180)]])
  node_number = 2
  for angle in alpha_range:
    for rad in r_range:
      mesh['nodes_sph'] = np.vstack([mesh['nodes_sph'], [node_number, rad, angle]])
      mesh['nodes_cart'] = np.vstack([mesh['nodes_cart'], [node_number, rad*np.cos(angle), rad*np.sin(angle)]])
      node_number += 1

  radial_edge_nodes = np.empty((0, 6))
  angular_edge_nodes = np.empty((0, 6))
  diagonal_edge_nodes = np.empty((0, 6))
  for i, angle in enumerate(alpha_range):
    radial_edge_nodes = np.vstack([radial_edge_nodes, [1, i*N_r+2,
                                                       mesh['nodes_cart'][1-1][1],mesh['nodes_cart'][i*N_r+2-1][1],
                                                       mesh['nodes_cart'][1-1][2],mesh['nodes_cart'][i*N_r+2-1][2]
                                                       ]])
    for nr in np.arange(N_r-1):
      radial_edge_nodes = np.vstack([radial_edge_nodes, [i*N_r+2+nr, i*N_r+2+nr+1,
                                                         mesh['nodes_cart'][i*N_r+2+nr-1][1],mesh['nodes_cart'][i*N_r+2+nr+1-1][1],
                                                         mesh['nodes_cart'][i*N_r+2+nr-1][2],mesh['nodes_cart'][i*N_r+2+nr+1-1][2]]])


  for i in np.arange(N_alpha):
    for nr in np.arange(N_r):
      angular_edge_nodes = np.vstack([angular_edge_nodes, [i*N_r+2+nr, i*N_r+2+nr+N_r,
                                                           mesh['nodes_cart'][i*N_r+2+nr-1][1],mesh['nodes_cart'][i*N_r+2+nr+N_r-1][1],
                                                           mesh['nodes_cart'][i*N_r+2+nr-1][2],mesh['nodes_cart'][i*N_r+2+nr+N_r-1][2]
                                                           ]])

  for i in np.arange(N_alpha):
    for nr in np.arange(N_r-1):
      diagonal_edge_nodes = np.vstack([diagonal_edge_nodes, [i*N_r+2+nr, i*N_r+2+nr+N_r+1,
                                                             mesh['nodes_cart'][i*N_r+2+nr-1][1],mesh['nodes_cart'][i*N_r+2+nr+N_r+1-1][1],
                                                             mesh['nodes_cart'][i*N_r+2+nr-1][2],mesh['nodes_cart'][i*N_r+2+nr+N_r+1-1][2]
                                                             ]])

  mesh['radial_edge_nodes'] = radial_edge_nodes  # n1, n2, x1, x2, y1, y2
  mesh['angular_edge_nodes'] = angular_edge_nodes # n1, n2, x1, x2, y1, y2
  mesh['diagonal_edge_nodes'] = diagonal_edge_nodes # n1, n2, x1, x2, y1, y2

  elements = np.empty((0,6))
  element_number = 1
  for edge in mesh['radial_edge_nodes']:
    n1, n2 = int(edge[0]-1), int(edge[1]-1)
    if n1 < N_r * N_alpha and n2 <= N_r * N_alpha:
      n3 = n2 + N_r
      x_elem_cent, y_elem_cent = calc_centroid_triangle(mesh['nodes_cart'], n1, n2, n3)
      elements = np.vstack([elements, [element_number, n1+1, n2+1, n3+1, x_elem_cent, y_elem_cent]])
      element_number += 1
    if n2 > N_r and n1 !=0:
      n4 = n1 - N_r
      x_elem_cent, y_elem_cent = calc_centroid_triangle(mesh['nodes_cart'], n1, n2, n4)
      elements = np.vstack([elements, [element_number, n1+1, n2+1, n4+1, x_elem_cent, y_elem_cent]])
      element_number += 1
  mesh['elements'] = elements
  textstr = '\n'.join((
    'Elements #%i to #%i' % (mesh['elements'][0,0],mesh['elements'][-1,0]),
    'Radius = %.2f' % (mesh['Radius'],),
    'Radial Cells = %i' % (mesh['Radial_Cells'],),
    'Radial Expansion Ratio = %.2f' % (mesh['Radial_Expansion_Ratio'],),
    'Angle = %.2f' % (np.rad2deg(mesh['Angle']), ),
    'Angular Cells = %i' % (mesh['Angular_Cells'],),
    'Angular Expansion Ratio = %.2f' % (mesh['Angular_Expansion_Ratio'],),
  ))
  mesh['textstr'] = textstr
  return mesh

def combine_mesh(mesh1, mesh2):
  if mesh1['Radius'] != mesh2['Radius']:
    raise ValueError("The radii of the two meshes must be the same.")
  if mesh1['Radial_Cells'] != mesh2['Radial_Cells']:
    raise ValueError("The number of radial cells in the two meshes must be the same.")
  if mesh1['Final_Angle'] != mesh2['Initial_Angle']:
    raise ValueError("The initial angle of the first mesh must be equal to the final angle of the previous mesh.")
  if mesh1['Radial_Expansion_Ratio'] != mesh2['Radial_Expansion_Ratio']:
    raise ValueError("The radial expansion ratio of the first mesh must be equal to the radial expansion ratio of the previous mesh.")
  mesh = {
      'nodes_sph':np.empty((0, 3)),
      'nodes_cart':np.empty((0, 3)),
      'elements':np.array([]),
      'edges':np.array([]),
  }
  N_r = mesh1['Radial_Cells']
  N_alpha = mesh1['Angular_Cells']
  # mesh2['nodes_sph'][:N_r+1] = np.full((N_r+1, 3), np.nan)
  # mesh2['nodes_cart'][:N_r+1] = np.full((N_r+1, 3), np.nan)
  mesh2['nodes_sph'][:N_r+1,0] = np.full((N_r+1, ), np.nan)
  mesh2['nodes_cart'][:N_r+1, 0] = np.full((N_r+1, ), np.nan)
  mesh1_last_node_number = mesh1['nodes_cart'][-1, 0]
  # set_trace()
  mesh2['nodes_sph'][:,0] += mesh1_last_node_number - N_r - 1
  mesh2['nodes_cart'][:,0] += mesh1_last_node_number - N_r - 1
  mesh['nodes_sph'] = np.vstack([mesh1['nodes_sph'], mesh2['nodes_sph']])
  mesh['nodes_cart'] = np.vstack([mesh1['nodes_cart'], mesh2['nodes_cart']])

  mesh1_last_element_number = mesh1['elements'][-1, 0]
  mesh2_elements = mesh2['elements']
  mesh2_elements[:,0] += mesh1_last_element_number
  for ne, elem in enumerate(mesh2_elements):
    if elem[1] != 1:
      elem[1] += N_r * N_alpha
    elem[2] += N_r * N_alpha
    elem[3] += N_r * N_alpha
    mesh2_elements[ne, 1] = elem[1]
    mesh2_elements[ne, 2] = elem[2]
    mesh2_elements[ne, 3] = elem[3]
  mesh2_elements[:,1] += 0
  mesh2_elements[:,2] += 0
  mesh['elements'] = np.vstack([mesh1['elements'], mesh2_elements])

  # mesh2['radial_edge_nodes'][:N_r-1] = np.full((N_r-1, 6), np.nan)
  mesh2['radial_edge_nodes'][:,0] = mesh2['radial_edge_nodes'][:,0] + mesh1_last_node_number
  mesh2['radial_edge_nodes'][:,1] = mesh2['radial_edge_nodes'][:,1] + mesh1_last_node_number
  mesh['radial_edge_nodes'] = np.vstack([mesh1['radial_edge_nodes'], mesh2['radial_edge_nodes']])
  mesh2['angular_edge_nodes'][:,0] = mesh2['angular_edge_nodes'][:,0] + mesh1_last_node_number - N_r - 1
  mesh2['angular_edge_nodes'][:,1] = mesh2['angular_edge_nodes'][:,1] + mesh1_last_node_number- N_r - 1
  mesh['angular_edge_nodes'] = np.vstack([mesh1['angular_edge_nodes'], mesh2['angular_edge_nodes']])

  mesh2['diagonal_edge_nodes'][:,0] = mesh2['diagonal_edge_nodes'][:,0] + mesh1_last_node_number - N_r - 1
  mesh2['diagonal_edge_nodes'][:,1] = mesh2['diagonal_edge_nodes'][:,1] + mesh1_last_node_number - N_r - 1
  mesh['diagonal_edge_nodes'] = np.vstack([mesh1['diagonal_edge_nodes'], mesh2['diagonal_edge_nodes']])

  mesh2_str_lines = mesh2['textstr'].splitlines()
  mesh2['textstr'] = '\n'.join(mesh2_str_lines[1:])
  mesh['textstr'] = '\n'.join((
    mesh1['textstr'],
    '',
    'Elements #%i to #%i' % (mesh2['elements'][0,0],mesh2['elements'][-1,0]),
    mesh2['textstr']
  ))
  mesh['edges'] = np.vstack([mesh1['radial_edge_nodes'], 
                             mesh2['radial_edge_nodes'], 
                             mesh1['angular_edge_nodes'], 
                             mesh2['angular_edge_nodes'],
                             mesh1['diagonal_edge_nodes'], 
                             mesh2['diagonal_edge_nodes']])
  mesh["Radius"] = mesh1['Radius']
  mesh["Radial_Cells"] = mesh1['Radial_Cells']
  mesh["Angular_Cells"] = mesh1['Angular_Cells'] + mesh2["Angular_Cells"]
  mesh["Radial_Expansion_Ratio"] = mesh1['Radial_Expansion_Ratio']
  mesh["Sector_Angle"] = np.rad2deg(mesh1['Angle'] + mesh2['Angle'])
  return mesh

def plot_grid(mesh, plotting_inputs):
  """
  function plots the grid.
  plotting_inputs dictionary gives freedom to the user to pass plotting options
  from the widgets.
  plotting_input = {
    "node_color":Node_Color,
    "edge_color":Edge_Color,
    "edge_linewidth":Edge_Linewidth,
    "show_axes":Show_Axes,
    "show_node_numbers":Show_Node_Numbers,
    "show_element_numbers":Show_Element_Numbers,
    "save_figure":Save_Figure
  }
  """
  x_nodes = mesh['nodes_cart'][:, 1]
  y_nodes = mesh['nodes_cart'][:, 2]
  x_cent = mesh['elements'][:, 4]
  y_cent = mesh['elements'][:, 5]
  colors = {'Blue': 'b', 'Red': 'r', 'Black': 'k'}
  c_nodes = colors[plotting_inputs['node_color']]
  c_edges = colors[plotting_inputs['edge_color']]
  lw = plotting_inputs['edge_linewidth']

  fig, ax = plt.subplots(figsize=(12,9))
  plt.scatter(x_nodes, y_nodes, facecolors='none', edgecolors=c_nodes, label='Nodes', s=25)

  for edge in mesh['radial_edge_nodes']:
    plt.plot(edge[2:4], edge[4:6], c_edges, linewidth=lw)
  for edge in mesh['angular_edge_nodes']:
    plt.plot(edge[2:4], edge[4:6], c_edges, linewidth=lw)
  for edge in mesh['diagonal_edge_nodes']:
    plt.plot(edge[2:4], edge[4:6], c_edges, linewidth=lw)


  # Add node labels (optional, can make plot cluttered for large meshes)
  if plotting_inputs['show_node_numbers']:
    labels = mesh['nodes_cart'][:, 0].astype(int)
    for xi, yi, label in zip(x_nodes, y_nodes, labels):
      if label > 0:
        plt.text(xi, yi, str(label), fontsize=12, ha='center', va='bottom')

  if plotting_inputs['show_element_numbers']:
    # Add element labels (optional, can make plot cluttered for large meshes)
    elem_labels = mesh['elements'][:, 0].astype(int)
    for xi, yi, elem_label in zip(x_cent, y_cent, elem_labels):
        plt.text(xi, yi, str(elem_label), fontsize=12, color='r', ha='center', va='center')
  if plotting_inputs['show_axes']:
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Generated Grid')
  plt.axis('equal')
  plt.grid(True)
  plt.legend()

  props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)

  # place a text box in upper left in axes coords
  ax.text(0.7, 0.95, mesh['textstr'], transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
  if plotting_inputs['save_figure']:
    plt.savefig("figure.png")
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

  # edges
  data_titles = ["Node 1", "Node 2", "x1", "x2", "y1", "y2"]
  df = pd.DataFrame(mesh['edges'], columns=data_titles)
  filename = "edges" + Filename_Suffix + ".xlsx"
  df.to_excel(filename, index=False)

