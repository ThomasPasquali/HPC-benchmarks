from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

def add_zoom_inset(
  ax, zoom_region, inset_position=(0.6, 0.6, 0.3, 0.3),
  draw_rect=True, rect_kwargs=None, zoom_ax_kwargs=None,
  rect_scaling=1.0, rect_region=None,
):
  """
  Adds a zoomed inset with full size and position control using inset_axes.

  Parameters:
  - ax: matplotlib.axes.Axes
          The main axes.
  - zoom_region: tuple (x1, x2, y1, y2)
          Limits of the region to zoom into.
  - inset_position: tuple (x0, y0, width, height)
          Inset position in axes fraction coordinates (not figure coords).
  - draw_rect: bool
          Draw a dashed rectangle on the main plot to show zoom region.
  - rect_kwargs: dict
          Styling for the zoom rectangle.
  - zoom_ax_kwargs: dict
          Dict of method calls on the inset axes.
  - rect_scaling: float
          Scaling factor for the zoom rectangle size (default=1.0).
  - rect_region: tuple (x, y, w, h) (default=None)
          Overwrite the default rect coords
  
  Returns:
  - axins: The inset axes object.
  """
  rect_kwargs = rect_kwargs or {'edgecolor': 'black', 'linestyle': 'dashed', 'linewidth': 1}
  zoom_ax_kwargs = zoom_ax_kwargs or {}

  # Create inset axes
  bbox = inset_position
  axins = inset_axes(
    ax,
    width="100%", height="100%",
    bbox_to_anchor=bbox,
    bbox_transform=ax.transAxes,
    loc='lower left',
    borderpad=0
  )

  # Set zoom limits
  x1, x2, y1, y2 = zoom_region
  axins.set_xlim(x1, x2)
  axins.set_ylim(y1, y2)

  # Copy each line fully
  for line in ax.get_lines():
    axins.plot(
      line.get_xdata(),
      line.get_ydata(),
      color=line.get_color(),
      linestyle=line.get_linestyle(),
      linewidth=line.get_linewidth(),
      marker=line.get_marker(),
      markersize=line.get_markersize(),
      markeredgecolor=line.get_markeredgecolor(),
      markerfacecolor=line.get_markerfacecolor(),
      alpha=line.get_alpha(),
      label=line.get_label(),
      zorder=line.get_zorder(),
    )

  # Apply additional inset customizations
  for method, args in zoom_ax_kwargs.items():
    if isinstance(args, dict):
      getattr(axins, method)(**args)
    elif isinstance(args, tuple):
      getattr(axins, method)(*args)
    else:
      getattr(axins, method)(args)

  # Draw rectangle on main plot with scaling    
  if draw_rect:
    if rect_region:
      rect = Rectangle((rect_region[0], rect_region[1]), rect_region[2], rect_region[3], fill=False, **rect_kwargs)
    else:
      x_center = (x1 + x2) / 2
      y_center = (y1 + y2) / 2
      width = (x2 - x1) * rect_scaling
      height = (y2 - y1) * rect_scaling
      rect_x = x_center - width / 2
      rect_y = y_center - height / 2
      rect = Rectangle((rect_x, rect_y), width, height, fill=False, **rect_kwargs)
      
    ax.add_patch(rect)

  return axins