#!/usr/bin/env python3

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup
from svgpathtools import parse_path 

def colorize_and_label_svg(
	svg_in_path      : str,
	svg_out_path     : str,
	region_values    : dict,
	path_values		 : dict,
	*,
	value_type		 : str = "unnormalized",
	colormap_name    : str = "viridis",
	font_size_px     : int = 8,
	font_family      : str = "Arial",
	font_fill        : str = "#000000"
):
	"""
	svg_in_path   – template SVG whose <path id="ABC"> are cortical regions
	svg_out_path  – where to write the recolored, labelled SVG
	region_values – dict mapping region abbreviation → scalar value
	colormap_name – any Matplotlib sequential/ diverging colormap
	"""
	
	#label_names = {'FRP':'FRP', 'ACAd': 'ACAd', 'ACAv': 'v', 'PL': 'PL', 'ILA':'ILA', 'ORBl', 'ORBm', 'ORBvl',
					#'AId', 'AIv', 'AIp', 'GU':'GU', 'VISC':'VISC', 'TEa':'TEa', 'PERI':'PERI', 'ECT':'ECT',
					#'SSs':'SSs', 'SSp', 'MOs', 'MOp',
					#'VISal':'al', 'VISli':'li', 'VISpl':'pl', 'VISpor':'por', 'VISrl':'rl', 'VISp':'VISp', 'VISl':'l', 
					#'VISam':'am', 'VISpm':'pm', 'RSPagl':'agl', 'RSPd':'RSPd', 'RSPv':'v', 'VISa':'a', 
					#'AUDd', 'AUDpo', 'AUDp', 'AUDv'}
	
	# read the SVG template
	with open(svg_in_path, "r", encoding="utf‑8") as fh:
		soup = BeautifulSoup(fh.read(), "xml")

	# prepare colormap
	vals = list(region_values.values())
	if not vals:
		raise ValueError("region_values is empty!")

	if value_type == 'normalized':
		vmin = -1.0; vmax = 1.0
	else:
		vmin, vmax = min(vals), max(vals)
	norm = mcolors.Normalize(vmin, vmax, clip=True)
	cmap = cm.get_cmap(colormap_name)

	# make (or reuse) <g id="labels"> at the END of the SVG
	labels_layer = soup.find('g', id='labels')
	if labels_layer is None:
		labels_layer = soup.new_tag('g', id='labels')
		soup.svg.append(labels_layer)        # <-- LAST = top‑most

	# recolour paths and collect centroids
	for path in soup.find_all('path'):
		reg = path.get('id')
		if reg not in region_values:
			continue

		# colour
		hex_colour = mcolors.to_hex(cmap(norm(region_values[reg])))
		style = path.get('style', '')
		if 'stroke:' not in style:
			if path_values[reg] < 0.05:
				style += ';stroke:#000;stroke-width:0.8'
			else:
				style += ';stroke:#000;stroke-width:0.2'
		style += f';fill:{hex_colour}'
		path['style'] = style

		# centroid
		xmin, xmax, ymin, ymax = parse_path(path['d']).bbox()
		cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

		# label
		lbl = soup.new_tag('text',
						   x=f'{cx}',
						   y=f'{cy}',
						   style=(
								f"font-size:{font_size_px}px;"
								f"font-family:{font_family};"
								f"text-anchor:middle;"          # ⬅ center horizontally
								f"dominant-baseline:middle;"    # ⬅ center vertically
								f"pointer-events:none;"         # ⬅ do not block clicks
								f"fill:{font_fill}"
							))
		lbl.string = reg
		labels_layer.append(lbl)             # all labels share top layer
	
	# write the modified SVG
	with open(svg_out_path, "w", encoding="utf‑8") as fh:
		fh.write(str(soup))
	print(f"✨  wrote {svg_out_path}")


def colorize_svg(
	svg_in_path, 
	svg_out_path, 
	region_values, 
	value_type,
	colormap_name='viridis',
):
	# 1. Read the entire SVG
	with open(svg_in_path, 'r', encoding='utf-8') as f:
		soup = BeautifulSoup(f.read(), 'xml')

	# 2. Prepare a colormap that maps min->max of your data to colors
	vals = list(region_values.values())
	if value_type == 'normalized':
		vmin = -1.0; vmax = 1.0
	else:
		vmin, vmax = min(vals), max(vals)
	norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
	cmap = cm.get_cmap(colormap_name)

	# 3. For each <path id="REGION"> in the SVG, fill with the appropriate color
	for path in soup.find_all('path'):
		region_id = path.get('id', None)
		if region_id in region_values:
			value = region_values[region_id]
			rgba = cmap(norm(value))   # rgba is (r,g,b,a), each in [0..1]
			hex_color = mcolors.to_hex(rgba)
			# Set path fill color.  You might keep or overwrite stroke color
			path['style'] = f"fill:{hex_color};stroke:#000000;stroke-width:0.5"
			#path['style'] = f"fill:{hex_color};stroke:{hex_color};stroke-width:0.5"

	# 4. Write out the modified SVG
	with open(svg_out_path, 'w', encoding='utf-8') as out:
		out.write(str(soup))
		

if __name__ == "__main__":
	region_values_example = {
		"ORBm":  5.2,
		"FRP": 10.8,
		"PL":   7.1,
		"ACAv": 12.3, 
		# etc.
	}
	
	colorize_svg(
		"cortical_map.svg", 
		"cortical_map_colorized.svg", 
		region_values_example, 
		colormap_name='plasma'
	)
	
	
	
