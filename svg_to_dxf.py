#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = 'Ahmad Abdulnasir Shuaib <me@ahmadabdulnasir.com.ng>'
__homepage__ = https://ahmadabdulnasir.com.ng
__copyright__ = 'Copyright (c) 2025, salafi'
__version__ = "0.01t"
"""

#!/usr/bin/env python3
"""
Comprehensive SVG to DXF Converter
Preserves curve smoothness and all geometric details
Supports complex paths, Bézier curves, and multiple path elements
"""

import xml.etree.ElementTree as ET
import re
import math
from typing import List, Tuple, Union, Optional
import argparse
import os
from dataclasses import dataclass

try:
    import ezdxf
    from ezdxf import colors
    from ezdxf.math import Vec3
except ImportError:
    print("Installing required package: ezdxf")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ezdxf"])
    import ezdxf
    from ezdxf import colors
    from ezdxf.math import Vec3

@dataclass
class Point:
    """2D Point class for geometric operations"""
    x: float
    y: float
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_vec3(self):
        return Vec3(self.x, self.y, 0)

class BezierCurve:
    """Class for handling Bézier curve calculations"""
    
    @staticmethod
    def cubic_bezier_point(t: float, p0: Point, p1: Point, p2: Point, p3: Point) -> Point:
        """Calculate point on cubic Bézier curve at parameter t"""
        u = 1 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        
        p = Point(0, 0)
        p.x = uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x
        p.y = uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y
        return p
    
    @staticmethod
    def quadratic_bezier_point(t: float, p0: Point, p1: Point, p2: Point) -> Point:
        """Calculate point on quadratic Bézier curve at parameter t"""
        u = 1 - t
        tt = t * t
        uu = u * u
        
        p = Point(0, 0)
        p.x = uu * p0.x + 2 * u * t * p1.x + tt * p2.x
        p.y = uu * p0.y + 2 * u * t * p1.y + tt * p2.y
        return p
    
    @staticmethod
    def adaptive_bezier_tessellation(p0: Point, p1: Point, p2: Point, p3: Point = None, 
                                   tolerance: float = 0.1, max_depth: int = 10) -> List[Point]:
        """
        Adaptive tessellation of Bézier curves based on curvature
        Returns high-quality point approximation preserving curve smoothness
        """
        if p3 is None:  # Quadratic Bézier
            return BezierCurve._adaptive_quadratic(p0, p1, p2, tolerance, max_depth)
        else:  # Cubic Bézier
            return BezierCurve._adaptive_cubic(p0, p1, p2, p3, tolerance, max_depth)
    
    @staticmethod
    def _adaptive_quadratic(p0: Point, p1: Point, p2: Point, tolerance: float, depth: int) -> List[Point]:
        """Adaptive tessellation for quadratic Bézier curves"""
        if depth <= 0:
            return [p0, p2]
        
        # Calculate midpoint of chord
        chord_mid = Point((p0.x + p2.x) / 2, (p0.y + p2.y) / 2)
        
        # Calculate midpoint of curve
        curve_mid = BezierCurve.quadratic_bezier_point(0.5, p0, p1, p2)
        
        # Check if curve is flat enough
        if chord_mid.distance_to(curve_mid) < tolerance:
            return [p0, p2]
        
        # Subdivide curve
        # First half control points
        q0 = p0
        q1 = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
        q2 = curve_mid
        
        # Second half control points  
        r0 = curve_mid
        r1 = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        r2 = p2
        
        # Recursively tessellate both halves
        left = BezierCurve._adaptive_quadratic(q0, q1, q2, tolerance, depth - 1)
        right = BezierCurve._adaptive_quadratic(r0, r1, r2, tolerance, depth - 1)
        
        # Combine results (avoid duplicate midpoint)
        return left + right[1:]
    
    @staticmethod
    def _adaptive_cubic(p0: Point, p1: Point, p2: Point, p3: Point, tolerance: float, depth: int) -> List[Point]:
        """Adaptive tessellation for cubic Bézier curves"""
        if depth <= 0:
            return [p0, p3]
        
        # Calculate midpoint of chord
        chord_mid = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
        
        # Calculate midpoint of curve
        curve_mid = BezierCurve.cubic_bezier_point(0.5, p0, p1, p2, p3)
        
        # Check if curve is flat enough
        if chord_mid.distance_to(curve_mid) < tolerance:
            return [p0, p3]
        
        # Subdivide using De Casteljau's algorithm
        # First level
        q0 = p0
        q1 = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
        q2 = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        q3 = Point((p2.x + p3.x) / 2, (p2.y + p3.y) / 2)
        q4 = p3
        
        # Second level
        r0 = q0
        r1 = Point((q1.x + q2.x) / 2, (q1.y + q2.y) / 2)
        r2 = Point((q2.x + q3.x) / 2, (q2.y + q3.y) / 2)
        r3 = q4
        
        # Third level
        s0 = r0
        s1 = Point((r1.x + r2.x) / 2, (r1.y + r2.y) / 2)  # This is curve_mid
        s2 = r3
        
        # Left curve: (q0, q1, r1, s1)
        # Right curve: (s1, r2, q3, q4)
        
        left = BezierCurve._adaptive_cubic(q0, q1, r1, s1, tolerance, depth - 1)
        right = BezierCurve._adaptive_cubic(s1, r2, q3, q4, tolerance, depth - 1)
        
        return left + right[1:]

class SVGPathParser:
    """Parser for SVG path data with comprehensive command support"""
    
    def __init__(self, curve_tolerance: float = 0.1):
        self.curve_tolerance = curve_tolerance
        self.current_pos = Point(0, 0)
        self.start_pos = Point(0, 0)
        self.last_control = None
        
    def parse_path(self, path_data: str) -> List[List[Point]]:
        """Parse SVG path data and return list of polylines"""
        # Clean and tokenize path data
        path_data = re.sub(r'([MmLlHhVvCcSsQqTtAaZz])', r' \1 ', path_data)
        path_data = re.sub(r',', ' ', path_data)
        path_data = re.sub(r'\s+', ' ', path_data).strip()
        
        tokens = path_data.split()
        polylines = []
        current_polyline = []
        
        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            i += 1
            
            if cmd in 'Mm':
                # Move to
                x, y = float(tokens[i]), float(tokens[i+1])
                i += 2
                if cmd == 'm' and current_polyline:  # Relative move (not first)
                    x += self.current_pos.x
                    y += self.current_pos.y
                
                # Start new polyline
                if current_polyline:
                    polylines.append(current_polyline)
                    current_polyline = []
                
                self.current_pos = Point(x, y)
                self.start_pos = Point(x, y)
                current_polyline.append(self.current_pos)
                
            elif cmd in 'Ll':
                # Line to
                x, y = float(tokens[i]), float(tokens[i+1])
                i += 2
                if cmd == 'l':
                    x += self.current_pos.x
                    y += self.current_pos.y
                
                self.current_pos = Point(x, y)
                current_polyline.append(self.current_pos)
                
            elif cmd in 'Hh':
                # Horizontal line
                x = float(tokens[i])
                i += 1
                if cmd == 'h':
                    x += self.current_pos.x
                
                self.current_pos = Point(x, self.current_pos.y)
                current_polyline.append(self.current_pos)
                
            elif cmd in 'Vv':
                # Vertical line
                y = float(tokens[i])
                i += 1
                if cmd == 'v':
                    y += self.current_pos.y
                
                self.current_pos = Point(self.current_pos.x, y)
                current_polyline.append(self.current_pos)
                
            elif cmd in 'Cc':
                # Cubic Bézier curve
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x2, y2 = float(tokens[i+2]), float(tokens[i+3])
                x, y = float(tokens[i+4]), float(tokens[i+5])
                i += 6
                
                if cmd == 'c':
                    x1 += self.current_pos.x
                    y1 += self.current_pos.y
                    x2 += self.current_pos.x
                    y2 += self.current_pos.y
                    x += self.current_pos.x
                    y += self.current_pos.y
                
                # Tessellate cubic Bézier curve
                p0 = self.current_pos
                p1 = Point(x1, y1)
                p2 = Point(x2, y2)
                p3 = Point(x, y)
                
                curve_points = BezierCurve.adaptive_bezier_tessellation(
                    p0, p1, p2, p3, self.curve_tolerance
                )
                
                # Add curve points (skip first as it's current position)
                current_polyline.extend(curve_points[1:])
                self.current_pos = p3
                self.last_control = p2
                
            elif cmd in 'Ss':
                # Smooth cubic Bézier curve
                x2, y2 = float(tokens[i]), float(tokens[i+1])
                x, y = float(tokens[i+2]), float(tokens[i+3])
                i += 4
                
                if cmd == 's':
                    x2 += self.current_pos.x
                    y2 += self.current_pos.y
                    x += self.current_pos.x
                    y += self.current_pos.y
                
                # Calculate first control point (reflection of last)
                if self.last_control:
                    x1 = 2 * self.current_pos.x - self.last_control.x
                    y1 = 2 * self.current_pos.y - self.last_control.y
                else:
                    x1, y1 = self.current_pos.x, self.current_pos.y
                
                p0 = self.current_pos
                p1 = Point(x1, y1)
                p2 = Point(x2, y2)
                p3 = Point(x, y)
                
                curve_points = BezierCurve.adaptive_bezier_tessellation(
                    p0, p1, p2, p3, self.curve_tolerance
                )
                
                current_polyline.extend(curve_points[1:])
                self.current_pos = p3
                self.last_control = p2
                
            elif cmd in 'Qq':
                # Quadratic Bézier curve
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x, y = float(tokens[i+2]), float(tokens[i+3])
                i += 4
                
                if cmd == 'q':
                    x1 += self.current_pos.x
                    y1 += self.current_pos.y
                    x += self.current_pos.x
                    y += self.current_pos.y
                
                p0 = self.current_pos
                p1 = Point(x1, y1)
                p2 = Point(x, y)
                
                curve_points = BezierCurve.adaptive_bezier_tessellation(
                    p0, p1, p2, tolerance=self.curve_tolerance
                )
                
                current_polyline.extend(curve_points[1:])
                self.current_pos = p2
                self.last_control = p1
                
            elif cmd in 'Tt':
                # Smooth quadratic Bézier curve
                x, y = float(tokens[i]), float(tokens[i+1])
                i += 2
                
                if cmd == 't':
                    x += self.current_pos.x
                    y += self.current_pos.y
                
                # Calculate control point (reflection of last)
                if self.last_control:
                    x1 = 2 * self.current_pos.x - self.last_control.x
                    y1 = 2 * self.current_pos.y - self.last_control.y
                else:
                    x1, y1 = self.current_pos.x, self.current_pos.y
                
                p0 = self.current_pos
                p1 = Point(x1, y1)
                p2 = Point(x, y)
                
                curve_points = BezierCurve.adaptive_bezier_tessellation(
                    p0, p1, p2, tolerance=self.curve_tolerance
                )
                
                current_polyline.extend(curve_points[1:])
                self.current_pos = p2
                self.last_control = p1
                
            elif cmd in 'Zz':
                # Close path
                if current_polyline and self.start_pos:
                    if self.current_pos.distance_to(self.start_pos) > 1e-6:
                        current_polyline.append(self.start_pos)
                    self.current_pos = self.start_pos
        
        if current_polyline:
            polylines.append(current_polyline)
            
        return polylines

class SVGToDXFConverter:
    """Main converter class handling SVG to DXF conversion"""
    
    def __init__(self, curve_tolerance: float = 0.1, use_splines: bool = True):
        self.curve_tolerance = curve_tolerance
        self.use_splines = use_splines
        self.parser = SVGPathParser(curve_tolerance)
        
    def parse_svg_file(self, svg_file: str) -> ET.Element:
        """Parse SVG file and return root element"""
        try:
            tree = ET.parse(svg_file)
            return tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid SVG file: {e}")
    
    def extract_svg_dimensions(self, root: ET.Element) -> Tuple[float, float]:
        """Extract SVG canvas dimensions"""
        width = root.get('width', '100')
        height = root.get('height', '100')
        
        # Remove units and convert to float
        width = re.sub(r'[^\d.]', '', width)
        height = re.sub(r'[^\d.]', '', height)
        
        try:
            return float(width), float(height)
        except ValueError:
            return 100.0, 100.0
    
    def parse_style(self, style_str: str) -> dict:
        """Parse CSS style string into dictionary"""
        style = {}
        if style_str:
            for item in style_str.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    style[key.strip()] = value.strip()
        return style
    
    def parse_color(self, color_str: str) -> int:
        """Parse color string and return DXF color index"""
        if not color_str or color_str == 'none':
            return 7  # White/Black
        
        # RGB color
        if color_str.startswith('rgb('):
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                # Convert RGB to nearest DXF color (simplified)
                return self.rgb_to_dxf_color(r, g, b)
        
        # Hex color
        elif color_str.startswith('#'):
            try:
                color_str = color_str[1:]
                if len(color_str) == 3:
                    color_str = ''.join([c*2 for c in color_str])
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return self.rgb_to_dxf_color(r, g, b)
            except ValueError:
                pass
        
        # Named colors (basic set)
        color_map = {
            'black': 0, 'red': 1, 'yellow': 2, 'green': 3,
            'cyan': 4, 'blue': 5, 'magenta': 6, 'white': 7
        }
        return color_map.get(color_str.lower(), 7)
    
    def rgb_to_dxf_color(self, r: int, g: int, b: int) -> int:
        """Convert RGB to nearest DXF color index (simplified mapping)"""
        # Simple mapping to basic DXF colors
        if r > 200 and g < 100 and b < 100:
            return 1  # Red
        elif r > 200 and g > 200 and b < 100:
            return 2  # Yellow
        elif r < 100 and g > 200 and b < 100:
            return 3  # Green
        elif r < 100 and g > 200 and b > 200:
            return 4  # Cyan
        elif r < 100 and g < 100 and b > 200:
            return 5  # Blue
        elif r > 200 and g < 100 and b > 200:
            return 6  # Magenta
        elif r < 100 and g < 100 and b < 100:
            return 0  # Black
        else:
            return 7  # White/Default
    
    def create_dxf_polyline(self, msp, points: List[Point], color: int = 7, 
                          closed: bool = False, layer: str = "0"):
        """Create DXF polyline from points"""
        if len(points) < 2:
            return
        
        # Create polyline
        polyline = msp.add_lwpolyline(
            [(p.x, p.y) for p in points],
            close=closed,
            dxfattribs={'layer': layer, 'color': color}
        )
        return polyline
    
    def create_dxf_spline(self, msp, points: List[Point], color: int = 7, 
                         closed: bool = False, layer: str = "0"):
        """Create DXF spline from points for smoother curves"""
        if len(points) < 2:
            return
        
        try:
            # Convert points to tuples for spline
            fit_points = [(p.x, p.y, 0) for p in points]
            
            # Create spline using fit_points (not control_points)
            spline = msp.add_spline(
                fit_points=fit_points,
                degree=3,  # Cubic spline
                closed=closed,
                dxfattribs={'layer': layer, 'color': color}
            )
            return spline
        except Exception as e:
            print(f"Warning: Failed to create spline: {e}")
            print("Falling back to polyline")
            # Fall back to polyline if spline creation fails
            return self.create_dxf_polyline(msp, points, color, closed, layer)
    
    def convert_svg_to_dxf(self, svg_file: str, dxf_file: str, 
                          layer_name: str = "SVG_IMPORT") -> bool:
        """Main conversion function"""
        try:
            # Parse SVG
            print(f"Parsing SVG file: {svg_file}")
            root = self.parse_svg_file(svg_file)
            
            # Create DXF document
            print("Creating DXF document...")
            doc = ezdxf.new('R2010')  # Use modern DXF version
            msp = doc.modelspace()
            
            # Create layer
            doc.layers.new(name=layer_name, dxfattribs={'color': 7})
            
            # Get SVG dimensions for reference
            width, height = self.extract_svg_dimensions(root)
            print(f"SVG dimensions: {width} x {height}")
            
            # Find all path elements
            paths_converted = 0
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            
            for path_elem in root.findall('.//svg:path', namespace):
                path_data = path_elem.get('d')
                if not path_data:
                    continue
                
                # Parse style and attributes
                style = self.parse_style(path_elem.get('style', ''))
                fill_color = style.get('fill', path_elem.get('fill', 'none'))
                stroke_color = style.get('stroke', path_elem.get('stroke', 'black'))
                
                # Determine color
                if stroke_color and stroke_color != 'none':
                    color = self.parse_color(stroke_color)
                elif fill_color and fill_color != 'none':
                    color = self.parse_color(fill_color)
                else:
                    color = 7  # Default
                
                # Parse path and convert to polylines
                try:
                    polylines = self.parser.parse_path(path_data)
                    
                    for polyline_points in polylines:
                        if len(polyline_points) >= 2:
                            # Check if path is closed
                            closed = (len(polyline_points) > 2 and 
                                    polyline_points[0].distance_to(polyline_points[-1]) < 1e-6)
                            
                            try:
                                if self.use_splines and len(polyline_points) > 3:
                                    # Use splines for smoother curves
                                    self.create_dxf_spline(msp, polyline_points, color, 
                                                         closed, layer_name)
                                    paths_converted += 1
                                else:
                                    # Use polylines
                                    self.create_dxf_polyline(msp, polyline_points, color, 
                                                           closed, layer_name)
                                    paths_converted += 1
                            except Exception as e:
                                print(f"Warning: Failed to convert path: {e}")
                                # Try with polyline as fallback
                                try:
                                    self.create_dxf_polyline(msp, polyline_points, color, 
                                                           closed, layer_name)
                                    paths_converted += 1
                                except Exception as e2:
                                    print(f"Warning: Fallback also failed: {e2}")
                                    continue
                    
                except Exception as e:
                    print(f"Warning: Failed to parse path: {e}")
                    continue
            
            # Handle other SVG elements (circles, rectangles, etc.)
            self._convert_other_elements(root, msp, layer_name, namespace)
            
            # Save DXF file
            print(f"Saving DXF file: {dxf_file}")
            doc.saveas(dxf_file)
            
            print(f"Conversion complete! Converted {paths_converted} paths.")
            print(f"Curve tolerance used: {self.curve_tolerance}")
            print(f"Output method: {'Splines' if self.use_splines else 'Polylines'}")
            
            return True
            
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False
    
    def _convert_other_elements(self, root: ET.Element, msp, layer_name: str, namespace: dict):
        """Convert other SVG elements like circles, rectangles, etc."""
        
        # Convert circles
        for circle in root.findall('.//svg:circle', namespace):
            try:
                cx = float(circle.get('cx', 0))
                cy = float(circle.get('cy', 0))
                r = float(circle.get('r', 0))
                
                style = self.parse_style(circle.get('style', ''))
                stroke_color = style.get('stroke', circle.get('stroke', 'black'))
                color = self.parse_color(stroke_color)
                
                msp.add_circle((cx, cy), r, dxfattribs={'layer': layer_name, 'color': color})
                
            except (ValueError, TypeError):
                continue
        
        # Convert rectangles
        for rect in root.findall('.//svg:rect', namespace):
            try:
                x = float(rect.get('x', 0))
                y = float(rect.get('y', 0))
                width = float(rect.get('width', 0))
                height = float(rect.get('height', 0))
                
                style = self.parse_style(rect.get('style', ''))
                stroke_color = style.get('stroke', rect.get('stroke', 'black'))
                color = self.parse_color(stroke_color)
                
                # Create rectangle as polyline
                points = [
                    (x, y), (x + width, y), 
                    (x + width, y + height), (x, y + height)
                ]
                msp.add_lwpolyline(points, close=True, 
                                 dxfattribs={'layer': layer_name, 'color': color})
                
            except (ValueError, TypeError):
                continue
        
        # Convert lines
        for line in root.findall('.//svg:line', namespace):
            try:
                x1 = float(line.get('x1', 0))
                y1 = float(line.get('y1', 0))
                x2 = float(line.get('x2', 0))
                y2 = float(line.get('y2', 0))
                
                style = self.parse_style(line.get('style', ''))
                stroke_color = style.get('stroke', line.get('stroke', 'black'))
                color = self.parse_color(stroke_color)
                
                msp.add_line((x1, y1), (x2, y2), 
                           dxfattribs={'layer': layer_name, 'color': color})
                
            except (ValueError, TypeError):
                continue

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Convert SVG files to DXF format with high-quality curve preservation'
    )
    parser.add_argument('input', help='Input SVG file path')
    parser.add_argument('output', help='Output DXF file path')
    parser.add_argument('--tolerance', '-t', type=float, default=0.1,
                       help='Curve tessellation tolerance (smaller = higher quality, default: 0.1)')
    parser.add_argument('--polylines', '-p', action='store_true',
                       help='Use polylines instead of splines (better compatibility)')
    parser.add_argument('--layer', '-l', default='SVG_IMPORT',
                       help='DXF layer name (default: SVG_IMPORT)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return 1
    
    if not args.input.lower().endswith('.svg'):
        print("Warning: Input file doesn't have .svg extension")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Perform conversion
    converter = SVGToDXFConverter(
        curve_tolerance=args.tolerance,
        use_splines=not args.polylines
    )
    
    success = converter.convert_svg_to_dxf(args.input, args.output, args.layer)
    
    return 0 if success else 1


def convert_svg_to_dxf_file(input_svg_path, output_dxf_path, curve_tolerance=0.1, use_splines=True, layer_name="SVG_IMPORT"):
    """
    API function to convert an SVG file to DXF format
    
    Args:
        input_svg_path (str): Path to the input SVG file
        output_dxf_path (str): Path where the output DXF file will be saved
        curve_tolerance (float): Curve tessellation tolerance (smaller = higher quality)
        use_splines (bool): Whether to use splines (True) or polylines (False)
        layer_name (str): Name of the DXF layer
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    # Validate input file
    if not os.path.exists(input_svg_path):
        print(f"Error: Input file '{input_svg_path}' not found!")
        return False
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_dxf_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Perform conversion
    converter = SVGToDXFConverter(
        curve_tolerance=curve_tolerance,
        use_splines=use_splines
    )
    
    return converter.convert_svg_to_dxf(input_svg_path, output_dxf_path, layer_name)

def boot():
    exit(main())

if __name__ == "__main__":
    boot()


# Example usage:
"""
# Basic conversion
python svg_to_dxf.py input.svg output.dxf

# High-quality conversion with very fine tolerance
python svg_to_dxf.py input.svg output.dxf --tolerance 0.01

# Use polylines for better compatibility
python svg_to_dxf.py input.svg output.dxf --polylines

# Custom layer name
python svg_to_dxf.py input.svg output.dxf --layer "GARMENT_PATTERNS"

# Programmatic usage:
converter = SVGToDXFConverter(curve_tolerance=0.05, use_splines=True)
success = converter.convert_svg_to_dxf('pattern.svg', 'pattern.dxf', 'PATTERNS')
"""