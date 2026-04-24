import os
import re
import xml.etree.ElementTree as ET

# Namespace used by the geometry XML section.
GEOM_NS = {"geom": "http://pds.nasa.gov/pds4/geom/v1"}

# Namespace used by the PDS4 product description section.
# Needed to read image dimensions from Axis_Array elements.
PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

# Number of bytes to read from the IMG file header.
# This is large enough to include the CAHVOR metadata block.
IMG_HEADER_READ_BYTES = 200000


def _get_float(element, tag_name):
    # Read one float child from an XML element.
    child = element.find(f"geom:{tag_name}", GEOM_NS)

    if child is None or child.text is None:
        raise ValueError(f"Tag '{tag_name}' not found in XML element.")

    return float(child.text)


def _read_vector_axis_from_xml(root):
    # Read the A vector from XML as a fallback.
    axis = root.find(".//geom:Vector_Axis", GEOM_NS)

    if axis is None:
        raise ValueError("Vector_Axis not found in XML.")

    return (
        _get_float(axis, "x_unit"),
        _get_float(axis, "y_unit"),
        _get_float(axis, "z_unit")
    )


def _read_vector_horizontal_from_xml(root):
    # Read the H vector from XML as a fallback.
    horizontal = root.find(".//geom:Vector_Horizontal", GEOM_NS)

    if horizontal is None:
        raise ValueError("Vector_Horizontal not found in XML.")

    return (
        _get_float(horizontal, "x_pixel"),
        _get_float(horizontal, "y_pixel"),
        _get_float(horizontal, "z_pixel")
    )


def _read_vector_vertical_from_xml(root):
    # Read the V vector from XML as a fallback.
    vertical = root.find(".//geom:Vector_Vertical", GEOM_NS)

    if vertical is None:
        raise ValueError("Vector_Vertical not found in XML.")

    return (
        _get_float(vertical, "x_pixel"),
        _get_float(vertical, "y_pixel"),
        _get_float(vertical, "z_pixel")
    )


def _read_dimensions_from_xml(root):
    # Read image width (Sample axis) and height (Line axis) from the PDS4 Axis_Array elements.
    # PDS4 labels contain one Axis_Array per image dimension
    # This avoids hardcoding image sizes per camera group, which breaks when different sequences have different crop dimensions.
    width = None
    height = None

    for axis_array in root.findall(".//pds:Axis_Array", PDS_NS):
        name_el = axis_array.find("pds:axis_name", PDS_NS)
        elems_el = axis_array.find("pds:elements", PDS_NS)

        if name_el is None or elems_el is None:
            continue

        name = name_el.text.strip().lower()
        value = int(elems_el.text.strip())

        if name == "sample":
            width = value
        elif name == "line":
            height = value

    if width is None or height is None:
        raise ValueError(
            "Could not read image dimensions from XML Axis_Array elements. "
            f"Found width={width}, height={height}."
        )

    return width, height


def _read_img_header(img_path):
    # Read the beginning of the IMG file and decode it as text.
    # The CAHVOR metadata appears in the header text section.
    with open(img_path, "rb") as f:
        raw = f.read(IMG_HEADER_READ_BYTES)

    return raw.decode("latin1", errors="ignore")


def _parse_model_component(header_text, component_number):
    # Extract one CAHVOR model component from the IMG header.
    #
    # Example lines in the header look like: MODEL_COMPONENT_2 = (1.513950e-01,-9.821702e-01,-1.113792e-01)
    #
    # Component mapping:
    # 1 = C
    # 2 = A
    # 3 = H
    # 4 = V
    # 5 = O
    # 6 = R
    pattern = rf"MODEL_COMPONENT_{component_number}\s*=\s*\(([^)]*)\)"
    match = re.search(pattern, header_text, flags=re.DOTALL)

    if match is None:
        raise ValueError(
            f"MODEL_COMPONENT_{component_number} not found in IMG header."
        )

    values_text = match.group(1)

    # Split the tuple contents by comma and convert to floats.
    parts = [part.strip() for part in values_text.split(",")]

    if len(parts) != 3:
        raise ValueError(
            f"MODEL_COMPONENT_{component_number} does not contain 3 values."
        )

    return tuple(float(part) for part in parts)


def _read_cahvor_from_img(img_path):
    # Read A, H, V directly from the IMG header.
    # This is the preferred source because it is closer to the original metadata than the XML fallback.
    header_text = _read_img_header(img_path)

    # Basic sanity check so we fail early with a useful message.
    if "MODEL_TYPE" not in header_text or "CAHVOR" not in header_text:
        raise ValueError(f"CAHVOR model block not found in IMG header: {img_path}")

    A = _parse_model_component(header_text, 2)
    H = _parse_model_component(header_text, 3)
    V = _parse_model_component(header_text, 4)

    return A, H, V


def _resolve_img_path(xml_path):
    # Resolve the IMG file path from the XML path.
    # This to account for current set up structure
    file_stem = os.path.splitext(os.path.basename(xml_path))[0]
    project_root = os.path.dirname(os.path.dirname(xml_path))
    img_path = os.path.join(project_root, "IMG_files", f"{file_stem}.IMG")
    return img_path


def parse_cahvor_xml(xml_path):
    # Parse one XML path, but prefer the corresponding IMG header as the main source of A, H, V.
    # The XML is always parsed (even when the IMG exists) because width and height can only be read from Axis_Array in the XML.
    file_stem = os.path.splitext(os.path.basename(xml_path))[0]
    img_path = _resolve_img_path(xml_path)

    # Always parse the XML tree to get image dimensions.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width, height = _read_dimensions_from_xml(root)

    # First try to read A, H, V from the IMG header.
    if os.path.exists(img_path):
        A, H, V = _read_cahvor_from_img(img_path)

        return {
            "file_stem": file_stem,
            "xml_path": xml_path,
            "img_path": img_path,
            "source": "IMG_HEADER",
            "A": A,
            "H": H,
            "V": V,
            "width": width,
            "height": height
        }

    # If the IMG file does not exist, fall back to XML for A, H, V too.
    # This keeps the code usable even if only XML files are available.
    return {
        "file_stem": file_stem,
        "xml_path": xml_path,
        "img_path": None,
        "source": "XML_FALLBACK",
        "A": _read_vector_axis_from_xml(root),
        "H": _read_vector_horizontal_from_xml(root),
        "V": _read_vector_vertical_from_xml(root),
        "width": width,
        "height": height
    }


def parse_cahvor_folder(data_folder):
    # Parse every XML file in the folder.
    # Even though we loop through XML files, the actual A/H/V values will be taken from the matching IMG header whenever available.
    # Width and height are always read from the XML Axis_Array elements.
    if not os.path.isdir(data_folder):
        raise ValueError(f"Provided path '{data_folder}' is not a valid directory.")

    parsed_items = []

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith(".xml"):
            continue

        xml_path = os.path.join(data_folder, filename)
        parsed_items.append(parse_cahvor_xml(xml_path))

    if not parsed_items:
        raise ValueError(f"No XML files found in directory '{data_folder}'.")

    return parsed_items
