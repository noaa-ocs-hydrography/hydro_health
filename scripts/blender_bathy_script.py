import bpy
import math
import os
import signal
import re

# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS TO YOUR LOCAL FILES
# ==============================================================================
# IMPORTANT: Point this to the .png you just exported with "Use Renderer" checked!
FILEPATH_COLOR = "C:/Users/aubrey.mccutchan/Documents/blender_files/florida_clip.png" 

# Provide a list of all your bathymetry files in chronological order
FILEPATH_BATHY_LIST = [
    "C:/Users/aubrey.mccutchan/Documents/blender_files/bathy_2004_filled.tif",
    "C:/Users/aubrey.mccutchan/Documents/blender_files/bathy_2006_filled.tif",
    "C:/Users/aubrey.mccutchan/Documents/blender_files/bathy_2010_filled.tif",
    "C:/Users/aubrey.mccutchan/Documents/blender_files/bathy_2015_filled.tif",
    "C:/Users/aubrey.mccutchan/Documents/blender_files/bathy_2022_filled.tif"
]
# Direct absolute path to your documents folder to prevent saving to the C: root drive
OUTPUT_FOLDER = "C:/Users/aubrey.mccutchan/Documents/tampa_bay_frames" 

NODATA_THRESHOLD = -100.0    # Values below this are considered "nodata" and filtered out
DEFAULT_ELEVATION = 2.0      # Land elevation (Raised slightly to ensure it sits above the Z=0 water level)

# --- BATHYMETRY COLORMAP SETTINGS ---
BATHY_COLOR_DEEP = -15.0     # The depth (in meters) that represents the deepest/darkest blue color
BATHY_COLOR_SHALLOW = 0.0    # The depth (in meters) that represents the shallow/sand color

PROTOTYPE_MODE = True     # SET TO TRUE: Fast, low-res preview rendering. FALSE: Cinematic HD quality.

DISPLACEMENT_SCALE = 0.5  # Adjust this to make the depth more or less extreme
SCENE_SCALE = 10.0        # Size of the terrain plane

# Animation settings
FPS = 25
HOLD_SECONDS = 1.0        # How long to pause on each year
TRANSITION_SECONDS = 0.5  # How long the morph takes between years

HOLD_FRAMES = int(FPS * HOLD_SECONDS)
TRANSITION_FRAMES = int(FPS * TRANSITION_SECONDS)
NUM_YEARS = len(FILEPATH_BATHY_LIST)

FRAME_START = 1
# Calculate total frames automatically based on number of years and durations
FRAME_END = HOLD_FRAMES + (NUM_YEARS - 1) * (TRANSITION_FRAMES + HOLD_FRAMES)

# ==============================================================================

def clear_scene():
    """Clears all mesh objects, lights, and cameras from the default scene."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_render_engine():
    """Configures Cycles for realistic volumetric rendering and true displacement."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Attempt to use GPU if available, fallback to CPU
    scene.cycles.device = 'GPU' 
    
    # Increase light bounces so deep water doesn't turn solid black
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.volume_bounces = 4
    scene.cycles.transparent_max_bounces = 12
    
    # Exposure reset to 0.0 - Uniform lighting doesn't need the massive boost the Sun model needed
    scene.view_settings.exposure = 0.0
    
    # Set frame range, resolution, and FPS
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.render.fps = FPS
    
    if PROTOTYPE_MODE:
        scene.render.resolution_x = 960  # Half resolution
        scene.render.resolution_y = 540
        scene.cycles.samples = 16        # Very low samples for blazing fast renders
    else:
        scene.render.resolution_x = 1920 # Full HD
        scene.render.resolution_y = 1080
        scene.cycles.samples = 128       # High quality raytracing samples
    
    # Configure Output for Image Sequence
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

def create_terrain():
    """Creates the terrain plane with dense subdivision and animated displacement."""
    # 1. Create a dense base grid
    # calc_uvs=True forcefully ensures texture mapping works regardless of Blender version
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=50, y_subdivisions=50, size=SCENE_SCALE, location=(0, 0, 0), calc_uvs=True)
    terrain = bpy.context.active_object
    terrain.name = "Tampa_Bay_Terrain"
    
    # Smooth shading hides blocky polygons in prototype mode
    bpy.ops.object.shade_smooth()

    # 2. Add Subdivision Surface modifier for High-Res Displacement
    subsurf = terrain.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.subdivision_type = 'SIMPLE'
    subsurf.levels = 2
    if PROTOTYPE_MODE:
        subsurf.render_levels = 3  # Keep it light for fast previews
    else:
        subsurf.render_levels = 6  # Cinematic density

    # 3. Create Terrain Material
    mat = bpy.data.materials.new(name="Terrain_Mat")
    if getattr(mat, "node_tree", None) is None:
        mat.use_nodes = True
    terrain.data.materials.append(mat)
    
    # Enable True Displacement in material settings
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = 'DISPLACEMENT'
    else:
        mat.cycles.displacement_method = 'DISPLACEMENT_ONLY'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output Node
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (800, 0)

    # Principled BSDF (Surface)
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (400, 0)
    # INCREASE ROUGHNESS: Stops the land from looking like shiny black glass
    node_bsdf.inputs['Roughness'].default_value = 0.95
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    # --- UV MAPPING & TEXTURE COORDINATES ---
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_tex_coord.location = (-1500, 200)

    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_mapping.location = (-1300, 200)
    # Using 'UV' instead of 'Generated' prevents displacement from warping the image coordinates
    links.new(node_tex_coord.outputs['UV'], node_mapping.inputs['Vector'])

    # Load Land Texture
    node_tex_color = nodes.new(type='ShaderNodeTexImage')
    node_tex_color.location = (-100, 300)
    links.new(node_mapping.outputs['Vector'], node_tex_color.inputs['Vector'])
    
    # Check if image exists to prevent Python/Blender errors from injecting pure black
    image_loaded = False
    if os.path.exists(FILEPATH_COLOR):
        try:
            img = bpy.data.images.load(FILEPATH_COLOR)
            node_tex_color.image = img
            image_loaded = True
        except:
            print("Warning: Texture found but failed to load. Ensure it is a valid PNG/JPG.")
    else:
        print(f"Warning: {FILEPATH_COLOR} not found. Using fallback green color.")

    # Fallback color just in case the texture fails
    node_fallback_rgb = nodes.new(type='ShaderNodeRGB')
    node_fallback_rgb.outputs[0].default_value = (0.2, 0.3, 0.1, 1.0) # Matte earth green
    node_fallback_rgb.location = (-100, 150)

    # --- DYNAMIC BATHYMETRY & NODATA LOGIC FOR N YEARS ---
    
    current_mix = None
    global_mask_out = None
    current_frame = 1 + HOLD_FRAMES
    
    mask_nodes = []
    clean_nodes = []
    
    for i, filepath in enumerate(FILEPATH_BATHY_LIST):
        # Shift each year's nodes down visually in the Node Editor
        y_off = -200 - (i * 450)
        
        # Load Image Node
        node_tex_bathy = nodes.new(type='ShaderNodeTexImage')
        node_tex_bathy.location = (-1200, y_off)
        
        # Safely attempt to link/load the image map
        img_name = filepath.split('/')[-1].split('\\')[-1]
        img = bpy.data.images.get(img_name)
        if not img and os.path.exists(filepath):
            try:
                img = bpy.data.images.load(filepath)
            except:
                print(f"Failed to load {filepath}")
        if img:
            img.colorspace_settings.name = 'Non-Color'
            node_tex_bathy.image = img
            
        links.new(node_mapping.outputs['Vector'], node_tex_bathy.inputs['Vector'])

        # Identify Nodata
        node_m_gt = nodes.new(type='ShaderNodeMath')
        node_m_gt.operation = 'GREATER_THAN'
        node_m_gt.inputs[1].default_value = NODATA_THRESHOLD
        node_m_gt.location = (-900, y_off)
        links.new(node_tex_bathy.outputs['Color'], node_m_gt.inputs[0])
        mask_nodes.append(node_m_gt)

        # Clean Data (Replace nodata with DEFAULT_ELEVATION)
        node_clean = nodes.new(type='ShaderNodeMix')
        node_clean.data_type = 'FLOAT'
        node_clean.location = (-700, y_off)
        node_clean.inputs[2].default_value = DEFAULT_ELEVATION 
        links.new(node_m_gt.outputs['Value'], node_clean.inputs['Factor'])
        links.new(node_tex_bathy.outputs['Color'], node_clean.inputs[3]) 
        clean_nodes.append(node_clean)

        if i == 0:
            # Base case: Year 1 simply acts as the starting point
            current_mix = node_clean.outputs['Result']
            global_mask_out = node_m_gt.outputs['Value']
        else:
            # STRICT Transition Logic from Year [i-1] to Year [i]
            # No hole-filling logic: strictly enforces the visual timeline so years don't bleed through!
            node_t_val = nodes.new(type='ShaderNodeValue')
            node_t_val.location = (-900, y_off - 150)
            node_t_val.outputs['Value'].default_value = 0.0
            node_t_val.outputs['Value'].keyframe_insert(data_path="default_value", frame=current_frame)
            current_frame += TRANSITION_FRAMES
            node_t_val.outputs['Value'].default_value = 1.0
            node_t_val.outputs['Value'].keyframe_insert(data_path="default_value", frame=current_frame)
            current_frame += HOLD_FRAMES

            node_new_mix = nodes.new(type='ShaderNodeMix')
            node_new_mix.data_type = 'FLOAT'
            node_new_mix.location = (-200, y_off)
            
            # The exact timeline factor directly controls the morph
            links.new(node_t_val.outputs['Value'], node_new_mix.inputs['Factor'])
            links.new(current_mix, node_new_mix.inputs[2])
            links.new(clean_nodes[i].outputs['Result'], node_new_mix.inputs[3])

            current_mix = node_new_mix.outputs['Result']

            # Update master land/water mask to include the new year
            node_global_mask = nodes.new(type='ShaderNodeMath')
            node_global_mask.operation = 'MAXIMUM'
            node_global_mask.location = (-200, y_off + 150)
            links.new(global_mask_out, node_global_mask.inputs[0])
            links.new(mask_nodes[i].outputs['Value'], node_global_mask.inputs[1])
            global_mask_out = node_global_mask.outputs['Value']

    # --- DYNAMIC BATHYMETRY COLORMAP ---
    
    # Map raw depth (-15m to 0m) to a 0.0 -> 1.0 gradient factor
    node_depth_map = nodes.new(type='ShaderNodeMapRange')
    node_depth_map.location = (100, -100)
    node_depth_map.inputs[1].default_value = BATHY_COLOR_DEEP    # From Min (Deepest)
    node_depth_map.inputs[2].default_value = BATHY_COLOR_SHALLOW # From Max (Shallowest)
    node_depth_map.inputs[3].default_value = 0.0                 # To Min
    node_depth_map.inputs[4].default_value = 1.0                 # To Max
    links.new(current_mix, node_depth_map.inputs[0])

    # The actual ColorRamp (Deep Blue to Sand/Cyan)
    node_colorramp = nodes.new(type='ShaderNodeValToRGB')
    node_colorramp.location = (300, -100)
    # Deep color (e.g. Dark Blue)
    node_colorramp.color_ramp.elements[0].position = 0.0
    node_colorramp.color_ramp.elements[0].color = (0.01, 0.05, 0.2, 1.0)
    # Shallow color (e.g. Cyan/Sand)
    node_colorramp.color_ramp.elements[1].position = 1.0
    node_colorramp.color_ramp.elements[1].color = (0.2, 0.7, 0.7, 1.0)
    # Add a mid-depth transition color
    mid_element = node_colorramp.color_ramp.elements.new(0.5)
    mid_element.color = (0.05, 0.3, 0.5, 1.0)
    
    links.new(node_depth_map.outputs['Result'], node_colorramp.inputs['Fac'])

    # --- MASKING LAND VS WATER ---

    # Mix the Land Color with the Bathy Colormap based on the master mask
    node_color_mix = nodes.new(type='ShaderNodeMixRGB')
    node_color_mix.location = (300, 100)
    links.new(global_mask_out, node_color_mix.inputs['Fac'])
    
    # Wire the fallback logic
    if image_loaded:
        links.new(node_tex_color.outputs['Color'], node_color_mix.inputs['Color1']) 
    else:
        links.new(node_fallback_rgb.outputs['Color'], node_color_mix.inputs['Color1']) 
        
    links.new(node_colorramp.outputs['Color'], node_color_mix.inputs['Color2']) # Bathy Ramp
    
    # Connect to final Base Color
    links.new(node_color_mix.outputs['Color'], node_bsdf.inputs['Base Color'])

    # --- FINAL DISPLACEMENT ---
    
    node_math = nodes.new(type='ShaderNodeMath')
    node_math.operation = 'MULTIPLY'
    node_math.inputs[1].default_value = DISPLACEMENT_SCALE
    node_math.location = (300, -400)
    links.new(current_mix, node_math.inputs[0])

    node_disp = nodes.new(type='ShaderNodeDisplacement')
    node_disp.location = (400, -400)
    node_disp.inputs['Midlevel'].default_value = 0.0 
    links.new(node_math.outputs['Value'], node_disp.inputs['Height'])
    links.new(node_disp.outputs['Displacement'], node_output.inputs['Displacement'])

def create_ocean():
    """Creates a see-through volumetric water layer."""
    bpy.ops.mesh.primitive_plane_add(size=SCENE_SCALE * 1.5, location=(0, 0, 0))
    water = bpy.context.active_object
    water.name = "Volumetric_Ocean"
    
    # Smooth surface
    bpy.ops.object.shade_smooth()
    
    # Add thickness to the water
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, -15)})
    bpy.ops.object.mode_set(mode='OBJECT')

    mat = bpy.data.materials.new(name="Water_Volumetric")
    if getattr(mat, "node_tree", None) is None:
        mat.use_nodes = True
    water.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (800, 0)

    # Glass-like surface for reflections
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (400, 100)
    node_bsdf.inputs['Base Color'].default_value = (0.7, 0.85, 0.9, 1) 
    node_bsdf.inputs['Roughness'].default_value = 0.05 
    node_bsdf.inputs['Transmission Weight'].default_value = 1.0 
    node_bsdf.inputs['IOR'].default_value = 1.333 
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    # Volume Absorption for depth scaling
    node_vol = nodes.new(type='ShaderNodeVolumeAbsorption')
    node_vol.location = (400, -100)
    node_vol.inputs['Color'].default_value = (0.2, 0.6, 0.8, 1.0) 
    
    # Lowered density from 2.5 to 0.5 so the bathymetry colormap shines through!
    node_vol.inputs['Density'].default_value = 0.5 
    links.new(node_vol.outputs['Volume'], node_output.inputs['Volume'])

def setup_lighting_and_camera():
    """Sets up flat, uniform ambient lighting and a sweeping cinematic camera."""
    # Sky Texture -> Swapped to flat Background for uniform data visualization
    world = bpy.context.scene.world
    if getattr(world, "node_tree", None) is None:
        world.use_nodes = True
    wnodes = world.node_tree.nodes
    wlinks = world.node_tree.links
    wnodes.clear()
    
    node_woutput = wnodes.new(type='ShaderNodeOutputWorld')
    
    # Create a uniform background to prevent directional shadows from warping the data's appearance
    node_bg = wnodes.new(type='ShaderNodeBackground')
    node_bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0) # Pure white
    node_bg.inputs['Strength'].default_value = 1.0 
    
    wlinks.new(node_bg.outputs['Background'], node_woutput.inputs['Surface'])

    # Camera setup
    bpy.ops.object.camera_add(location=(0, -SCENE_SCALE * 1.5, SCENE_SCALE * 1.2))
    cam = bpy.context.active_object
    cam.name = "Cinematic_Camera"
    bpy.context.scene.camera = cam
    
    # Point camera
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    target = bpy.context.active_object
    target.name = "Camera_Target"
    cam_constraint.target = target
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    # Animate Camera pan 
    cam.location = (0, -SCENE_SCALE * 1.5, SCENE_SCALE * 1.2)
    cam.keyframe_insert(data_path="location", frame=FRAME_START)
    
    cam.location = (0, -SCENE_SCALE * 1.0, SCENE_SCALE * 0.9)
    cam.keyframe_insert(data_path="location", frame=FRAME_END)

def create_text_overlay():
    """Creates a pure white 3D text object parented to the camera so it acts as a UI overlay."""
    bpy.ops.object.text_add()
    txt = bpy.context.active_object
    txt.name = "Year_Overlay"
    
    txt.data.body = "2004"
    txt.data.align_x = 'LEFT'
    txt.data.align_y = 'TOP_BASELINE'
    txt.data.size = 0.05 # Appropriate size for being near the camera lens
    
    # Create a shadowless, emission material so the text is pure bright white
    mat = bpy.data.materials.new(name="Text_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for n in nodes: nodes.remove(n)
    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_emit = nodes.new('ShaderNodeEmission')
    node_emit.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    mat.node_tree.links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])
    txt.data.materials.append(mat)
    
    cam = bpy.data.objects.get("Cinematic_Camera")
    if cam:
        txt.parent = cam
        # Position in camera's local space (Upper Left Corner)
        # X: left/right, Y: up/down, Z: forward into the scene
        txt.location = (-0.38, 0.22, -1.0) 
        txt.rotation_euler = (0, 0, 0)

def get_year_string_for_frame(frame):
    """Calculates which year text should be displayed for a given frame."""
    years = []
    # Extract the 4 digit years from the filenames provided in the config
    for path in FILEPATH_BATHY_LIST:
        match = re.search(r'(\d{4})', os.path.basename(path))
        years.append(match.group(1) if match else "Year")
        
    cycle_length = HOLD_FRAMES + TRANSITION_FRAMES
    
    # Guard against going perfectly to the end frame
    if frame >= FRAME_END:
        return years[-1]
        
    segment_index = (frame - 1) // cycle_length
    
    # Prevent array bounds error if rounding gets weird
    if segment_index >= len(years) - 1:
        return years[-1]
        
    time_in_cycle = (frame - 1) % cycle_length
    
    if time_in_cycle < HOLD_FRAMES:
        return years[segment_index]
    else:
        # We are morphing between two years!
        return f"{years[segment_index]} -> {years[segment_index + 1]}"

def render_animation():
    """Renders the animation frames to the output folder using a cancellable loop."""
    print(f"Starting render. Rendering PNG frames to {os.path.abspath(OUTPUT_FOLDER)}...")
    print("Press Ctrl+C at any time to safely cancel the render.\n")
    
    scene = bpy.context.scene
    base_filepath = os.path.join(OUTPUT_FOLDER, "frame_")
    
    # Fetch the text object so we can dynamically update it per frame
    txt_obj = bpy.data.objects.get("Year_Overlay")
    
    # Use a mutable list to track cancellation from within the signal handler
    cancel_flag = [False]
    
    def handle_sigint(sig, frame):
        print("\n\n[!] Ctrl+C detected. Cancelling after current frame completes/aborts...")
        cancel_flag[0] = True
        
    # Register the custom signal handler for Ctrl+C
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    
    try:
        # We manually loop through frames so Python can break execution
        for frame in range(FRAME_START, FRAME_END + 1):
            if cancel_flag[0]:
                print("\n[!] Render loop safely cancelled by user.")
                break
                
            scene.frame_set(frame)
            
            # Update the text overlay to match the correct year based on the frame!
            if txt_obj:
                txt_obj.data.body = get_year_string_for_frame(frame)
                
            scene.render.filepath = f"{base_filepath}{frame:04d}.png"
            
            try:
                # write_still=True renders exactly one frame
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"\n[CRITICAL ERROR] Render failed on frame {frame}!")
                print(f"Details: {e}")
                print("\nPossible fixes:")
                print("1. If you see a CUDA/OptiX GPU error, your Dell laptop GPU isn't supported.")
                print("   Fix: Go up to line 61 and change `scene.cycles.device = 'GPU'` to `'CPU'`")
                print(f"2. Make sure this folder isn't blocked by OneDrive: {OUTPUT_FOLDER}")
                
                # Force the script to abort completely instead of moving to the video compiler
                cancel_flag[0] = True 
                break
                
            # Check immediately after the frame finishes or errors out
            if cancel_flag[0]:
                print("\n[!] Render loop safely cancelled by user.")
                break
        else:
            print("\n\nRender complete! Moving to video compilation phase...")
    finally:
        # Restore the default system Ctrl+C behavior
        signal.signal(signal.SIGINT, original_sigint)

def compile_video():
    """Compiles the rendered PNG frames into an MP4 video using OpenCV (Non-Blender method)."""
    print("\nCompiling frames into an MP4 video using OpenCV...")
    
    try:
        import cv2
    except ImportError:
        print("\n[!] ERROR: OpenCV is not installed in your Python environment.")
        print("To compile the video, please open your terminal/command prompt and run:")
        print("    pip install opencv-python")
        print(f"\nYour rendered PNG frames are safe in: {OUTPUT_FOLDER}")
        return
        
    frames_dir = os.path.abspath(OUTPUT_FOLDER)
    if not os.path.exists(frames_dir):
        print(f"Error: Output folder {frames_dir} does not exist. Skipping MP4 generation.")
        return

    # Dynamically find all rendered frames
    frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
    if not frames:
        print(f"Error: Could not find any rendered PNG frames in {frames_dir}. Skipping MP4 generation.")
        return

    first_frame_path = os.path.join(frames_dir, frames[0])
    first_image = cv2.imread(first_frame_path)
    
    if first_image is None:
        print(f"Error: OpenCV could not read the first frame: {first_frame_path}")
        return
        
    height, width, layers = first_image.shape
    size = (width, height)
    
    mp4_path = os.path.join(frames_dir, "tampa_bay_animation.mp4")
    
    # 'mp4v' is a widely supported video codec identifier for MP4 containers in OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    try:
        out = cv2.VideoWriter(mp4_path, fourcc, FPS, size)
        
        print(f"Writing {len(frames)} frames to {mp4_path} at {FPS} FPS...")
        
        for frame_file in frames:
            frame_path = os.path.join(frames_dir, frame_file)
            img = cv2.imread(frame_path)
            if img is not None:
                out.write(img)
                
        out.release()
        print(f"\nSuccess! MP4 Video saved to:\n {mp4_path}")
        
    except Exception as e:
        print(f"\nFailed to compile video with OpenCV. Error details: {e}")
        print(f"Your frames are safely saved in: {frames_dir}")

def main():
    clear_scene()
    setup_render_engine()
    create_terrain()
    create_ocean()
    setup_lighting_and_camera()
    create_text_overlay()
    render_animation()
    compile_video()

if __name__ == "__main__":
    main()