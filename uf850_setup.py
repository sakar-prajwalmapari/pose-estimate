"""
Setup script to attach Orbbec Gemini2 camera to UF850 robot arm on top of link6
Based on hierarchy positions:
- Camera (orbbec_gemini2_v1_0): World Position (0, 0, 0)
- Robot link6: World Position (0.149999, -0.000004, 1.232252)

USAGE:
Run with: /home/ubuntu24/isaac-sim/python.sh uf850_setup.py <path_to_usd_file>
Example: /home/ubuntu24/isaac-sim/python.sh uf850_setup.py /path/to/scene.usd
"""

import sys
import os
import argparse

# Initialize Isaac Sim in headless mode
from isaacsim import SimulationApp

# Parse arguments before creating SimulationApp
parser = argparse.ArgumentParser(description="Attach camera to UF850 robot")
parser.add_argument("usd_file", nargs="?", help="Path to USD scene file to modify")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": args.headless})

# Now import omni modules after SimulationApp is created
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf, Sdf
import omni.usd


def attach_camera_to_robot():
    """
    Attach the Orbbec Gemini2 camera to the UF850 robot's link6
    Moves camera from /World/orbbec_gemini2_v1_0 to /World/uf850/link6/orbbec_gemini2_v1_0
    """
    if not ISAAC_SIM_AVAILABLE:
        print("Cannot attach camera - Isaac Sim libraries not loaded")
        return False
    
    stage = get_current_stage()
    
    if stage is None:
        print("Error: No stage is currently loaded")
        print("Please open a scene in Isaac Sim first")
        return False
    
    camera_prim_path = "/World/orbbec_gemini2_v1_0"
    link6_prim_path = "/World/uf850/link6"
    
    camera_prim = get_prim_at_path(camera_prim_path)
    link6_prim = get_prim_at_path(link6_prim_path)
    
    if not camera_prim.IsValid():
        print(f"Error: Camera prim not found at {camera_prim_path}")
        return False
    
    if not link6_prim.IsValid():
        print(f"Error: Link6 prim not found at {link6_prim_path}")
        return False
    
    print(f"Found camera at {camera_prim_path}")
    print(f"Found link6 at {link6_prim_path}")
    
    # Define new path under link6
    new_camera_path = f"{link6_prim_path}/orbbec_gemini2_v1_0"
    
    # Move the camera prim to be under link6
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(camera_prim_path, new_camera_path)
    
    if stage.GetRootLayer().Apply(edit):
        print(f"✓ Successfully moved camera from {camera_prim_path} to {new_camera_path}")
        
        # Get the moved camera prim
        moved_camera_prim = get_prim_at_path(new_camera_path)
        moved_camera_xform = UsdGeom.Xformable(moved_camera_prim)
        
        # Clear any existing transform ops to start fresh
        moved_camera_xform.ClearXformOpOrder()
        
        # Set local transform relative to link6
        # Mount camera on top of link6 with offset
        # Z offset: 0.05m (5cm) above link6 to mount on top
        translate_op = moved_camera_xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.05))
        
        # Rotate camera to point forward/down as needed
        # Default orientation or adjust based on your requirements
        rotate_op = moved_camera_xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        print(f"✓ Camera mounted on top of link6 with local offset (0, 0, 0.05)")
        print(f"✓ Camera will now follow link6 movements")
        return True
    else:
        print("✗ Failed to move camera prim")
        return False


def setup_camera_mount_with_xform():
    """
    Alternative method: Create a camera mount Xform under link6, then attach camera to it
    This provides an intermediate mounting point for easier positioning adjustments
    """
    if not ISAAC_SIM_AVAILABLE:
        print("Cannot attach camera - Isaac Sim libraries not loaded")
        return False
    
    stage = get_current_stage()
    
    if stage is None:
        print("Error: No stage is currently loaded")
        print("Please open a scene in Isaac Sim first")
        return False
    
    camera_prim_path = "/World/orbbec_gemini2_v1_0"
    link6_prim_path = "/World/uf850/link6"
    
    # Get the camera prim
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    
    if not camera_prim.IsValid():
        print(f"Error: Camera prim not found at {camera_prim_path}")
        return False
    
    print(f"Found camera at {camera_prim_path}")
    print(f"Creating camera mount under {link6_prim_path}")
    
    # Create an Xform under link6 to hold the camera
    camera_mount_path = f"{link6_prim_path}/camera_mount"
    camera_mount_xform = UsdGeom.Xform.Define(stage, camera_mount_path)
    
    # Set the mount position on top of link6
    # Local offset from link6 center
    translate_op = camera_mount_xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.08))  # 8cm offset in Z (on top)
    
    rotate_op = camera_mount_xform.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    
    print(f"✓ Created camera mount at {camera_mount_path}")
    
    # Now reparent the camera under the mount
    new_camera_path = f"{camera_mount_path}/orbbec_gemini2_v1_0"
    
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(camera_prim_path, new_camera_path)
    
    if stage.GetRootLayer().Apply(edit):
        print(f"✓ Successfully attached camera to robot arm via mount")
        print(f"✓ Camera path: {new_camera_path}")
        print(f"✓ Mount offset from link6: (0, 0, 0.08)")
        
        # Optionally adjust camera orientation within the mount
        moved_camera_prim = get_prim_at_path(new_camera_path)
        moved_camera_xform = UsdGeom.Xformable(moved_camera_prim)
        moved_camera_xform.ClearXformOpOrder()
        
        # Small additional offset if needed
        cam_translate = moved_camera_xform.AddTranslateOp()
        cam_translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        cam_rotate = moved_camera_xform.AddRotateXYZOp()
        cam_rotate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        return True
    else:
        print("✗ Failed to attach camera to mount")
        return False


if __name__ == "__main__":
    if not ISAAC_SIM_AVAILABLE:
        # Error message already printed above
        sys.exit(1)
    
    # Run the camera attachment
    print("=" * 60)
    print("ATTACHING ORBBEC GEMINI2 CAMERA TO UF850 ROBOT ARM")
    print("=" * 60)
    print("\nTarget: Mount camera on top of link6")
    print("Link6 world position: (0.149999, -0.000004, 1.232252)")
    print("\nAttempting direct attachment method...")
    print("-" * 60)
    
    success = attach_camera_to_robot()
    
    if success:
        print("-" * 60)
        print("✓ CAMERA SUCCESSFULLY ATTACHED TO ROBOT ARM")
        print("=" * 60)
    else:
        print("-" * 60)
        print("✗ Direct attachment failed")
        print("\nTrying alternative method with camera mount...")
        print("-" * 60)
        success = setup_camera_mount_with_xform()
        if success:
            print("-" * 60)
            print("✓ CAMERA SUCCESSFULLY ATTACHED USING MOUNT METHOD")
            print("=" * 60)
        else:
            print("-" * 60)
            print("✗ BOTH METHODS FAILED")
            print("Please check that the prims exist and are accessible")
            print("=" * 60)
