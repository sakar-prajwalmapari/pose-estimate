#!/usr/bin/env python3
"""
UF850 Robot Manual Joint Control with Image Capture
====================================================
This script provides:
1. Manual multi-joint control for the UF850 robot (type commands and press Enter)
2. Continuous image capture to dataset folder (RGB + Depth)
3. Smooth joint interpolation
"""

import os
import sys
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Set up Isaac Sim environment
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "graphics,utility,compute"

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom
import omni.replicator.core as rep

# Configuration
USD_PATH = "/home/ubuntu24/isaac-sim1/obj_detect_uf850.usd"
ROBOT_PATH = "/World/uf850"
CAMERA_RGB_PATH = "/World/uf850/link6/orbbec_gemini2_v1_0/Orbbec_Gemini2/camera_rgb/camera_rgb/Stream_rgb"
CAMERA_DEPTH_PATH = "/World/uf850/link6/orbbec_gemini2_v1_0/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth"
DATASET_ROOT = "/home/ubuntu24/pnp/datasets/manual_capture"
JOINT_INCREMENT = 0.05  # radians (~2.9 degrees)
CAPTURE_INTERVAL = 0.5  # seconds between captures in continuous mode
IMAGE_RESOLUTION = (640, 480)

# Global flags
capturing = False
image_count = 0

print("\n" + "="*80)
print("🎮 UF850 MANUAL CONTROL + IMAGE CAPTURE")
print("="*80)


def setup_dataset_folder():
    """Create image dataset folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = Path(DATASET_ROOT) / timestamp
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = f"""# UF850 Manual Control Image Dataset

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Info
Images captured during manual control of UF850 robot arm.
Camera: Orbbec Gemini2 (RGB + Depth, mounted on link6)

## Controls Used
- Multi-joint control via keyboard
- Continuous/single image capture
- Resolution: {IMAGE_RESOLUTION[0]}x{IMAGE_RESOLUTION[1]}
- Format: RGB (PNG), Depth (NPY)
"""
    
    with open(dataset_path / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Dataset folder: {dataset_path}")
    return dataset_path


def get_end_effector_position(robot_path):
    """Get the current end-effector position in world coordinates"""
    try:
        eef_prim = get_prim_at_path(f"{robot_path}/tool0")
        if eef_prim.IsValid():
            xformable = UsdGeom.Xformable(eef_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            translation = world_transform.ExtractTranslation()
            return np.array([translation[0], translation[1], translation[2]])
    except:
        pass
    return np.array([0.0, 0.0, 0.0])


def print_robot_state(robot):
    """Print current robot state"""
    joint_positions = robot.get_joint_positions()
    eef_pos = get_end_effector_position(ROBOT_PATH)
    
    print(f"\n📐 Joint Angles (degrees):")
    print(f"   J1: {np.degrees(joint_positions[0]):6.2f}°  J2: {np.degrees(joint_positions[1]):6.2f}°  J3: {np.degrees(joint_positions[2]):6.2f}°")
    print(f"   J4: {np.degrees(joint_positions[3]):6.2f}°  J5: {np.degrees(joint_positions[4]):6.2f}°  J6: {np.degrees(joint_positions[5]):6.2f}°")
    print(f"📍 End-Effector: ({eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f})")


def apply_joint_changes(robot, world, joint_deltas):
    """Apply incremental changes to multiple joints simultaneously with smooth interpolation"""
    current_joints = robot.get_joint_positions()
    new_joints = current_joints.copy()
    
    # Joint limits for UF850
    joint_limits = [
        (-3.05, 3.05),   # J1
        (-2.61, 2.61),   # J2
        (-3.14, 3.14),   # J3
        (-3.05, 3.05),   # J4
        (-2.61, 2.61),   # J5
        (-6.28, 6.28)    # J6
    ]
    
    # Apply all deltas
    for joint_idx, delta in joint_deltas.items():
        new_joints[joint_idx] += delta
        new_joints[joint_idx] = np.clip(new_joints[joint_idx], 
                                         joint_limits[joint_idx][0], 
                                         joint_limits[joint_idx][1])
    
    # Print changes
    print(f"➡️  Moving {len(joint_deltas)} joint(s):")
    for joint_idx, delta in joint_deltas.items():
        print(f"    J{joint_idx+1}: {np.degrees(current_joints[joint_idx]):6.2f}° → {np.degrees(new_joints[joint_idx]):6.2f}° (Δ{np.degrees(delta):+6.2f}°)")
    
    # Smooth interpolation over multiple steps
    steps = 10
    for i in range(steps):
        alpha = (i + 1) / steps
        interpolated = current_joints + (new_joints - current_joints) * alpha
        robot.set_joint_positions(interpolated)
        robot.set_joint_velocities(np.zeros(6))
        world.step(render=True)


def reset_to_zero(robot, world):
    """Reset all joints to zero position with smooth motion"""
    print("\n🔄 Resetting to zero position...")
    zero_joints = np.zeros(6)
    current_joints = robot.get_joint_positions()
    
    steps = 50
    for i in range(steps):
        alpha = (i + 1) / steps
        interpolated = current_joints + (zero_joints - current_joints) * alpha
        robot.set_joint_positions(interpolated)
        robot.set_joint_velocities(np.zeros(6))
        world.step(render=True)
    
    print("✅ Reset complete")
    print_robot_state(robot)


def capture_frame(dataset_path, count):
    """Capture RGB and depth frames and save to disk"""
    global image_count
    
    try:
        width, height = IMAGE_RESOLUTION
        
        # Capture RGB
        rgb_render_product = rep.create.render_product(CAMERA_RGB_PATH, (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([rgb_render_product])
        
        # Capture Depth
        depth_render_product = rep.create.render_product(CAMERA_DEPTH_PATH, (width, height))
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        depth_annotator.attach([depth_render_product])
        
        # Trigger capture
        rep.orchestrator.step(rt_subframes=4)
        
        # Get RGB data
        rgb_data = rgb_annotator.get_data()
        depth_data = depth_annotator.get_data()
        
        if rgb_data is None and depth_data is None:
            print("⚠️  Failed to capture images")
            return False
        
        # Save RGB
        if rgb_data is not None:
            from PIL import Image as PILImage
            img_array = np.array(rgb_data)
            if len(img_array.shape) == 3:
                # Handle RGBA to RGB conversion
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                pil_image = PILImage.fromarray(img_array.astype(np.uint8))
                rgb_path = dataset_path / f"rgb_{count:06d}.png"
                pil_image.save(str(rgb_path))
        
        # Save Depth
        if depth_data is not None:
            depth_array = np.array(depth_data)
            depth_path = dataset_path / f"depth_{count:06d}.npy"
            np.save(str(depth_path), depth_array)
        
        print(f"📸 Captured frame {count:06d}")
        return True
        
    except Exception as e:
        print(f"❌ Error capturing: {e}")
        return False





def print_controls():
    """Print control instructions"""
    print("\n" + "="*80)
    print("🎮 KEYBOARD CONTROLS")
    print("="*80)
    print("Joint 1 (Base):     1/Q  (increase/decrease)")
    print("Joint 2 (Shoulder): 2/W  (increase/decrease)")
    print("Joint 3 (Elbow):    3/E  (increase/decrease)")
    print("Joint 4 (Wrist 1):  4/R  (increase/decrease)")
    print("Joint 5 (Wrist 2):  5/T  (increase/decrease)")
    print("Joint 6 (Wrist 3):  6/Y  (increase/decrease)")
    print("\n🔀 Multi-Joint Control:")
    print("  Combine letters to move multiple joints at once!")
    print("  Examples: '1w' (J1+ and J2-), '23' (J2+ and J3+), 'qe' (J1- and J3-)")
    print("\n📸 Camera/Capture:")
    print("  SPACE - Toggle continuous capture ON/OFF")
    print("  C     - Capture single frame (saves RGB+Depth to dataset)")
    print("\n⚙️  Other Commands:")
    print("  0 - Reset to zero position")
    print("  P - Print current state")
    print("  H - Show this help")
    print("  X - Exit program")
    print("="*80)
    print(f"\nIncrement: {np.degrees(JOINT_INCREMENT):.1f}° per keypress")
    print("💡 TIP: Type commands and press Enter. Simulation runs continuously!\n")


def main():
    """Main control loop"""
    global capturing, image_count
    
    try:
        # Setup dataset folder
        dataset_path = setup_dataset_folder()
        
        # Load USD
        print(f"\n📂 Loading scene: {USD_PATH}")
        if not os.path.exists(USD_PATH):
            print(f"❌ ERROR: USD file not found at {USD_PATH}")
            simulation_app.close()
            sys.exit(1)
        
        open_stage(USD_PATH)
        print("✅ Scene loaded")
        
        # Initialize world
        print("🌍 Initializing world...")
        world = World()
        
        # Stabilize
        for _ in range(10):
            world.step(render=True)
        
        # Get robot
        print(f"🤖 Connecting to robot at {ROBOT_PATH}...")
        robot = Articulation(ROBOT_PATH, name="uf850")
        world.scene.add(robot)
        world.reset()
        robot.initialize()
        print(f"✅ Robot initialized ({robot.num_dof} DOF)")
        
        # Verify cameras
        rgb_prim = get_prim_at_path(CAMERA_RGB_PATH)
        depth_prim = get_prim_at_path(CAMERA_DEPTH_PATH)
        if rgb_prim.IsValid() and depth_prim.IsValid():
            print(f"✅ Cameras ready (RGB + Depth)")
        else:
            print(f"⚠️  Warning: Camera paths may be incorrect")
        
        # Print initial state
        print("\n" + "="*80)
        print("📊 INITIAL ROBOT STATE")
        print("="*80)
        print_robot_state(robot)
        print_controls()
        
        # Command mappings
        command_map = {
            '1': (0, JOINT_INCREMENT),   'q': (0, -JOINT_INCREMENT),
            '2': (1, JOINT_INCREMENT),   'w': (1, -JOINT_INCREMENT),
            '3': (2, JOINT_INCREMENT),   'e': (2, -JOINT_INCREMENT),
            '4': (3, JOINT_INCREMENT),   'r': (3, -JOINT_INCREMENT),
            '5': (4, JOINT_INCREMENT),   't': (4, -JOINT_INCREMENT),
            '6': (5, JOINT_INCREMENT),   'y': (5, -JOINT_INCREMENT),
        }
        
        def parse_multi_joint_command(cmd_str):
            """Parse command string for multiple joint commands"""
            joint_deltas = {}
            for char in cmd_str:
                if char in command_map:
                    joint_idx, delta = command_map[char]
                    if joint_idx in joint_deltas:
                        joint_deltas[joint_idx] += delta
                    else:
                        joint_deltas[joint_idx] = delta
            return joint_deltas
        
        print(f"\n📸 Capture: {'🟢 CONTINUOUS' if capturing else '🔴 OFF'}")
        print("⌨️  Ready for commands! Type and press Enter:\n")
        
        last_capture_time = time.time()
        
        # Main loop
        while simulation_app.is_running():
            world.step(render=True)
            
            current_time = time.time()
            
            # Continuous capture
            if capturing and (current_time - last_capture_time >= CAPTURE_INTERVAL):
                if capture_frame(dataset_path, image_count):
                    image_count += 1
                last_capture_time = current_time
            
            # Check for user input (non-blocking)
            try:
                import select
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    command = sys.stdin.readline().strip().lower()
                    
                    if command == 'x':
                        print("👋 Exiting...")
                        break
                    elif command == 'p':
                        print_robot_state(robot)
                    elif command == '0':
                        reset_to_zero(robot, world)
                    elif command == 'h':
                        print_controls()
                    elif command in [' ', 'space']:
                        capturing = not capturing
                        print(f"\n📸 Capture: {'🟢 CONTINUOUS' if capturing else '🔴 OFF'}")
                        print(f"   Images captured: {image_count}")
                    elif command == 'c':
                        if capture_frame(dataset_path, image_count):
                            image_count += 1
                    else:
                        # Multi-joint command
                        joint_deltas = parse_multi_joint_command(command)
                        if joint_deltas:
                            apply_joint_changes(robot, world, joint_deltas)
                        elif command:
                            print(f"⚠️  Unknown: '{command}' (type 'h' for help)")
                    
                    if command != 'x':
                        print(f"⌨️  [{'🟢 CAPTURING' if capturing else '🔴 PAUSED'}] Command: ", end='', flush=True)
                        
            except:
                # Fallback for systems without select
                pass
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n📊 Session Summary:")
        print(f"   Images captured: {image_count}")
        print(f"   Dataset: {dataset_path}")
        
        print("\n👋 Closing simulation...")
        simulation_app.close()
        print("✅ Done!")


if __name__ == "__main__":
    main()
