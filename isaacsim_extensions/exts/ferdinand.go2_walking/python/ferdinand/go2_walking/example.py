"""Interactive Isaac Sim example for the Go2 4L walking policy."""

from __future__ import annotations

import math

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni
import omni.appwindow
import torch
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.core.utils.viewports import set_active_viewport_camera, set_camera_view
from isaacsim.examples.interactive.base_sample.base_sample_experimental import BaseSample
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

from .policy import Go2VelocityPolicy


class Go2WalkingExample(BaseSample):
    """Spawn the training-model Go2 and drive its velocity command by keyboard."""

    CAMERA_PATH = "/World/Go2FollowCamera"
    CAMERA_OFFSET = (-4.0, -4.0, 2.5)
    CAMERA_TARGET_HEIGHT = 0.2
    CAMERA_FOLLOW_ALPHA = (0.12, 0.12, 0.03)

    def __init__(self) -> None:
        super().__init__()
        self._world_settings.update(
            stage_units_in_meters=1.0,
            physics_dt=0.005,
            rendering_dt=0.02,
            device="cuda",
            backend="torch",
        )
        self.go2: Go2VelocityPolicy | None = None
        self._base_command = None
        self._physics_ready = False
        self._physics_callback_id = None
        self._timeline_stop_callback_id = None
        self._sub_keyboard = None
        self._input = None
        self._keyboard = None
        self._previous_device: str | None = None
        self._previous_fabric: bool | None = None
        self._camera_state = None
        self._camera_position: list[float] | None = None
        self._camera_target: list[float] | None = None
        self._camera_update_elapsed = 0.0
        self._speed_scale = 0.5
        self._pressed_motion_keys: set[str] = set()
        self._key_commands = {
            "NUMPAD_8": (1.0, 0.0, 0.0),
            "UP": (1.0, 0.0, 0.0),
            "NUMPAD_2": (-1.0, 0.0, 0.0),
            "DOWN": (-1.0, 0.0, 0.0),
            "NUMPAD_4": (0.0, 1.0, 0.0),
            "LEFT": (0.0, 1.0, 0.0),
            "NUMPAD_6": (0.0, -1.0, 0.0),
            "RIGHT": (0.0, -1.0, 0.0),
            "NUMPAD_7": (0.0, 0.0, 1.0),
            "N": (0.0, 0.0, 1.0),
            "NUMPAD_9": (0.0, 0.0, -1.0),
            "M": (0.0, 0.0, -1.0),
        }

    def _snapshot_simulation_state(self) -> None:
        try:
            self._previous_device = SimulationManager.get_physics_sim_device()
            extension_manager = omni.kit.app.get_app().get_extension_manager()
            self._previous_fabric = extension_manager.is_extension_enabled("omni.physx.fabric")
        except Exception as error:
            carb.log_warn(f"Could not snapshot simulation state: {error}")

    def _restore_simulation_state(self) -> None:
        try:
            if self._previous_device is not None:
                SimulationManager.set_physics_sim_device(self._previous_device)
            if self._previous_fabric is not None:
                SimulationManager.enable_fabric(self._previous_fabric)
        except Exception as error:
            carb.log_warn(f"Could not restore simulation state: {error}")
        self._previous_device = None
        self._previous_fabric = None

    @staticmethod
    def _apply_ground_material() -> None:
        stage = omni.usd.get_context().get_stage()
        material = UsdShade.Material.Define(stage, "/World/ground/Looks/Go2PhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(1.0)
        physics_material.CreateDynamicFrictionAttr().Set(1.0)
        physics_material.CreateRestitutionAttr().Set(0.0)
        ground = stage.GetPrimAtPath("/World/ground/GroundPlane/CollisionPlane")
        if ground.IsValid():
            UsdShade.MaterialBindingAPI.Apply(ground).Bind(material)

    def setup_scene(self) -> None:
        self._snapshot_simulation_state()
        SimulationManager.set_backend("torch")
        SimulationManager.set_physics_sim_device("cuda")
        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac Sim assets root is unavailable")
        stage_utils.add_reference_to_stage(
            usd_path=f"{assets_root}/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )
        self._apply_ground_material()
        self.go2 = Go2VelocityPolicy(prim_path="/World/Go2", position=(0.0, 0.0, 0.5))
        UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), self.CAMERA_PATH)
        set_camera_view(
            eye=(-4.0, -4.0, 3.0),
            target=(0.0, 0.0, 0.3),
            camera_prim_path=self.CAMERA_PATH,
        )
        set_active_viewport_camera(self.CAMERA_PATH)

    async def setup_post_load(self) -> None:
        if self.go2 is None:
            raise RuntimeError("Go2 scene was not created")
        app_window = omni.appwindow.get_default_app_window()
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        viewport = get_active_viewport()
        self._camera_state = ViewportCameraState(self.CAMERA_PATH, viewport)
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._base_command = torch.zeros(3, device=torch.device(str(self.go2.robot._device)))
        self._physics_callback_id = SimulationManager.register_callback(
            self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP
        )
        self._timeline_stop_callback_id = SimulationManager.register_callback(
            self.on_timeline_stop, IsaacEvents.TIMELINE_STOP, name="Go2WalkingExample.timeline_stop"
        )

    async def setup_pre_reset(self) -> None:
        self._physics_ready = False

    async def setup_post_reset(self) -> None:
        self._physics_ready = False

    async def setup_post_clear(self) -> None:
        self.physics_cleanup()

    def on_physics_step(self, dt: float, _context: object | None = None) -> None:
        if self.go2 is None or self._base_command is None:
            return
        if not self.go2.robot.is_physics_tensor_entity_valid():
            self._physics_ready = False
            return
        if not self._physics_ready:
            self.go2.initialize()
            self.go2.post_reset()
            self._physics_ready = True
            self._update_follow_camera(force=True)
            return
        self.go2.forward(dt, self._base_command)
        self._camera_update_elapsed += float(dt)
        if self._camera_update_elapsed >= 0.02:
            self._update_follow_camera()

    def _update_follow_camera(self, force: bool = False) -> None:
        """Follow the base position without changing the camera's fixed orientation."""
        if self.go2 is None or self._camera_state is None:
            return
        if not force and self._camera_update_elapsed < 0.02:
            return
        self._camera_update_elapsed = 0.0
        base_positions, base_orientations = self.go2.robot.get_world_poses()
        base_position = base_positions.numpy()[0]
        base_orientation = base_orientations.numpy()[0]
        w, x, y, z = (float(value) for value in base_orientation)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        offset_x = cos_yaw * self.CAMERA_OFFSET[0] - sin_yaw * self.CAMERA_OFFSET[1]
        offset_y = sin_yaw * self.CAMERA_OFFSET[0] + cos_yaw * self.CAMERA_OFFSET[1]
        desired_position = (
            float(base_position[0]) + offset_x,
            float(base_position[1]) + offset_y,
            float(base_position[2]) + self.CAMERA_OFFSET[2],
        )
        desired_target = (
            float(base_position[0]),
            float(base_position[1]),
            float(base_position[2]) + self.CAMERA_TARGET_HEIGHT,
        )
        if force or self._camera_position is None:
            self._camera_position = list(desired_position)
            self._camera_target = list(desired_target)
        else:
            for axis, alpha in enumerate(self.CAMERA_FOLLOW_ALPHA):
                self._camera_position[axis] += alpha * (desired_position[axis] - self._camera_position[axis])
                self._camera_target[axis] += alpha * (desired_target[axis] - self._camera_target[axis])
        self._camera_state.set_position_world(Gf.Vec3d(*self._camera_position), True)
        self._camera_state.set_target_world(Gf.Vec3d(*self._camera_target), True)

    def on_timeline_stop(self, _event: object) -> None:
        self._physics_ready = False
        self._camera_update_elapsed = 0.0
        self._camera_position = None
        self._camera_target = None
        if self.go2 is not None:
            self.go2.invalidate_physics_handles()

    def _on_keyboard_event(self, event: object, *_args: object, **_kwargs: object) -> bool:
        raw_key_name = event.input if isinstance(event.input, str) else event.input.name
        key_name = raw_key_name.upper()
        key_name = {"NUMPADADD": "NUMPAD_ADD", "NUMPADSUBTRACT": "NUMPAD_SUBTRACT"}.get(
            key_name, key_name
        )
        if self._base_command is None:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and key_name in {"EQUAL", "NUMPAD_ADD"}:
            self._speed_scale = min(2.0, round(self._speed_scale + 0.1, 1))
            carb.log_info(f"Go2 command speed: {self._speed_scale:.1f}x")
            self._rebuild_command()
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and key_name in {"MINUS", "NUMPAD_SUBTRACT"}:
            self._speed_scale = max(0.1, round(self._speed_scale - 0.1, 1))
            carb.log_info(f"Go2 command speed: {self._speed_scale:.1f}x")
            self._rebuild_command()
            return True
        if key_name not in self._key_commands:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed_motion_keys.add(key_name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_motion_keys.discard(key_name)
        self._rebuild_command()
        return True

    def _rebuild_command(self) -> None:
        """Recompute the velocity command from held keys and the speed multiplier."""
        if self._base_command is None:
            return
        self._base_command.zero_()
        for key_name in self._pressed_motion_keys:
            self._base_command += torch.tensor(self._key_commands[key_name], device=self._base_command.device)
        self._base_command *= self._speed_scale
        limits = torch.tensor((2.0, 2.0, 2.0), device=self._base_command.device)
        self._base_command.copy_(torch.maximum(torch.minimum(self._base_command, limits), -limits))

    def physics_cleanup(self) -> None:
        if self._physics_callback_id is not None:
            try:
                SimulationManager.deregister_callback(self._physics_callback_id)
            except Exception as error:
                carb.log_warn(f"Could not deregister Go2 physics callback: {error}")
            self._physics_callback_id = None
        if self._timeline_stop_callback_id is not None:
            try:
                SimulationManager.deregister_callback(self._timeline_stop_callback_id)
            except Exception as error:
                carb.log_warn(f"Could not deregister Go2 timeline callback: {error}")
            self._timeline_stop_callback_id = None
        if self._sub_keyboard is not None and self._input is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
        self.go2 = None
        self._base_command = None
        self._camera_state = None
        self._camera_position = None
        self._camera_target = None
        self._camera_update_elapsed = 0.0
        self._pressed_motion_keys.clear()
        self._physics_ready = False
        self._restore_simulation_state()
