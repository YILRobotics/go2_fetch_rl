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
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .policy import Go2VelocityPolicy


class Go2WalkingExample(BaseSample):
    """Spawn the training-model Go2 and drive its velocity command by keyboard."""

    CAMERA_PATH = "/World/Go2FollowCamera"
    CAMERA_OFFSET = (-3.0, -4.0, 2.5)
    CAMERA_TARGET_HEIGHT = 0.1
    CAMERA_FOLLOW_ALPHA = (0.05, 0.05, 0.03)

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
        physics_material.CreateStaticFrictionAttr().Set(1.8)
        physics_material.CreateDynamicFrictionAttr().Set(1.8)
        physics_material.CreateRestitutionAttr().Set(0.0)
        ground = stage.GetPrimAtPath("/World/ground/GroundPlane/CollisionPlane")
        if ground.IsValid():
            UsdShade.MaterialBindingAPI.Apply(ground).Bind(material)

    @staticmethod
    def _add_cinematic_lighting() -> None:
        """Add a warm soft key, cool rim, and subdued ambient fill."""
        stage = omni.usd.get_context().get_stage()

        def add_rect_light(
            path: str,
            position: Gf.Vec3d,
            target: Gf.Vec3d,
            color: Gf.Vec3f,
            intensity: float,
            exposure: float,
            size: tuple[float, float],
        ) -> None:
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateWidthAttr(size[0])
            light.CreateHeightAttr(size[1])
            light.CreateIntensityAttr(intensity)
            light.CreateExposureAttr(exposure)
            light.CreateColorAttr(color)
            light.CreateNormalizeAttr(True)

            orientation = Gf.Matrix4d().SetLookAt(position, target, Gf.Vec3d(0.0, 0.0, 1.0))
            orientation = orientation.GetInverse().ExtractRotation().GetQuat()
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(position)
            xform.AddOrientOp().Set(Gf.Quatf(orientation))

        # Large warm source from camera-left: the main modelling light.
        add_rect_light(
            path="/World/Go2CinematicKey",
            position=Gf.Vec3d(-4.0, -5.0, 6.0),
            target=Gf.Vec3d(1.0, 0.0, 0.5),
            color=Gf.Vec3f(1.0, 0.82, 0.68),
            intensity=3500.0,
            exposure=1.0,
            size=(7.0, 7.0),
        )

        # Cooler back light outlines the robot against the environment.
        add_rect_light(
            path="/World/Go2CinematicRim",
            position=Gf.Vec3d(4.0, 4.0, 3.5),
            target=Gf.Vec3d(0.5, 0.0, 0.45),
            color=Gf.Vec3f(0.55, 0.70, 1.0),
            intensity=1800.0,
            exposure=0.5,
            size=(4.0, 4.0),
        )

        ambient = UsdLux.DomeLight.Define(stage, "/World/Go2CinematicAmbient")
        ambient.CreateIntensityAttr(250.0)
        ambient.CreateExposureAttr(0.0)
        ambient.CreateColorAttr(Gf.Vec3f(0.58, 0.68, 0.9))

    @staticmethod
    def _add_obstacles() -> None:
        """Add a shallow ramp and three loose cubes in front of the robot."""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/Go2Obstacles")

        ramp_material = UsdShade.Material.Define(stage, "/World/Go2Obstacles/RampPhysicsMaterial")
        ramp_physics_material = UsdPhysics.MaterialAPI.Apply(ramp_material.GetPrim())
        ramp_physics_material.CreateStaticFrictionAttr().Set(2.5)
        ramp_physics_material.CreateDynamicFrictionAttr().Set(2.5)
        ramp_physics_material.CreateRestitutionAttr().Set(0.0)

        ramp_specs = (
            ("RampUp", 2.0, -10.0),
            ("RampDown", 3.48, 10.0),
        )
        for name, x, angle in ramp_specs:
            ramp = UsdGeom.Cube.Define(stage, f"/World/Go2Obstacles/{name}")
            ramp.CreateSizeAttr(1.0)
            ramp.CreateDisplayColorAttr([Gf.Vec3f(0.28, 0.32, 0.38)])
            ramp_xform = UsdGeom.Xformable(ramp.GetPrim())
            ramp_xform.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.155))
            ramp_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, angle, 0.0))
            ramp_xform.AddScaleOp().Set(Gf.Vec3f(1.5, 1.2, 0.05))
            UsdPhysics.CollisionAPI.Apply(ramp.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(ramp.GetPrim()).Bind(
                ramp_material,
                bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics",
            )

        block_specs = (
            ("Block15cm", 0.15, (0.0, -1.0)),
            ("Block20cm", 0.20, (0.8, -1.0)),
            ("Block30cm", 0.30, (1.6, -1.0)),
        )
        colors = (
            Gf.Vec3f(0.75, 0.32, 0.20),
            Gf.Vec3f(0.82, 0.55, 0.18),
            Gf.Vec3f(0.62, 0.22, 0.18),
        )
        for (name, size, (x, y)), color in zip(block_specs, colors):
            cube = UsdGeom.Cube.Define(stage, f"/World/Go2Obstacles/{name}")
            cube.CreateSizeAttr(size)
            cube.CreateDisplayColorAttr([color])
            cube_xform = UsdGeom.Xformable(cube.GetPrim())
            cube_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, size * 0.5 + 0.01))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateDensityAttr(100.0)

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
        self._add_cinematic_lighting()
        self._add_obstacles()
        self.go2 = Go2VelocityPolicy(prim_path="/World/Go2", position=(0.0, 0.0, 0.5))
        UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), self.CAMERA_PATH)
        set_camera_view(
            eye=(-4.0, -4.0, 2.5),
            target=(0.0, 0.0, 0.2),
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
