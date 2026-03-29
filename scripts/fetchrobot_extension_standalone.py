"""GUI-native runner for the FetchRobot scene.

RUN: 

cd ~/isaacsim
./isaac-sim.sh --python scripts/fetchrobot_extension_standalone.py


OR

Use when you want the normal Isaac Sim interface (Play/Stop in the UI) and
DON'T want to start a separate SimulationApp.

How to run
1) Open Isaac Sim normally: `./isaac-sim.sh`
2) Window -> Script Editor
3) Run this file, or paste:
   exec(open("scripts/fetchrobot_extension_gui.py").read())

It will build the scene and stop the timeline. Use the UI Play/Stop to run.
"""

import os

import numpy as np
import omni
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


class FetchRobotGuiScene:
    def __init__(self) -> None:
        self.stage = None
        self.world = None
        self.go2 = None

    def _enable_gpu_dynamics_for_deformables(self, stage):
        scene_prim = None
        for path in ("/physicsScene", "/World/physicsScene"):
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                scene_prim = prim
                break

        if scene_prim is None:
            scene_prim = UsdPhysics.Scene.Define(stage, "/World/physicsScene").GetPrim()

        scene = UsdPhysics.Scene(scene_prim)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

        if hasattr(physx_scene_api, "CreateEnableGPUDynamicsAttr"):
            physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        else:
            scene_prim.CreateAttribute(
                "physxScene:enableGPUDynamics",
                Sdf.ValueTypeNames.Bool,
            ).Set(True)

        if hasattr(physx_scene_api, "CreateBroadphaseTypeAttr"):
            physx_scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        else:
            scene_prim.CreateAttribute(
                "physxScene:broadphaseType",
                Sdf.ValueTypeNames.Token,
            ).Set("GPU")

    def _create_foam_material(self, cube_path: str = "/World/PlayCube") -> None:
        if self.stage is None:
            raise RuntimeError("Stage is not set")

        material_path = "/World/PhysicsMaterials/foammaterial"
        deformableUtils.add_deformable_body_material(
            self.stage,
            material_path,
            youngs_modulus=1.0e6,
            poissons_ratio=0.35,
            damping_scale=1.2,
            elasticity_damping=0.005,
            dynamic_friction=0.8,
            density=10.0,
        )

        cube_prim = self.stage.GetPrimAtPath(cube_path)
        if cube_prim and cube_prim.IsValid():
            physicsUtils.add_physics_material_to_prim(self.stage, cube_prim, material_path)

    def _apply_matte_foam_visual(self, cube_path: str = "/World/PlayCube") -> None:
        if self.stage is None:
            raise RuntimeError("Stage is not set")

        material_path = "/World/Looks/FoamVisualMaterial"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, f"{material_path}/PreviewSurface")

        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.8, 0.3))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.98)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))

        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        cube_prim = self.stage.GetPrimAtPath(cube_path)
        if cube_prim and cube_prim.IsValid():
            UsdShade.MaterialBindingAPI(cube_prim).Bind(material)

    def add_playcube(self) -> None:
        if self.stage is None:
            raise RuntimeError("Stage is not set")

        result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        if not result:
            raise RuntimeError("Failed to create cube prim")

        omni.kit.commands.execute("MovePrim", path_from=path, path_to="/World/PlayCube")
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

        cube_mesh = UsdGeom.Mesh.Get(self.stage, "/World/PlayCube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.5, 0.0, 0.5))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(0.095, 0.095, 0.095))
        cube_mesh.CreateDisplayColorAttr([(0.0, 0.8, 0.3)])

        deformableUtils.add_physx_deformable_body(
            self.stage,
            "/World/PlayCube",
            collision_simplification=True,
            simulation_hexahedral_resolution=12,
            self_collision=False,
            solver_position_iteration_count=32,
        )

        cube_prim = omni.usd.get_prim_at_path("/World/PlayCube")
        cube_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(cube_prim)
        cube_collision_api.CreateRestOffsetAttr().Set(0.0)
        cube_collision_api.CreateContactOffsetAttr().Set(0.01)

        self._create_foam_material("/World/PlayCube")
        self._apply_matte_foam_visual("/World/PlayCube")

    def build(self, urdf_path: str) -> None:
        self.world = World.instance()
        if self.world is None:
            self.world = World(stage_units_in_meters=1.0)

        self.world.scene.add_default_ground_plane(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.0,
        )

        self.stage = omni.usd.get_context().get_stage()
        self._enable_gpu_dynamics_for_deformables(self.stage)

        self.add_playcube()

        import_config = _urdf.ImportConfig()
        import_config.fix_base = False
        import_config.make_default_prim = True

        if not os.path.isfile(urdf_path):
            print(f"ERROR: URDF not found: {urdf_path}")
            return

        parse_ok, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )
        if not parse_ok or robot_model is None:
            print(f"ERROR: Failed to parse URDF: {urdf_path}")
            return

        import_ok, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=urdf_path,
            urdf_robot=robot_model,
            import_config=import_config,
        )
        if not import_ok or not prim_path:
            print("ERROR: Failed to import URDF robot into stage")
            return

        self.go2 = self.world.scene.add(
            SingleArticulation(
                prim_path=prim_path,
                name="go2_robot",
                position=np.array([0.0, 0.0, 0.8]),
            )
        )

        self.world.reset()


# Convenience: run once when executed.
_URDF_PATH = os.environ.get(
    "GO2_URDF_PATH",
    "/home/ferdinand/fetchrobot/unitree_ros/robots/go2_description/urdf/go2_description.urdf",
)

scene = FetchRobotGuiScene()
scene.build(_URDF_PATH)

# Leave timeline stopped; user controls play/stop in UI.
omni.timeline.get_timeline_interface().stop()
_CMD = 'exec(open("scripts/fetchrobot_extension_gui.py").read())'
print("FetchRobot GUI scene ready. Use the Isaac Sim Play/Stop controls.")
print("Re-run from Script Editor with:")
print(_CMD)
