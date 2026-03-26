from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

import numpy as np
import os
import omni.usd
import omni.kit.commands

from isaacsim.asset.importer.urdf import _urdf
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt


class FetchRobot(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._deformable_cube_path = "/World/FoamCube"
        self._deformable_cube_prim = None
        self._go2 = None

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
                Sdf.ValueTypeNames.Bool
            ).Set(True)

        if hasattr(physx_scene_api, "CreateBroadphaseTypeAttr"):
            physx_scene_api.CreateBroadphaseTypeAttr().Set("GPU")
        else:
            scene_prim.CreateAttribute(
                "physxScene:broadphaseType",
                Sdf.ValueTypeNames.Token
            ).Set("GPU")

    def _create_foam_material(self, stage, path="/World/Materials/FoamMaterial"):
        material = UsdShade.Material.Define(stage, path)
        prim = material.GetPrim()

        prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
        prim.GetAttribute("omniphysics:staticFriction").Set(0.5)
        prim.GetAttribute("omniphysics:dynamicFriction").Set(0.4)
        prim.GetAttribute("omniphysics:density").Set(120.0)

        prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
        prim.GetAttribute("omniphysics:youngsModulus").Set(5.0e4)
        prim.GetAttribute("omniphysics:poissonsRatio").Set(0.30)

        prim.ApplyAPI("PhysxDeformableMaterialAPI")
        prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.25)

        return material

    def _create_simulation_tetmesh(self, stage, path, center=(0.2, 0.0, 1.0), size=0.095):
        tet_mesh = UsdGeom.TetMesh.Define(stage, path)

        cx, cy, cz = center
        s = size * 0.5

        points = Vt.Vec3fArray([
            Gf.Vec3f(cx - s, cy - s, cz - s),  # 0
            Gf.Vec3f(cx + s, cy - s, cz - s),  # 1
            Gf.Vec3f(cx + s, cy + s, cz - s),  # 2
            Gf.Vec3f(cx - s, cy + s, cz - s),  # 3
            Gf.Vec3f(cx - s, cy - s, cz + s),  # 4
            Gf.Vec3f(cx + s, cy - s, cz + s),  # 5
            Gf.Vec3f(cx + s, cy + s, cz + s),  # 6
            Gf.Vec3f(cx - s, cy + s, cz + s),  # 7
        ])

        # Valid 5-tet cube decomposition
        tet_indices = Vt.Vec4iArray([
            Gf.Vec4i(0, 1, 3, 4),
            Gf.Vec4i(1, 2, 3, 6),
            Gf.Vec4i(1, 4, 5, 6),
            Gf.Vec4i(3, 4, 6, 7),
            Gf.Vec4i(1, 3, 4, 6),
        ])

        tet_mesh.GetPointsAttr().Set(points)
        tet_mesh.GetTetVertexIndicesAttr().Set(tet_indices)

        # Required so the tet mesh surface can be used for collision
        surface_faces = UsdGeom.TetMesh.ComputeSurfaceFaces(tet_mesh)
        tet_mesh.GetSurfaceFaceVertexIndicesAttr().Set(surface_faces)

        # Simple visible color
        tet_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.75, 0.2)])

        return tet_mesh

    def _create_single_mesh_volume_deformable(self, stage, path):
        tet_mesh = self._create_simulation_tetmesh(
            stage=stage,
            path=path,
            center=(0.2, 0.0, 1.0),
            size=0.095,
        )
        tet_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])
        UsdGeom.Imageable(tet_mesh).MakeVisible()
        tet_mesh.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
        prim = tet_mesh.GetPrim()
        print("Created cube at:", prim.GetPath())
        print("Valid:", prim.IsValid())
        print("Points:", tet_mesh.GetPointsAttr().Get())

        # Mark as deformable body
        prim.ApplyAPI("OmniPhysicsDeformableBodyAPI")
        prim.GetAttribute("omniphysics:mass").Set(0.2)
        if prim.HasAttribute("omniphysics:kinematicEnabled"):
            prim.GetAttribute("omniphysics:kinematicEnabled").Set(False)
        if prim.HasAttribute("omniphysics:startsAsleep"):
            prim.GetAttribute("omniphysics:startsAsleep").Set(False)

        # Volume simulation rest shape
        prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI")
        prim.GetAttribute("omniphysics:restShapePoints").Set(
            tet_mesh.GetPointsAttr().Get()
        )
        prim.GetAttribute("omniphysics:restTetVtxIndices").Set(
            tet_mesh.GetTetVertexIndicesAttr().Get()
        )

        # Collision on tet mesh surface
        UsdPhysics.CollisionAPI.Apply(prim)

        # PhysX-specific deformable settings
        prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
        if prim.HasAttribute("physxDeformableBody:disableGravity"):
            prim.GetAttribute("physxDeformableBody:disableGravity").Set(False)

        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        if physx_collision_api:
            physx_collision_api.GetContactOffsetAttr().Set(0.02)
            physx_collision_api.GetRestOffsetAttr().Set(0.005)

        # Foam-like material
        foam_material = self._create_foam_material(stage)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            foam_material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )

        return prim

    def setup_scene(self):
        print("SETUP_SCENE STARTING FROM MY FILE")

        world = World.instance()
        world.scene.add_default_ground_plane(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.0,
        )

        stage = omni.usd.get_context().get_stage()
        self._enable_gpu_dynamics_for_deformables(stage)

        self._deformable_cube_prim = self._create_single_mesh_volume_deformable(
            stage=stage,
            path=self._deformable_cube_path,
        )

        if self._deformable_cube_prim is None:
            print("ERROR: Failed to create deformable cube")
            print("SETUP_SCENE FINISHED (without deformable cube)")
            return

        import_config = _urdf.ImportConfig()
        import_config.fix_base = False
        import_config.make_default_prim = True

        urdf_path = os.environ.get(
            "GO2_URDF_PATH",
            "/home/ferdinand/fetchrobot/unitree_ros/robots/go2_description/urdf/go2_description.urdf",
        )

        if not os.path.isfile(urdf_path):
            print(f"ERROR: URDF not found: {urdf_path}")
            print("SETUP_SCENE FINISHED (without robot)")
            return

        parse_ok, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )

        if not parse_ok or robot_model is None:
            print(f"ERROR: Failed to parse URDF: {urdf_path}")
            print("SETUP_SCENE FINISHED (without robot)")
            return

        import_ok, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=urdf_path,
            urdf_robot=robot_model,
            import_config=import_config,
        )

        if not import_ok or not prim_path:
            print("ERROR: Failed to import URDF robot into stage")
            print("SETUP_SCENE FINISHED (without robot)")
            return

        self._go2 = world.scene.add(
            SingleArticulation(
                prim_path=prim_path,
                name="go2_robot",
                position=np.array([0.0, 0.0, 0.8]),
            )
        )

        print("SETUP_SCENE FINISHED")

    async def setup_post_load(self):
        self._world = self.get_world()
        try:
            self._go2 = self._world.scene.get_object("go2_robot")
        except Exception:
            self._go2 = None

        stage = omni.usd.get_context().get_stage()
        self._deformable_cube_prim = stage.GetPrimAtPath(self._deformable_cube_path)

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return