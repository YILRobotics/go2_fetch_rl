"""PhysX integration test for the bundled Go2 4L velocity policy."""

import asyncio

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
import omni.timeline
import torch
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.storage.native import get_assets_root_path

from ferdinand.go2_walking.policy import Go2VelocityPolicy

class TestGo2Policy(omni.kit.test.AsyncTestCase):
    async def setUp(self) -> None:
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
        SimulationManager.set_backend("torch")
        SimulationManager.set_physics_sim_device("cuda")
        SimulationManager.set_physics_dt(0.005)
        stage_utils.add_reference_to_stage(
            usd_path=f"{get_assets_root_path()}/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )
        self.timeline = omni.timeline.get_timeline_interface()
        self.command = torch.zeros(3, device="cuda")
        self.callback_id = None

    async def tearDown(self) -> None:
        self.timeline.stop()
        if self.callback_id is not None:
            SimulationManager.deregister_callback(self.callback_id)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await asyncio.sleep(0.1)
        await omni.kit.app.get_app().next_update_async()

    async def test_policy_observation_and_motion(self) -> None:
        go2 = Go2VelocityPolicy(prim_path="/World/Go2", position=(0.0, 0.0, 0.5))
        self.timeline.play()
        await omni.kit.app.get_app().next_update_async()
        go2.initialize()
        go2.post_reset()
        observation = go2.compute_observation(self.command)
        self.assertEqual(tuple(observation.shape), (1, 49))
        self.assertTrue(bool(torch.isfinite(observation).all()))

        def on_step(dt: float, _context: object | None = None) -> None:
            go2.forward(dt, self.command)

        self.callback_id = SimulationManager.register_callback(on_step, IsaacEvents.POST_PHYSICS_STEP)
        start_position = go2.robot.get_world_poses()[0].numpy()[0]
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
        standing_position = go2.robot.get_world_poses()[0].numpy()[0]
        self.assertGreater(standing_position[2], 0.2)
        self.assertLess(float(np.linalg.norm(standing_position[:2] - start_position[:2])), 0.5)

        self.command[0] = 0.5
        for _ in range(160):
            await omni.kit.app.get_app().next_update_async()
        moving_position = go2.robot.get_world_poses()[0].numpy()[0]
        self.assertGreater(float(moving_position[0] - standing_position[0]), 0.1)
