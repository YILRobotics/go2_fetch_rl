import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from .fetchrobot import FetchRobot

class FetchRobotExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Fetch Robot"
        self.category = "#Fetchrobot"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Fetch Robot Example",
            "overview": "This is for testing the Unitree Go2 fetchrobot python code and environment setup.",
            "sample": FetchRobot(), 
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return
