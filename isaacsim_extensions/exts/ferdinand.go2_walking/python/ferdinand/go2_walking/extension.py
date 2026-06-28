"""Examples-browser registration for Go2 4L walking."""

import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample.base_sample_extension_experimental import BaseSampleUITemplate

from .example import Go2WalkingExample


class Go2WalkingExtension(omni.ext.IExt):
    """Register the interactive Go2 walking example."""

    def on_startup(self, ext_id: str) -> None:
        self._name = "Go2 4L Walking"
        self._category = "#Fetchrobot"
        overview = (
            "Run the latest 4L foot-force velocity policy on the Unitree Go2 training model.\n\n"
            "Keyboard: arrows/numpad 8,2,4,6 move; N/M or numpad 7/9 turn.\n"
            "+/- or numpad +/- changes all velocity commands from 0.1x to 2.0x.\n"
            "This policy supports PhysX only."
        )
        ui = BaseSampleUITemplate(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title=self._name,
            overview=overview,
            sample=Go2WalkingExample(),
        )
        get_browser_instance().register_example(
            name=self._name,
            ui_hook=ui.build_ui,
            category=self._category,
        )

    def on_shutdown(self) -> None:
        get_browser_instance().deregister_example(name=self._name, category=self._category)
