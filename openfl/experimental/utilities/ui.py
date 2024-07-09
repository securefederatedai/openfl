# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import webbrowser
from pathlib import Path

from openfl.experimental.utilities.metaflow_utils import DefaultCard, FlowGraph


class InspectFlow:

    def __init__(
        self,
        flow_obj,
        run_id,
        show_html=False,
        ds_root=f"{Path.home()}/.metaflow",
    ):
        self.ds_root = ds_root
        self.show_html = show_html
        self.run_id = run_id
        self.flow_name = flow_obj.__class__.__name__
        self._graph = FlowGraph(flow_obj.__class__)
        self._steps = [getattr(flow_obj, node.name) for node in self._graph]

        self.graph_dict, _ = self._graph.output_steps()
        self.show_ui()

    def get_pathspec(self):
        return f"{self.ds_root}/{self.flow_name}/{self.run_id}"

    def open_in_browser(self, card_path):
        url = "file://" + os.path.abspath(card_path)
        webbrowser.open(url)

    def show_ui(self):
        default_card = DefaultCard(graph=self.graph_dict)

        pathspec = self.get_pathspec()
        print(f"Flowgraph generated at :{pathspec}")
        html = default_card.render(pathspec)

        with open(f"{pathspec}/card_ui.html", "w") as f:
            f.write(html)

        if self.show_html:
            self.open_in_browser(f"{pathspec}/card_ui.html")
