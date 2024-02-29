# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.utilities.metaflow_utils import DefaultCard, FlowGraph
from pathlib import Path
import os
import webbrowser


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
