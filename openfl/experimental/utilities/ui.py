# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import webbrowser
from pathlib import Path

from openfl.experimental.utilities.metaflow_utils import DefaultCard, FlowGraph


class InspectFlow:
    """Class for inspecting a flow.

    Attributes:
        ds_root (str): The root directory for the data store. Defaults to
            "~/.metaflow".
        show_html (bool): Whether to show the UI in a web browser. Defaults to
            False.
        run_id (str): The run ID of the flow.
        flow_name (str): The name of the flow.
        graph_dict (dict): The graph of the flow.
    """

    def __init__(
        self,
        flow_obj,
        run_id,
        show_html=False,
        ds_root=f"{Path.home()}/.metaflow",
    ):
        """Initializes the InspectFlow with a flow object, run ID, an optional
        flag to show the UI in a web browser, and an optional root directory
        for the data store.

        Args:
            flow_obj (Flow): The flow object to inspect.
            run_id (str): The run ID of the flow.
            show_html (bool, optional): Whether to show the UI in a web
                browser. Defaults to False.
            ds_root (str, optional): The root directory for the data store.
                Defaults to "~/.metaflow".
        """
        self.ds_root = ds_root
        self.show_html = show_html
        self.run_id = run_id
        self.flow_name = flow_obj.__class__.__name__
        self._graph = FlowGraph(flow_obj.__class__)
        self._steps = [getattr(flow_obj, node.name) for node in self._graph]

        self.graph_dict, _ = self._graph.output_steps()
        self.show_ui()

    def get_pathspec(self):
        """Gets the path specification of the flow.

        Returns:
            str: The path specification of the flow.
        """
        return f"{self.ds_root}/{self.flow_name}/{self.run_id}"

    def open_in_browser(self, card_path):
        """Opens the specified path in a web browser.

        Args:
            card_path (str): The path to open.
        """
        url = "file://" + os.path.abspath(card_path)
        webbrowser.open(url)

    def show_ui(self):
        """Shows the UI of the flow in a web browser if show_html is True, and
        saves the UI as an HTML file."""

        default_card = DefaultCard(graph=self.graph_dict)

        pathspec = self.get_pathspec()
        print(f"Flowgraph generated at :{pathspec}")
        html = default_card.render(pathspec)

        with open(f"{pathspec}/card_ui.html", "w") as f:
            f.write(html)

        if self.show_html:
            self.open_in_browser(f"{pathspec}/card_ui.html")
