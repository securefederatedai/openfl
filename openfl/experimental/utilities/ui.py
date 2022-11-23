from metaflow.plugins.cards.card_modules.basic import DefaultCard
from metaflow.client import Task
from pathlib import Path
import os
import webbrowser

class InspectFlow:
    def __init__(self, graph_dict, run_id, flow_name, show_html=False,ds_root=f'{Path.home()}/.metaflow'):
        self.ds_root = ds_root
        self.show_html = show_html
        self.run_id = run_id
        self.flow_name = flow_name
        print(f'flow name = {flow_name}')
        self.graph_dict = graph_dict
        self.show_ui()

    def get_pathspec(self):
        return f'{self.ds_root}/{self.flow_name}/{self.run_id}/start/1'

    def open_in_browser(self, card_path):
        url = "file://" + os.path.abspath(card_path)
        webbrowser.open(url)

    def show_ui(self):
        default_card = DefaultCard(graph=self.graph_dict)

        pathspec = self.get_pathspec()
        print(pathspec)
        import basic
        html = default_card.render(pathspec)

        with open(f'{pathspec}/card_ui.html', "w") as f:
            f.write(html)

        if self.show_html:
            self.open_in_browser("./card_ui.html")
