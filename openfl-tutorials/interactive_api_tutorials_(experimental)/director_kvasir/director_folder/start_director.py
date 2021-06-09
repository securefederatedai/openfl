import asyncio
import logging
from rich.console import Console
from rich.logging import RichHandler
from openfl.component.director.director import serve

root = logging.getLogger()
root.setLevel(logging.INFO)
console = Console(width=160)
handler = RichHandler(console=console)
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)
handler.setFormatter(formatter)
root.addHandler(handler)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sample_shape = ['529', '622', '3']
    target_shape = ['529', '622']
    asyncio.run(serve(sample_shape=list(sample_shape), target_shape=list(target_shape)))
