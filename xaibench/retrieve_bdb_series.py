import asyncio
import os

import aiohttp
from aiohttp.client_exceptions import (
    ClientConnectorError,
    ClientPayloadError,
    ServerDisconnectedError,
)
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

from xaibench.utils import DATA_PATH

SET_PATH = os.path.join(DATA_PATH, "validation_sets")
SEM_VALUE = 30
LIM_CONN = 30
WAIT_TIME = 5


async def fetch(sem, client, url):
    """
    Fetches all associated ligands for a particular BindingDB
    protein-ligand validation set entry, asynchronously.
    """
    retrieved = False
    async with sem:
        while not retrieved:
            try:
                async with client.get(url) as resp:
                    assert resp.status == 200
                    tsv_text = await resp.text()
                    retrieved = True

                    # store tsv
                    name = url.split("/")[-1].split("_")[0]
                    output_f = os.path.join(SET_PATH, name)
                    os.makedirs(output_f, exist_ok=True)

                    with open(
                        os.path.join(output_f, "{}.tsv".format(name)), "w"
                    ) as handle:
                        handle.write(tsv_text)

            except (ServerDisconnectedError, ClientConnectorError, ClientPayloadError):
                print("Server disconnected, waiting and retrying...")
                await asyncio.sleep(WAIT_TIME)


async def main(tsvs):
    sem = asyncio.Semaphore(SEM_VALUE)
    conn = aiohttp.TCPConnector(limit=LIM_CONN, limit_per_host=LIM_CONN)
    async with aiohttp.ClientSession(
        connector=conn, headers={"Connection": "close"}
    ) as client:
        tasks = [asyncio.ensure_future(fetch(sem, client, tsv)) for tsv in tsvs]
        [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__":
    with open(
        os.path.join(DATA_PATH, "bindingdb.html"), encoding="iso-8859-1"
    ) as handle:
        bdb_html = handle.read()

    soup = BeautifulSoup(bdb_html, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    tsvs = [link for link in links if ".tsv" in link]

    os.makedirs(SET_PATH, exist_ok=True)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(tsvs))
