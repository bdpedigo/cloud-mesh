from google.cloud import pubsub_v1
import json
import logging
from typing import Iterable, Optional
import os

from tqdm.auto import tqdm

log = logging.getLogger(__name__)

class PSTaskQueue:
    """
    Minimal Pub/Sub TaskQueue adapter.

    Accepts queue_url in one of these forms:
      - "pubsub://projects/PROJECT/topics/TOPIC"
      - "projects/PROJECT/topics/TOPIC"

    insert(callables) expects a list/iterable of functools.partial objects
    where each partial was created as partial(run_for_root, root_id, datastack).
    Each message published will be a JSON payload: {"root_id": ..., "datastack": "..."}
    """

    def __init__(self, queue_url: str):
        if queue_url.startswith("pubsub://"):
            queue_url = queue_url[len("pubsub://"):]
        # At this point we expect the full resource form: "projects/PROJECT/topics/TOPIC"
        if not queue_url.startswith("projects/"):
            # allow a simple topic name by falling back to project inferred from env or ADC
            # but prefer explicit full resource strings to avoid ambiguity
            project = os.environ.get("GCP_PROJECT")
            if project:
                queue_url = f"projects/{project}/topics/{queue_url}"
            else:
                raise ValueError(
                    "queue_url must be 'projects/PROJECT/topics/TOPIC', "
                    "'pubsub://projects/PROJECT/topics/TOPIC', or a simple TOPIC when GCP_PROJECT env var is set."
                )
        self.topic = queue_url
        self.publisher = pubsub_v1.PublisherClient()

    def insert(self, callables: Iterable, *, progress: Optional[bool] = None):
        """Publish one Pub/Sub message per callable.

        Parameters
        ----------
        callables : iterable of functools.partial
            Each partial must have been built as
            ``partial(run_for_root, root_id, datastack, ...)``.
        progress : bool, optional
            Show a tqdm progress bar while publishing. Defaults to True when
            ``callables`` has a known length (so we can show ETA), else False.
            Pass ``progress=False`` to silence completely.
        """
        # Materialize so we can show a total in the progress bar; this is
        # also necessary because we iterate exactly once.
        if not isinstance(callables, (list, tuple)):
            callables = list(callables)
        if progress is None:
            progress = True

        bar = tqdm(callables, desc="publishing", unit="msg", disable=not progress)
        for c in bar:
            args = getattr(c, "args", None)
            if not args or len(args) < 2:
                raise ValueError("Task partial missing expected args (root_id, datastack)")
            root_id = int(args[0])
            datastack = args[1]
            payload = {"root_id": root_id, "datastack": datastack}

            # Optional targeted-region fields: if the partial was built with
            # synapse_id / query_point_nm / query_radius_nm (as kwargs, or as
            # positional args 3, 4, 5), include them in the published payload
            # so the worker routes to the chunked HKS pipeline and tags the
            # output file by synapse_id.
            kwargs = getattr(c, "keywords", None) or {}
            synapse_id = kwargs.get("synapse_id")
            query_point_nm = kwargs.get("query_point_nm")
            query_radius_nm = kwargs.get("query_radius_nm")
            if synapse_id is None and len(args) >= 3:
                synapse_id = args[2]
            if query_point_nm is None and len(args) >= 4:
                query_point_nm = args[3]
            if query_radius_nm is None and len(args) >= 5:
                query_radius_nm = args[4]

            any_targeted = (
                synapse_id is not None
                or query_point_nm is not None
                or query_radius_nm is not None
            )
            all_targeted = (
                synapse_id is not None
                and query_point_nm is not None
                and query_radius_nm is not None
            )
            if any_targeted and not all_targeted:
                raise ValueError(
                    "synapse_id, query_point_nm, and query_radius_nm must be "
                    "supplied together for targeted-region tasks"
                )
            if all_targeted:
                payload["synapse_id"] = int(synapse_id)
                payload["query_point_nm"] = [float(coord) for coord in query_point_nm]
                payload["query_radius_nm"] = float(query_radius_nm)

            data = json.dumps(payload).encode("utf-8")
            # publish returns a Future; .result() blocks until the message has
            # actually been accepted by Pub/Sub. We keep that synchronous wait
            # so the progress bar reflects real publish progress (not just how
            # fast we can hand bytes to the client library).
            future = self.publisher.publish(self.topic, data=data)
            message_id = future.result()
            log.debug(
                "root_id=%s synapse_id=%s -> message_id=%s",
                root_id, payload.get("synapse_id"), message_id,
            )
