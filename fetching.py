import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import time
from collections import deque
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

players_url = 'https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&start=0&limit=-1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20262027%20and%20seasonId%3E=20262027'
# players_url = 'https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&start=0&limit=100&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20262027%20and%20seasonId%3E=20262027'
# players_url = 'https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&start=0&limit=-1&cayenneExp=franchiseId%3D25%20and%20gameTypeId=2%20and%20seasonId%3C=20262027%20and%20seasonId%3E=20262027'
edge_url_by_player = "https://api-web.nhle.com/v1/edge/skater-detail/{player_id}/now"
_thread_local = threading.local()
logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_requests=25, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds

                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                wait_time = self.window_seconds - (now - self._timestamps[0])
                if wait_time <= 0:
                    self._timestamps.popleft()
                    continue

            time.sleep(wait_time + 0.05)


if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(
        total=0,
        status_forcelist=(429,),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _build_session()
        _thread_local.session = session
    return session


def get_regular_stats(url=players_url):
    logger.info("Fetching regular skater stats...")
    response = _get_session().get(url, timeout=(5, 25))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse JSON data automatically
        data = response.json()
        logger.info("Fetched %d skater rows", len(data.get("data", [])))
        return data["data"]
    else:
        logger.error("Failed to fetch NHL player data. Status code: %s", response.status_code)

def get_edge_data_for_player(player_id, rate_limiter, timeout=15):
    url = edge_url_by_player.format(player_id=player_id)

    try:
        rate_limiter.acquire()
        response = _get_session().get(url, timeout=(5, timeout))
        if response.status_code == 200:
            # Return full EDGE payload so downstream flattening can include
            # skating/shot/location numeric metrics, not only identity info.
            return response.json()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 10.0
            logger.warning("Rate limited for player %s. Sleeping %.1f seconds.", player_id, wait)
            time.sleep(wait)
            return {}

        logger.warning("Failed to fetch EDGE data for player %s. Status code: %s", player_id, response.status_code)
        return {}
    except requests.RequestException as e:
        logger.warning("Failed to fetch EDGE data for player %s. Error: %s", player_id, e)
        return {}

def get_edge_data(data, max_workers=3):
    if not data:
        logger.info("No data rows provided for EDGE enrichment")
        return []

    enhanced_data = [None] * len(data)
    edge_by_player_id = {}
    unique_player_ids = list(dict.fromkeys(row["playerId"] for row in data))
    total = len(unique_player_ids)
    started_at = time.perf_counter()
    rate_limiter = RateLimiter(max_requests=25, window_seconds=60)

    max_workers = min(max_workers, max(1, len(unique_player_ids)))
    logger.info(
        "Starting EDGE fetch: %d rows, %d unique players, %d workers (25 req/min budget)",
        len(data),
        total,
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_player = {
            executor.submit(get_edge_data_for_player, player_id, rate_limiter): player_id
            for player_id in unique_player_ids
        }

        for completed_count, future in enumerate(as_completed(future_to_player), start=1):
            player_id = future_to_player[future]
            try:
                edge_by_player_id[player_id] = future.result() or {}
            except Exception as e:
                logger.warning("Failed to fetch EDGE data for player %s: %s", player_id, e)
                edge_by_player_id[player_id] = {}

            elapsed = time.perf_counter() - started_at
            rate = completed_count / elapsed if elapsed > 0 else 0
            logger.info(
                "EDGE progress: %d/%d (%.0f%%) | %.2f req/s",
                completed_count,
                total,
                (completed_count / total) * 100,
                rate,
            )

    for i, row in enumerate(data):
        player_id = row["playerId"]
        edge_data = edge_by_player_id.get(player_id, {})
        enhanced_data[i] = row | edge_data if edge_data else row

    logger.info("Completed EDGE enrichment for %d rows", len(enhanced_data))
    return enhanced_data