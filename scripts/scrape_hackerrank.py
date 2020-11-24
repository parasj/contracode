import json
import math
import os
from pathlib import Path
import random
import time
import urllib.request

import fire
import requests
from tqdm import tqdm

DEFAULT_SESSION = (
    "da846953a352396a9985ecac686afac3405fc4e3ac7754794ed3134cbf11f53177b3097f58d795829e354db6de886e24c3d2ea1fed28c7d96759d76be3b210e5"
)


class HackerRankAPI:
    def __init__(self, hr_session):
        self.download_url = (
            lambda challenge, hacker: f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/hackers/{hacker}/download_solution"
        )
        self.challenge_list_url = (
            lambda track, offset, limit: f"https://www.hackerrank.com/rest/contests/master/tracks/{track}/challenges?offset={offset}&limit={limit}"
        )
        self.unlock_url = lambda challenge: f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/unlock_solution"
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.72 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 Safari/537.85.10",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
            "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36",
        ]
        self.hr_session = hr_session

    @staticmethod
    def get_leaderboard_url(challenge, language=None, offset=0, limit=50):
        if language is not None:
            url = f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/leaderboard/filter?include_practice=true&language={language}&filter_kinds=language"
        else:
            url = f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/leaderboard?include_practice=true"
        if offset is not None:
            url += f"&offset={offset}&limit={limit}"
        return url

    def fetch_hackerrank_url(self, url, json_decode=False):
        headers = {
            "user-agent": str(random.choice(self.user_agents)),
            "cookie": f"_hrank_session={self.hr_session}; hackerrank_mixpanel_token=826cd42f-383b-404d-956d-73ca0a128c7a; hrc_l_i=T; metrics_user_identifier=9af15a-f3ca15647ccd70ff12c935ba95dd57780a549a3f",
            "cache-control": "max-age=0",
        }

        response = requests.request("GET", url, headers=headers, data={})
        if response.status_code != 200:
            print("ERROR: Got", url, "with status", response.status_code)
            return None

        if json_decode:
            return response.json()
        else:
            return response.text

    def get_challenge_list(self, track, limit=500, per_page_limit=50):
        tracks = []
        track_id_set = set()
        for i in tqdm(range(math.ceil(limit / per_page_limit)), desc=f"Get challenge list from '{track}'", leave=False):
            ntrack_count = len(track_id_set)
            result = self.fetch_hackerrank_url(self.challenge_list_url(track, per_page_limit * i, per_page_limit), json_decode=True)
            for challenge in tqdm(result["models"], desc="    -> Download solution", leave=False):
                if challenge["slug"] not in track_id_set:
                    track_id_set.add(challenge["slug"])
                    tracks.append(challenge)
            if ntrack_count == len(track_id_set):  # early return
                return tracks
        return tracks

    def unlock_challenge(self, challenge):
        self.fetch_hackerrank_url(self.unlock_url(challenge))

    def get_submission_list(self, challenge, language="javascript", limit=50, per_page_limit=50):
        solution_urls = []
        solution_hacker_set = set()
        for i in range(math.ceil(limit / per_page_limit)):
            nsol_count = len(solution_hacker_set)
            solutions = self.fetch_hackerrank_url(
                self.get_leaderboard_url(challenge, language="javascript", offset=per_page_limit * i), json_decode=True
            )
            for sol in solutions["models"]:
                if sol["hacker"] not in solution_hacker_set and sol["hacker"] != "[deleted]" and sol["language"] == language:
                    solution_hacker_set.add(sol["hacker"])
                    solution_urls.append(sol)
            if len(solution_hacker_set) == nsol_count:  # early return if no new solutions
                return solution_urls
        return solution_urls

    def download_solution(self, sol, challenge):
        if "download_link" in sol.keys():
            url = "https://www.hackerrank.com" + sol["download_link"]
        else:
            url = self.download_url(challenge, sol["hacker"])
        try:
            return self.fetch_hackerrank_url(url)
        except Exception as e:
            tqdm.write(f"ERROR downloading URL {url}")
            tqdm.write(e)
            return None


def get_challenge_list(session=DEFAULT_SESSION, track="algorithms"):
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "hackerrank"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracks = HackerRankAPI(session).get_challenge_list(track)
    challenges = [x["slug"] for x in tracks]
    with (log_dir / f"{track}_challenge_index.json").open("w") as f:
        json.dump(tracks, f)
    with (log_dir / f"{track}_challenge_list.txt").open("w") as f:
        f.write("\n".join(challenges))
    print("Got", len(challenges), "challenges")


def get_challenge_submissions(challenge, session=DEFAULT_SESSION, language="javascript", limit=10):
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "hackerrank"
    challenge_dir = log_dir / challenge
    download_dir = challenge_dir / "src"
    download_dir.mkdir(parents=True, exist_ok=True)

    # get metadata
    api = HackerRankAPI(session)
    api.unlock_challenge(challenge)
    solution_urls = api.get_submission_list(challenge, language, limit=limit)
    with (challenge_dir / "challenge_data_no_source.json").open("w") as f:
        json.dump(solution_urls, f)

    # download URLs
    for sol in tqdm(solution_urls, desc=f"Downloading {len(solution_urls)} URLs for challenge {challenge}"):
        time.sleep(0.1)
        sol["src"] = api.download_solution(sol, challenge)
        if sol["src"] != None:
            ext = sol["language"] if sol["language"] != "javascript" else "js"
            with (download_dir / f"{sol['score']}_{sol['hacker_id']}.{ext}").open("w") as f:
                f.write(sol["src"])

    # save final output
    with (challenge_dir / "challenge_data.json").open("w") as f:
        json.dump(solution_urls, f)


if __name__ == "__main__":
    fire.Fire(
        {"get_challenge_list": get_challenge_list, "get_challenge_submissions": get_challenge_submissions,}
    )
