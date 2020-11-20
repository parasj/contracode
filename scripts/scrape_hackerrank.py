import json
import math
import os
from pathlib import Path
import time
import urllib.request

from tqdm import tqdm

HACKER_TOKEN = "BAhbCFsGaQNa8ZpJIiIkMmEkMTAkcDF5QjJ4UDJVcGhCUGhtNzFrSUF4dQY6BkVUSSIWMTYwNTA0MzAyMy44NTg2NjQGOwBG--d3b4ed2afa6bce50168fc3a997c2fe79b45b9337"
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'


def fetch_hackerrank_url(url, hacker_token=HACKER_TOKEN, json_decode=False):
    req = urllib.request.Request(url)
    req.add_header('Cookie', f"remember_hacker_token={hacker_token}; user_type=hacker")
    req.add_header('User-Agent', USER_AGENT)
    with urllib.request.urlopen(req) as url:
        data = url.read().decode()
        if json_decode:
            return json.loads(data)
        else:
            return data


def get_challenge_submissions(challenge, language=None, limit=50, per_page_limit=50):
    def get_unlock_url(challenge):
        return f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/unlock_solution"

    def get_leaderboard_url(challenge, language=None, offset=0):
        url = f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/leaderboard?include_practice=true"
        if offset is not None:
            url += f"&offset={offset}&limit=100"
        if language is not None:
            url += f"&language={language}&filter_kinds=language"
        return url

    def get_solution_url(challenge, username):
        return f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/hackers/{username}/download_solution"
    
    _ = fetch_hackerrank_url(get_unlock_url(challenge))

    solution_urls = []
    solution_hacker_set = set()
    for i in tqdm(range(math.ceil(limit / per_page_limit)), desc=f"    -> Load challenges from {challenge}", leave=False):
        nsol_count = len(solution_hacker_set)
        solutions = fetch_hackerrank_url(get_leaderboard_url(challenge, language='javascript', offset=per_page_limit * i), json_decode=True)
        for sol in solutions['models']:
            if sol['hacker'] not in solution_hacker_set and sol['hacker'] != '[deleted]':
                solution_hacker_set.add(sol['hacker'])
                sol['download_url'] = get_solution_url(challenge, sol['hacker'])
                solution_urls.append(sol)
        if len(solution_hacker_set) == nsol_count:  # early return if no new solutions
            return solution_urls
        time.sleep(0.25)
    return solution_urls


def get_challenge_list(track, limit=500, per_page_limit=50):
    def get_challenge_index_url(track, offset=0, limit=50):
        return f"https://www.hackerrank.com/rest/contests/master/tracks/{track}/challenges?offset={offset}&limit=50"
    
    tracks = []
    track_id_set = set()
    for i in tqdm(range(math.ceil(limit / per_page_limit)), desc=f"Get challenge list from '{track}'", leave=False):
        ntrack_count = len(track_id_set)
        url = get_challenge_index_url(track, per_page_limit * i, limit=per_page_limit)
        result = fetch_hackerrank_url(url, json_decode=True)
        for challenge in result['models']:
            if challenge['slug'] not in track_id_set:
                track_id_set.add(challenge['slug'])
                tracks.append(challenge)
        if ntrack_count == len(track_id_set):  # early return
            return tracks
        time.sleep(0.25)
    return tracks


def main():
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data"

    print("Loading challenges")
    tracks = get_challenge_list('algorithms')
    challenges = [x['slug'] for x in tracks]
    with (log_dir / 'challenge_index.json').open('w') as f:
        json.dump(tracks, f)
    print("Got", len(challenges), "challenges")


    for challenge in tqdm(challenges, desc="Challenge list"):
        print("Processing challenge", challenge)
        challenge_dir = (log_dir / challenge)
        challenge_dir.mkdir(parents=True, exist_ok=True)

        urls = get_challenge_submissions(challenge, language='javascript')
        with (challenge_dir / 'url_index.json').open('w') as f:
            json.dump(urls, f)
        
        srcs = []
        for sol in tqdm(urls, desc=f"    -> Downloading solutions for challenge {challenge}"):
            try:
                sol['src'] = fetch_hackerrank_url(sol['download_url'])
                srcs.append(sol)
                time.sleep(1)
            except:
                tqdm.write('Error reading url ' + sol['download_url'])

        with (challenge_dir / 'full_index.json').open('w') as f:
            json.dump(srcs, f)

if __name__ == "__main__":
    main()
