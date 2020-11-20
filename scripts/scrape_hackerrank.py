import fire
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import random
import requests

from tqdm import tqdm


class HackerRankAPI:
    def __init__(self, hr_session):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.72 Safari/537.36",\
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25",\
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",\
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",\
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 Safari/537.85.10",\
            "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",\
            "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",\
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36"\
        ]
        # self.hacker_token = self.get_login_key(username, password)
        self.hr_session = hr_session
    

    @staticmethod
    def get_challenge_index_url(track, offset=0, limit=50):
        return f"https://www.hackerrank.com/rest/contests/master/tracks/{track}/challenges?offset={offset}&limit=50"
    
    @staticmethod
    def get_leaderboard_url(challenge, language=None, offset=0):
        url = f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/leaderboard?include_practice=true"
        if offset is not None:
            url += f"&offset={offset}&limit=100"
        if language is not None:
            url += f"&language={language}&filter_kinds=language"
        return url

    @staticmethod
    def get_unlock_url(challenge):
        return f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/unlock_solution"

    @staticmethod
    def get_solution_url(challenge, hacker):
        return f"https://www.hackerrank.com/rest/contests/master/challenges/{challenge}/hackers/{hacker}/download_solution"
    
    def fetch_hackerrank_url_old(self, url, json_decode=False):
        tqdm.write("Fetching URL " + url)
        time.sleep(1)
        req = urllib.request.Request(url)
        # req.add_header('Cookie', f"remember_hacker_token={self.hacker_token}; user_type=hacker")
        req.add_header('User-Agent', random.choice(self.user_agents))
        req.add_header('Cookie', "_hrank_session={self.hr_session}")
        req.add_header('referer', 'https://www.hackerrank.com/dashboard')
        req.add_header('origin', 'https://www.hackerrank.com')
        req.add_header('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9')
        req.add_header('cache-control', 'max-age=0')
        with urllib.request.urlopen(req) as url:
            data = url.read().decode()
            if json_decode:
                return json.loads(data)
            else:
                return data
    
    def fetch_hackerrank_url(self, url, json_decode=False):
        headers = {
            'user-agent': str(random.choice(self.user_agents)),
            'cookie': f'_hrank_session={self.hr_session}; hackerrank_mixpanel_token=826cd42f-383b-404d-956d-73ca0a128c7a; hrc_l_i=T; metrics_user_identifier=9af15a-f3ca15647ccd70ff12c935ba95dd57780a549a3f'
        }
        response = requests.request("GET", url, headers=headers, data={})
        print("Got", url, "with status", response.status_code)
        if json_decode:
            return response.json()
        else:
            return response.text
    
    def get_login_key(self, username, password):
        url = "https://www.hackerrank.com/rest/auth/login"
        data = {"login": username, "password": password, "remember_me": True,"fallback": False}
        req = urllib.request.Request(url, data=json.dumps(data).encode())
        req.get_method = lambda: "POST"
        req.add_header('User-Agent', random.choice(self.user_agents))
        req.add_header('referer', 'https://www.hackerrank.com/dashboard')
        req.add_header('origin', 'https://www.hackerrank.com')
        req.add_header('content-type', 'application/json')
        with urllib.request.urlopen(req) as response:
            data = response.read().decode()
            key = json.loads(data)['csrf_token']
            cookie = response.info().get_all('Set-Cookie')
            print("Got response", response, response.info(), data)
            print("Got csrf key", key)
            print("Got cookie", cookie)
            return key

    def get_challenge_list(self, track, limit=500, per_page_limit=50):
        tracks = []
        track_id_set = set()
        for i in tqdm(range(math.ceil(limit / per_page_limit)), desc=f"Get challenge list from '{track}'", leave=False):
            ntrack_count = len(track_id_set)
            url = self.get_challenge_index_url(track, per_page_limit * i, limit=per_page_limit)
            result = self.fetch_hackerrank_url(url, json_decode=True)
            for challenge in tqdm(result['models'], desc="    -> Download solution", leave=False):
                if challenge['slug'] not in track_id_set:
                    track_id_set.add(challenge['slug'])
                    tracks.append(challenge)
            if ntrack_count == len(track_id_set):  # early return
                return tracks
        return tracks
    
    def unlock_challenge(self, challenge):
        unlock_url = self.get_unlock_url(challenge)
        print("Unlocking challenge at", unlock_url)
        self.fetch_hackerrank_url(unlock_url)

    def get_challenge_submission_metadata(self, challenge, language=None, limit=50, per_page_limit=50):
        solution_urls = []
        solution_hacker_set = set()
        for i in tqdm(range(math.ceil(limit / per_page_limit)), desc=f"Load challenges from {challenge}", leave=False):
            nsol_count = len(solution_hacker_set)
            sol_url = self.get_leaderboard_url(challenge, language='javascript', offset=per_page_limit * i)
            solutions = self.fetch_hackerrank_url(sol_url, json_decode=True)
            for sol in solutions['models']:
                try:
                    if sol['hacker'] not in solution_hacker_set and sol['hacker'] != '[deleted]':
                        solution_hacker_set.add(sol['hacker'])
                        sol['download_url'] = self.get_solution_url(challenge, sol['hacker'])
                        sol['src'] = self.fetch_hackerrank_url(sol['download_url'])
                        solution_urls.append(sol)
                except Exception as e:
                    tqdm.write(f"ERROR: skipping solution set {sol}")
                    tqdm.write(str(e))
            if len(solution_hacker_set) == nsol_count:  # early return if no new solutions
                return solution_urls
        return solution_urls


def get_challenge_list(session, track='algorithms'):
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "hackerrank"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracks = HackerRankAPI(session).get_challenge_list(track)
    challenges = [x['slug'] for x in tracks]
    with (log_dir / f'{track}_challenge_index.json').open('w') as f:
        json.dump(tracks, f)
    with (log_dir / f'{track}_challenge_list.txt').open('w') as f:
        f.write('\n'.join(challenges))
    print("Got", len(challenges), "challenges")


def get_challenge_submissions(challenge, session, language='javascript', limit=10):
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "hackerrank"
    challenge_dir = (log_dir / challenge)
    challenge_dir.mkdir(parents=True, exist_ok=True)
    api = HackerRankAPI(session)
    api.unlock_challenge(challenge)
    solution_urls = api.get_challenge_submission_metadata(challenge, language, limit=limit)
    with (challenge_dir / 'challenge_data.json').open('w') as f:
        json.dump(solution_urls, f)
    
    for sol in solution_urls:
        with (challenge_dir / f"{sol['score']}_{sol['hacker_id']}.js").open('w') as f:
            f.write(sol['src'])

if __name__ == "__main__":
    fire.Fire({
        'get_challenge_list': get_challenge_list,
        'get_challenge_submissions': get_challenge_submissions,
    })
