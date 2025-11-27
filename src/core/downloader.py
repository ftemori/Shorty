import yt_dlp
import os
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

class VideoDownloader:
    def __init__(self, download_path="downloads"):
        self.download_path = download_path
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def get_channel_videos(self, channel_url, limit=5):
        # Check if user wants specific tab content
        is_tab_specific = any(x in channel_url for x in ['/videos', '/shorts', '/live', '/streams'])
        
        # 1. Fetch Info (Flat) - Fetch more to allow filtering
        fetch_limit = limit * 4 # Fetch extra to account for shorts
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': fetch_limit,
            'extractor_args': {'youtube': {'player_client': ['default']}},
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
        except Exception:
            return []

        # CASE 1: Single Video
        # If it's a video, 'entries' might be missing, or _type is 'video'
        if 'entries' not in info or info.get('_type') == 'video':
            return [info]

        # CASE 2: Channel/Playlist
        raw_entries = info.get('entries', [])
        channel_id = info.get('channel_id')
        
        # --- Strategy A: RSS Feed (Fastest, includes dates) ---
        # Only if not tab specific (RSS mixes everything)
        if not is_tab_specific and channel_id:
            try:
                rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
                response = requests.get(rss_url, timeout=3)
                
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    ns = {
                        'atom': 'http://www.w3.org/2005/Atom',
                        'media': 'http://search.yahoo.com/mrss/',
                        'yt': 'http://www.youtube.com/xml/schemas/2015'
                    }
                    
                    videos = []
                    for entry in root.findall('atom:entry', ns):
                        if len(videos) >= limit:
                            break
                            
                        # Filter Shorts
                        webpage_url = entry.find('atom:link', ns).attrib['href']
                        if '/shorts/' in webpage_url:
                            continue
                            
                        published = entry.find('atom:published', ns).text
                        upload_date = published[:10].replace('-', '')
                        
                        video = {
                            'title': entry.find('atom:title', ns).text,
                            'webpage_url': webpage_url,
                            'id': entry.find('yt:videoId', ns).text,
                            'upload_date': upload_date,
                            'uploader': entry.find('atom:author', ns).find('atom:name', ns).text,
                            'thumbnails': [{'url': entry.find('media:group', ns).find('media:thumbnail', ns).attrib['url']}]
                        }
                        videos.append(video)
                    
                    if videos:
                        return videos
            except Exception:
                pass # Fallback to Strategy B

        # --- Strategy B: Use Flat List + Fill Metadata ---
        # Filter Shorts first
        entries = []
        for e in raw_entries:
            url = e.get('url', '') or e.get('webpage_url', '')
            if '/shorts/' in url:
                continue
            entries.append(e)
            if len(entries) >= limit:
                break
        
        if not entries:
            return []

        # Build RSS Cache for dates (if we have channel_id) to avoid slow fetches
        rss_cache = {} # id -> date
        if channel_id:
            try:
                rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
                response = requests.get(rss_url, timeout=2)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    ns = {'atom': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015'}
                    for entry in root.findall('atom:entry', ns):
                        vid_id = entry.find('yt:videoId', ns).text
                        published = entry.find('atom:published', ns).text
                        rss_cache[vid_id] = published[:10].replace('-', '')
            except:
                pass

        # Merge Data
        channel_name = info.get('uploader') or info.get('channel')
        
        videos_to_fetch = []
        for entry in entries:
            # Propagate channel name
            if not entry.get('uploader') and channel_name:
                entry['uploader'] = channel_name
            
            # Try RSS cache for date
            vid_id = entry.get('id')
            if not entry.get('upload_date') and vid_id in rss_cache:
                entry['upload_date'] = rss_cache[vid_id]
            
            # If still missing date, mark for fetch
            if not entry.get('upload_date'):
                videos_to_fetch.append(entry)

        # Fetch missing metadata in parallel
        if videos_to_fetch:
            def fetch_meta(entry):
                url = entry.get('url') or entry.get('webpage_url')
                if not url: return entry
                opts = {'quiet': True, 'no_warnings': True}
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        return ydl.extract_info(url, download=False)
                except:
                    return entry

            with ThreadPoolExecutor(max_workers=len(videos_to_fetch)) as executor:
                future_to_entry = {executor.submit(fetch_meta, e): e for e in videos_to_fetch}
                fetched_map = {}
                for future in future_to_entry:
                    try:
                        res = future.result()
                        fetched_map[res.get('id')] = res
                    except:
                        pass
                
                for i, entry in enumerate(entries):
                    if entry.get('id') in fetched_map:
                        entries[i] = fetched_map[entry.get('id')]

        return entries

    def download_video(self, video_url, progress_hook=None):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),
            'progress_hooks': [progress_hook] if progress_hook else [],
            'extractor_args': {'youtube': {'player_client': ['default']}},
            'no_warnings': True,
            'quiet': True,
            'external_downloader_args': ['-loglevel', 'panic'],
            'postprocessor_args': {'ffmpeg': ['-loglevel', 'panic']},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            return ydl.prepare_filename(info)

    def get_video_info(self, video_url):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['default']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=False)
