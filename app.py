def get_video_thumbnails(youtube_url: str, max_imgs: int = 5) -> List[str]:
    # 1) Try yt_dlp first
    urls = []
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        thumbs = info.get("thumbnails") or []
        thumbs = sorted(thumbs, key=lambda t: (t.get("height", 0) * t.get("width", 0)), reverse=True)
        seen = set()
        for t in thumbs:
            u = t.get("url")
            if u and u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= max_imgs:
                break
    except Exception:
        pass

    # 2) Fallback to standard YouTube stills by video id
    try:
        vid = video_id_from_url(youtube_url)
        fallback = [
            f"https://img.youtube.com/vi/{vid}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{vid}/sddefault.jpg",
            f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
            f"https://img.youtube.com/vi/{vid}/mqdefault.jpg",
            f"https://img.youtube.com/vi/{vid}/default.jpg",
        ]
        for u in fallback:
            if u not in urls:
                urls.append(u)
            if len(urls) >= max_imgs:
                break
    except Exception:
        pass

    return urls[:max_imgs]
