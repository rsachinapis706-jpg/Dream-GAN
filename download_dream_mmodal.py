import urllib.request
import json
import os
import sys

def download_donders_dream_database(output_dir="Dream_Database_Donders"):
    """
    Downloads the 2GB 'Dream Database from Donders' from Figshare.
    This dataset contains both .edf EEG files and .docx textual dream reports.
    """
    print("Querying Figshare API for 'Dream Database from Donders'...")
    search_url = "https://api.figshare.com/v2/articles/search"
    search_data = json.dumps({"search_for": "Dream Database from Donders"}).encode('utf-8')
    
    req = urllib.request.Request(search_url, data=search_data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            results = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to query Figshare: {e}")
        return

    if not results:
        print("Dataset not found on Figshare.")
        return
        
    article = results[0]
    article_id = article['id']
    print(f"Found Article ID: {article_id} - {article['title']}")
    
    # Get download link
    files_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    try:
        with urllib.request.urlopen(files_url) as response:
            files_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to get files list: {e}")
        return
        
    if not files_data:
        print("No files attached to this article.")
        return
        
    target_file = files_data[0]
    download_url = target_file['download_url']
    file_name = target_file['name']
    file_size_mb = target_file['size'] / (1024 * 1024)
    
    print(f"\nTarget File: {file_name}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Download URL: {download_url}")
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_name)
    
    print(f"\nDownloading to {out_path}...")
    print("This is a 2GB file, this may take several minutes depending on your connection.")
    
    try:
        import requests
        
        # Check if file exists to resume
        headers = {}
        resume_byte_pos = 0
        file_mode = 'wb'
        if os.path.exists(out_path):
            resume_byte_pos = os.path.getsize(out_path)
            if resume_byte_pos > 0:
                print(f"Resuming download from byte {resume_byte_pos} ({resume_byte_pos/(1024*1024):.2f} MB)...")
                headers['Range'] = f'bytes={resume_byte_pos}-'
                file_mode = 'ab'
                
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            downloaded = resume_byte_pos
            with open(out_path, file_mode) as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024 * 1024 * 10) < 8192: # Print approximately every 10 MB
                        sys.stdout.write(f"\rDownloaded {downloaded/(1024*1024):.2f} MB / {file_size_mb:.2f} MB")
                        sys.stdout.flush()
        print(f"\n\nDownload Complete! Saved to: {out_path}")
        print("Please extract this .rar file to view the EEG and Docx files.")
    except Exception as e:
        print(f"\nDownload failed using requests: {e}")

if __name__ == "__main__":
    download_donders_dream_database()
